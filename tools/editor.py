"""Serve the local graphical plan editor; no third-party packages required."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from copy import deepcopy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import threading
from urllib.parse import parse_qs, urlparse
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from arksim import __version__
from arksim.editor import EditorData
from arksim.cli import DEFAULT_DATA_DIR, resolve_level_file
from arksim.replay_edit import frame_number, ordered, validate_edit, runtime_signature


class EditorServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple, data_dir: Path, output_root: Path):
        self.data = EditorData(data_dir)
        self.output_root = output_root.resolve()
        self.jobs: dict[str, dict] = {}
        self.saved: dict[str, Path] = {}
        self.draft_lock = threading.Lock()
        self.job_lock = threading.Lock()
        super().__init__(address, EditorHandler)

    def job_folder(self, key: str) -> Path:
        if not isinstance(key, str) or not re.fullmatch(r"\d{8}_\d{6}_[0-9a-f]{8}", key):
            raise ValueError("运行编号无效")
        folder = self.output_root / "editor" / "runs" / key
        if not folder.resolve().is_relative_to(self.output_root) or folder.is_symlink():
            raise ValueError("运行目录无效")
        return folder

    def get_job(self, key: str) -> dict:
        with self.job_lock:
            if key in self.jobs:
                return dict(self.jobs[key])
        folder = self.job_folder(key)
        if (folder / "job.json").is_file():
            return json.loads((folder / "job.json").read_text(encoding="utf-8"))
        if (folder / "replay.json").is_file():
            replay = json.loads((folder / "replay.json").read_text(encoding="utf-8"))
            return {"id": key, "status": "completed", "result": replay["cases"][0]["result"],
                    "replayUrl": f"/api/replay?job={key}", "viewerUrl": f"/index.html?job={key}"}
        raise ValueError("找不到本次运行；也可以打开本地回放文件")

    def prepare_branch(self, payload: dict) -> dict:
        job_id = payload.get("job")
        if job_id:
            if self.get_job(job_id)["status"] != "completed":
                raise ValueError("原模拟尚未完成")
            folder = self.job_folder(job_id)
            replay = json.loads((folder / "replay.json").read_text(encoding="utf-8"))
            fallback = json.loads((folder / "project.json").read_text(encoding="utf-8"))
        else:
            replay, fallback = payload.get("replay"), payload.get("project")
        if not isinstance(replay, dict) or replay.get("engineVersion") != __version__:
            raise ValueError("此回放的引擎版本不同，请先用当前版本运行原方案，再从回放续编")
        cases = replay.get("cases")
        case_index = payload.get("case", 0)
        if not isinstance(cases, list) or type(case_index) is not int or not 0 <= case_index < len(cases):
            raise ValueError("回放方案无效")
        case = cases[case_index]
        if not isinstance(case, dict):
            raise ValueError("回放方案无效")
        if case.get("randomDecisions", {}).get("outcome_replay"):
            raise ValueError("此回放指定了随机结果，暂不支持续编；请先正常运行原方案")
        frame = payload.get("frame")
        frames = case.get("frames", [])
        numbers = [frame_number(item) for item in frames]
        if (type(frame) is not int or frame not in numbers or frame >= numbers[-1]
                or any(a >= b for a, b in zip(numbers, numbers[1:]))):
            raise ValueError("请在战斗结束前选择一个有效回放帧")
        context = deepcopy(replay.get("editorContext") or {"project": fallback, "edits": []})
        if not context.get("project"):
            raise ValueError("此回放没有完整方案，请补选对应的project.json")
        original = self.data.compile(context["project"])["project"]
        context["project"] = original
        retained, previous, at_cut = [], -1, original
        for change in context.get("edits", []):
            if type(change.get("frame")) is not int or change["frame"] <= previous:
                raise ValueError("回放续编记录无效")
            previous = change["frame"]
            compiled = self.data.compile(change["project"])["project"]
            validate_edit(original, compiled, change["frame"])
            original = compiled
            if change["frame"] < frame:
                retained.append({"frame": change["frame"], "project": compiled})
                at_cut = compiled
        context["edits"] = retained
        validate_edit(at_cut, original, frame)
        if replay.get("stage") != original["stage"]:
            raise ValueError("回放与方案关卡不一致")
        seed = case.get("result", {}).get("seed")
        if seed is not None and seed != original["options"]["seed"]:
            raise ValueError("回放与原方案的随机种子不一致")
        for field, option in (("spawnTiming", "spawn_timing"), ("enemyAttackTiming", "enemy_attack_timing"),
                              ("enemyMuzzle", "enemy_muzzle"), ("enemyTurning", "enemy_turning"),
                              ("optionalRunes", "optional_runes"), ("battleBuffs", "battle_buffs")):
            recorded = replay.get(field, case.get(field))
            if recorded is not None and recorded != original["options"][option]:
                raise ValueError("回放与原方案的模拟选项不一致")
        if case.get("squad") is not None and set(case["squad"]) != {item["char_id"] for item in original["squad"]}:
            raise ValueError("回放与原方案的携带编队不一致")
        if (replay.get("runtimeSignature") and replay["runtimeSignature"] != runtime_signature(
                self.data.data_dir, resolve_level_file(self.data.stages, original["stage"]))):
            raise ValueError("本地运行数据或假设参数已改变，请先重新运行原方案")
        project = deepcopy(original)
        project["operations"] = ordered(project["operations"])
        key = uuid4().hex
        folder = self.output_root / "editor" / "branches" / key
        folder.mkdir(parents=True, exist_ok=False)
        source = {**replay, "cases": [case]}
        info = {"id": key, "frame": frame, "time": frame / 30, "project": project, "context": context}
        (folder / "source.json").write_text(json.dumps(source, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        (folder / "branch.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
        return {k: info[k] for k in ("id", "frame", "time", "project")}

    def branch_info(self, key: str) -> tuple[Path, dict]:
        if not isinstance(key, str) or not re.fullmatch(r"[0-9a-f]{32}", key):
            raise ValueError("续编编号无效")
        folder = self.output_root / "editor" / "branches" / key
        if not folder.resolve().is_relative_to(self.output_root) or folder.is_symlink():
            raise ValueError("续编目录无效")
        return folder, json.loads((folder / "branch.json").read_text(encoding="utf-8"))

    def simulate_branch(self, payload: dict) -> dict:
        folder, info = self.branch_info(payload["branch"])
        compiled = self.data.compile(payload["project"])
        validate_edit(info["project"], compiled["project"], info["frame"])
        context = deepcopy(info["context"])
        context["edits"].append({"frame": info["frame"], "project": compiled["project"]})
        return self.simulate(compiled, continuation={"context": context, "source": folder / "source.json"})

    def latest_draft(self, session: str = "") -> dict | None:
        if session and not re.fullmatch(r"[0-9a-f]{32}", session):
            raise ValueError("草稿会话编号无效")
        folder = self.output_root / "editor" / "drafts"
        files = sorted(folder.glob("*.json"), reverse=True)
        own = [p for p in files if f"_{session}_" in p.name] if session else []
        for path in own + [p for p in files if p not in own]:
            try:
                draft = json.loads(path.read_text(encoding="utf-8"))
                self.validate_draft(draft)
                if not isinstance(draft.get("savedAt"), str):
                    continue
                return draft
            except (OSError, ValueError, TypeError):
                continue
        return None

    @staticmethod
    def validate_draft(draft: dict) -> None:
        if not isinstance(draft, dict) or not isinstance(draft.get("session"), str) or not re.fullmatch(r"[0-9a-f]{32}", draft["session"]):
            raise ValueError("草稿会话编号无效")
        if type(draft.get("revision")) is not int or draft["revision"] < 1:
            raise ValueError("草稿修订编号无效")
        snapshot = draft.get("snapshot")
        if not isinstance(snapshot, dict) or snapshot.get("version") != 1:
            raise ValueError("草稿格式无效")
        project = snapshot.get("project")
        if (not isinstance(project, dict) or project.get("format") != "arksim-plan-editor"
                or type(project.get("version")) is not int or project["version"] != 1
                or not isinstance(project.get("stage"), str)
                or not isinstance(project.get("options"), dict)
                or not isinstance(project.get("squad"), list) or len(project["squad"]) > 100
                or not isinstance(project.get("operations"), list) or len(project["operations"]) > 10000
                or any(not isinstance(item, dict) for item in project["squad"] + project["operations"])
                or not isinstance(snapshot.get("ui"), dict)):
            raise ValueError("草稿方案格式无效")
        ui = snapshot["ui"]
        if (ui.get("step") not in ("setup", "operations")
                or ui.get("selectedBuild") is not None and not isinstance(ui["selectedBuild"], str)
                or any(not isinstance(ui.get(k), dict) for k in ("build", "operation", "options"))
                or any(type(ui.get(k)) is not bool for k in ("dirty", "buildDirty", "operationDirty"))
                or type(ui.get("facing")) is not int or not 0 <= ui["facing"] <= 3
                or type(ui.get("preview")) not in (int, float) or not 0 <= ui["preview"] < float("inf")
                or ui.get("editIndex") is not None and (type(ui["editIndex"]) is not int or not 0 <= ui["editIndex"] < len(project["operations"]))
                or ui.get("tile") is not None and (not isinstance(ui["tile"], list) or len(ui["tile"]) != 2 or any(type(v) is not int or v < 0 for v in ui["tile"]))):
            raise ValueError("草稿表单格式无效")

    def save_draft(self, draft: dict) -> dict:
        self.validate_draft(draft)
        with self.draft_lock:
            previous = self.latest_draft(draft["session"])
            if previous and previous["session"] == draft["session"] and previous["revision"] >= draft["revision"]:
                return {"revision": previous["revision"], "savedAt": previous["savedAt"]}
            saved_at = datetime.now(timezone.utc)
            folder = self.output_root / "editor" / "drafts"
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / (saved_at.strftime("%Y%m%d_%H%M%S_%f_") + draft["session"] + "_" + uuid4().hex[:8] + ".json")
            value = {"session": draft["session"], "revision": draft["revision"],
                     "savedAt": saved_at.isoformat(), "snapshot": draft["snapshot"]}
            encoded = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
            with path.open("x", encoding="utf-8") as stream:
                stream.write(encoded)
            return {"revision": value["revision"], "savedAt": value["savedAt"]}

    def save(self, compiled: dict) -> dict:
        key = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_") + uuid4().hex[:8]
        folder = self.output_root / "plans" / "editor" / key
        folder.mkdir(parents=True, exist_ok=False)
        for name, value in (("project.json", compiled["project"]), ("plan.json", compiled["plan"]),
                            ("squad.json", compiled["project"]["squad"])):
            (folder / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
        with self.job_lock:
            self.saved[key] = folder
        return {"folder": str(folder), "projectFile": str(folder / "project.json"),
                "planFile": str(folder / "plan.json"),
                "squadFile": str(folder / "squad.json"),
                "projectUrl": f"/api/saved?save={key}&name=project.json",
                "planUrl": f"/api/saved?save={key}&name=plan.json"}

    def simulate(self, compiled: dict, continuation: dict | None = None) -> dict:
        if not compiled["plan"]:
            raise ValueError("请先添加至少一条操作")
        root = self.data.compile(continuation["context"]["project"]) if continuation else None
        with self.job_lock:
            if any(j["status"] == "running" for j in self.jobs.values()):
                raise ValueError("已有模拟正在运行，请等待完成")
            job_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_") + uuid4().hex[:8]
            folder = self.output_root / "editor" / "runs" / job_id
            folder.mkdir(parents=True, exist_ok=False)
            for name, value in (("plan.json", compiled["plan"]), ("project.json", compiled["project"]),
                                ("squad.json", compiled["project"]["squad"])):
                (folder / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
            job = {"id": job_id, "status": "running", "message": "正在模拟…"}
            self.jobs[job_id] = job
            if continuation:
                context = continuation["context"]
                (folder / "original-plan.json").write_text(json.dumps(root["plan"], ensure_ascii=False), encoding="utf-8")
                (folder / "continuation.json").write_text(json.dumps(context, ensure_ascii=False), encoding="utf-8")
                shutil.copyfile(continuation["source"], folder / "source-replay.json")
            threading.Thread(target=self._run, args=(job_id, folder, compiled["project"], bool(continuation)), daemon=True).start()
            return dict(job)

    def _run(self, job_id: str, folder: Path, project: dict, continuation: bool = False) -> None:
        opts = project["options"]
        command = [sys.executable, "-B", str(ROOT / "tools/simulate.py"), "--data-dir", str(self.data.data_dir),
                   "--stage", project["stage"], "--plan", str(folder / ("original-plan.json" if continuation else "plan.json")),
                   "--squad", str(folder / "squad.json"),
                   "--spawn-timing", opts["spawn_timing"], "--enemy-attack-timing", opts["enemy_attack_timing"],
                   "--seed", str(opts["seed"]), "--output-dir", str(folder / "output")]
        if continuation:
            source = json.loads((folder / "source-replay.json").read_text(encoding="utf-8"))
            command += ["--continuation", str(folder / "continuation.json"), "--source-replay", str(folder / "source-replay.json"),
                        "--sample-interval", str(source["cases"][0].get("sampleInterval", 1 / 30))]
        else:
            command += ["--editor-project", str(folder / "project.json")]
        for key in ("enemy_muzzle", "enemy_turning"):
            if opts[key]:
                command.append("--" + key.replace("_", "-"))
        for rune in opts.get("optional_runes", []):
            command.extend(["--rune", rune])
        for supply in opts.get("battle_buffs", []):
            command.extend(["--buff", supply])
        try:
            result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
                                    errors="replace", creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            (folder / "run.log").write_text(result.stdout + result.stderr, encoding="utf-8")
            if result.returncode:
                raise ValueError((result.stderr or result.stdout or "模拟失败")[-4000:])
            latest = folder / "output" / "最新模拟"
            # Preserve this job's replay even when another job rotates latest.
            shutil.copyfile(latest / "replay.json", folder / "replay.json")
            from tools.simulate import promote
            history = self.output_root / "模拟历史"
            if history.is_symlink() or not history.resolve().is_relative_to(self.output_root):
                raise ValueError("历史目录超出输出目录")
            history.mkdir(parents=True, exist_ok=True)
            staged = history / ("_编辑器_" + job_id)
            shutil.copytree(latest, staged)
            promote(staged, self.output_root)
            summary = json.loads((latest / "result.json").read_text(encoding="utf-8"))
            update = {"status": "completed", "message": "模拟完成", "result": summary,
                      "replayUrl": f"/api/replay?job={job_id}",
                      "viewerUrl": f"/index.html?job={job_id}"}
        except (OSError, ValueError) as error:
            update = {"status": "failed", "message": str(error)}
        with self.job_lock:
            self.jobs[job_id].update(update)
            (folder / "job.json").write_text(json.dumps(self.jobs[job_id], ensure_ascii=False), encoding="utf-8")


class EditorHandler(BaseHTTPRequestHandler):
    server: EditorServer

    def log_message(self, *_args) -> None:
        pass

    def respond(self, payload, status=200) -> None:
        body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def allowed_request(self) -> bool:
        host = f"127.0.0.1:{self.server.server_port}"
        if self.headers.get("Host") not in (host, f"localhost:{self.server.server_port}"):
            self.respond({"error": "仅允许本机访问"}, 403)
            return False
        origin = self.headers.get("Origin")
        if origin and origin not in (f"http://{host}", f"http://localhost:{self.server.server_port}"):
            self.respond({"error": "不允许其他网页发起请求"}, 403)
            return False
        return True

    def do_GET(self) -> None:
        if not self.allowed_request():
            return
        url = urlparse(self.path)
        params = parse_qs(url.query)
        try:
            if url.path == "/api/catalog":
                return self.respond({**self.server.data.catalog(), "engineVersion": __version__})
            if url.path == "/api/stage":
                return self.respond(self.server.data.stage(params.get("id", [""])[0], params.get("rune", []), params.get("buff", [])))
            if url.path == "/api/operator":
                return self.respond(self.server.data.operator(params.get("id", [""])[0]))
            if url.path == "/api/draft":
                with self.server.draft_lock:
                    draft = self.server.latest_draft(params.get("session", [""])[0])
                return self.respond({"draft": draft})
            if url.path == "/api/saved":
                key, name = params.get("save", [""])[0], params.get("name", [""])[0]
                with self.server.job_lock:
                    folder = self.server.saved.get(key)
                if folder is None or name not in ("project.json", "plan.json"):
                    return self.respond({"error": "找不到已保存方案"}, 404)
                return self.send_file(folder / name, "application/json; charset=utf-8", name)
            if url.path in ("/api/job", "/api/replay"):
                key = params.get("job", [""])[0]
                try:
                    job = self.server.get_job(key)
                except ValueError:
                    return self.respond({"error": "找不到本次运行，请回编辑页重新生成或读取本地回放"}, 404)
                if url.path == "/api/job":
                    return self.respond(job)
                if job["status"] != "completed":
                    return self.respond({"error": "模拟尚未完成"}, 409)
                path = self.server.job_folder(key) / "replay.json"
                return self.send_file(path, "application/json; charset=utf-8")
            files = {"/": "index.html", "/planner.html": "planner.html", "/planner.css": "planner.css",
                     "/planner.js": "planner.js", "/index.html": "index.html",
                     "/workspace.js": "workspace.js", "/workspace.css": "workspace.css"}
            if url.path not in files:
                return self.respond({"error": "找不到页面"}, 404)
            path = ROOT / "viewer" / files[url.path]
            mime = {".html": "text/html", ".js": "text/javascript", ".css": "text/css"}[path.suffix]
            self.send_file(path, mime + "; charset=utf-8")
        except (KeyError, OSError, ValueError) as error:
            self.respond({"error": str(error)}, 400)

    def send_file(self, path: Path, mime: str, download_name: str | None = None) -> None:
        size = path.stat().st_size
        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Length", str(size))
        if download_name:
            self.send_header("Content-Disposition", f'attachment; filename="{download_name}"')
        self.end_headers()
        with path.open("rb") as stream:
            shutil.copyfileobj(stream, self.wfile)

    def do_POST(self) -> None:
        if not self.allowed_request():
            return
        try:
            if self.headers.get_content_type() != "application/json":
                raise ValueError("请求须使用JSON")
            length = int(self.headers.get("Content-Length", "0"))
            limit = 128_000_000 if self.path == "/api/branch/prepare" else 2_000_000
            if not 0 < length <= limit:
                raise ValueError("请求为空或超过大小限制")
            payload = json.loads(self.rfile.read(length))
            if self.path == "/api/branch/prepare":
                return self.respond(self.server.prepare_branch(payload))
            if self.path == "/api/branch/simulate":
                return self.respond(self.server.simulate_branch(payload))
            if self.path == "/api/draft":
                return self.respond(self.server.save_draft(payload))
            if self.path == "/api/build":
                config, summary = self.server.data.build(payload)
                return self.respond({"config": config, "summary": summary})
            if self.path == "/api/import-plan":
                return self.respond(self.server.data.import_plan(payload["plan"], payload["stage"]))
            if self.path in ("/api/validate", "/api/simulate", "/api/save"):
                compiled = self.server.data.compile(payload)
                if self.path == "/api/save":
                    return self.respond(self.server.save(compiled))
                return self.respond(compiled if self.path == "/api/validate" else self.server.simulate(compiled))
            self.respond({"error": "找不到接口"}, 404)
        except (OSError, ValueError, KeyError, TypeError) as error:
            self.respond({"error": str(error)}, 400)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--port", type=int, default=8873)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "local")
    args = parser.parse_args()
    if not 0 <= args.port <= 65535:
        parser.error("port must be 0..65535")
    try:
        server = EditorServer(("127.0.0.1", args.port), args.data_dir, args.output_dir)
    except (OSError, ValueError, KeyError) as error:
        parser.error(f"无法启动编辑器：{error}")
    print(f"战术工作台：http://127.0.0.1:{server.server_port}/index.html", flush=True)
    print("保持此窗口运行；结束时按 Ctrl+C。", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
