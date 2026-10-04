"""Serve the local graphical plan editor; no third-party packages required."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
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
from arksim.cli import DEFAULT_DATA_DIR


class EditorServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple, data_dir: Path, output_root: Path):
        self.data = EditorData(data_dir)
        self.output_root = output_root.resolve()
        self.jobs: dict[str, dict] = {}
        self.saved: dict[str, Path] = {}
        self.job_lock = threading.Lock()
        super().__init__(address, EditorHandler)

    def save(self, compiled: dict) -> dict:
        key = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_") + uuid4().hex[:8]
        folder = self.output_root / "plans" / "editor" / key
        folder.mkdir(parents=True, exist_ok=False)
        for name, value in (("project.json", compiled["project"]), ("plan.json", compiled["plan"])):
            (folder / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
        with self.job_lock:
            self.saved[key] = folder
        return {"folder": str(folder), "projectFile": str(folder / "project.json"),
                "planFile": str(folder / "plan.json"),
                "projectUrl": f"/api/saved?save={key}&name=project.json",
                "planUrl": f"/api/saved?save={key}&name=plan.json"}

    def simulate(self, compiled: dict) -> dict:
        if not compiled["plan"]:
            raise ValueError("请先添加至少一条操作")
        with self.job_lock:
            if any(j["status"] == "running" for j in self.jobs.values()):
                raise ValueError("已有模拟正在运行，请等待完成")
            job_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_") + uuid4().hex[:8]
            folder = self.output_root / "editor" / "runs" / job_id
            folder.mkdir(parents=True, exist_ok=False)
            for name, value in (("plan.json", compiled["plan"]), ("project.json", compiled["project"])):
                (folder / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
            job = {"id": job_id, "status": "running", "message": "正在模拟…"}
            self.jobs[job_id] = job
            threading.Thread(target=self._run, args=(job_id, folder, compiled["project"]), daemon=True).start()
            return dict(job)

    def _run(self, job_id: str, folder: Path, project: dict) -> None:
        opts = project["options"]
        command = [sys.executable, str(ROOT / "tools/simulate.py"), "--data-dir", str(self.data.data_dir),
                   "--stage", project["stage"], "--plan", str(folder / "plan.json"),
                   "--spawn-timing", opts["spawn_timing"], "--enemy-attack-timing", opts["enemy_attack_timing"],
                   "--seed", str(opts["seed"]), "--output-dir", str(self.output_root)]
        for key in ("enemy_muzzle", "enemy_turning"):
            if opts[key]:
                command.append("--" + key.replace("_", "-"))
        try:
            result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
                                    errors="replace", creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            (folder / "run.log").write_text(result.stdout + result.stderr, encoding="utf-8")
            if result.returncode:
                raise ValueError((result.stderr or result.stdout or "模拟失败")[-4000:])
            latest = self.output_root / "最新模拟"
            # Preserve this job's replay even when another job rotates latest.
            shutil.copyfile(latest / "replay.json", folder / "replay.json")
            summary = json.loads((latest / "result.json").read_text(encoding="utf-8"))
            update = {"status": "completed", "message": "模拟完成", "result": summary,
                      "replayUrl": f"/api/replay?job={job_id}",
                      "viewerUrl": f"/index.html?job={job_id}"}
        except (OSError, ValueError) as error:
            update = {"status": "failed", "message": str(error)}
        with self.job_lock:
            self.jobs[job_id].update(update)


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
                return self.respond(self.server.data.stage(params.get("id", [""])[0]))
            if url.path == "/api/operator":
                return self.respond(self.server.data.operator(params.get("id", [""])[0]))
            if url.path == "/api/saved":
                key, name = params.get("save", [""])[0], params.get("name", [""])[0]
                with self.server.job_lock:
                    folder = self.server.saved.get(key)
                if folder is None or name not in ("project.json", "plan.json"):
                    return self.respond({"error": "找不到已保存方案"}, 404)
                return self.send_file(folder / name, "application/json; charset=utf-8", name)
            if url.path in ("/api/job", "/api/replay"):
                key = params.get("job", [""])[0]
                with self.server.job_lock:
                    job = dict(self.server.jobs.get(key, {}))
                if not job:
                    return self.respond({"error": "找不到本次运行，请回编辑页重新生成或读取本地回放"}, 404)
                if url.path == "/api/job":
                    return self.respond(job)
                if job["status"] != "completed":
                    return self.respond({"error": "模拟尚未完成"}, 409)
                path = self.server.output_root / "editor" / "runs" / key / "replay.json"
                return self.send_file(path, "application/json; charset=utf-8")
            files = {"/": "planner.html", "/planner.html": "planner.html", "/planner.css": "planner.css",
                     "/planner.js": "planner.js", "/index.html": "index.html"}
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
            if not 0 < length <= 2_000_000:
                raise ValueError("请求为空或超过2MB")
            payload = json.loads(self.rfile.read(length))
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
    print(f"计划编辑器：http://127.0.0.1:{server.server_port}/planner.html", flush=True)
    print("保持此窗口运行；结束时按 Ctrl+C。", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
