"""WD-EX-3's single-shot anti-armor infantry, scoped to this stage."""
from .mechanics import ApplyWay, DamageType

STAGES = {"act18d0_ex03", "act18d0_ex03#f#"}
RPG = "enemy_1143_merrpg"


class WdExRuntime:
    def __init__(self, battle):
        self.b = battle
        self.enabled = battle.level.get("wdExStage") in STAGES
        self.states = {}
        self.events = []
        if self.enabled:
            battle.mechanic_diagnostics["WD_RPG_SPELL_NOTIFICATION_AND_MODE_HANDOFF_CANDIDATE"] = 1
            battle.mechanic_diagnostics["WD_RPG_PROJECTILE_COLLISION_PHASE_CANDIDATE"] = 1
            battle.mechanic_diagnostics["WD_RPG_ANIMATION_STATE_BINDING_CANDIDATE"] = 1

    def emit(self, kind, enemy, **details):
        self.events.append(dict(t=round(self.b.time, 4), frame=self.b.frame_index,
                                kind=kind, enemy=enemy.enemy_index, **details))

    def born(self, enemy, data):
        if not self.enabled or enemy.key != RPG:
            return
        board = {v["key"]: v["value"] for v in data.get("talentBlackboard") or []}
        self.states[enemy.enemy_index] = dict(entity=enemy, fired=False, switch_at=None,
            base_speed=enemy.move_speed, board=board)
        enemy.apply_way = ApplyWay.RANGED.value
        self.set_animation(enemy, "Attack")

    def set_animation(self, enemy, name):
        animation = self.b.attack_timing.get(RPG, {}).get("wdexAnimations", {}).get(name)
        if animation:
            enemy.attack_duration = animation["duration"]
            enemy.attack_hit_time = next(v["time"] for v in animation["events"] if v["name"] == "OnAttack")
            enemy.wait_for_attack_event = True
        elif name == "Attack_2":
            self.b.mechanic_diagnostics["WD_RPG_MELEE_ANIMATION_FALLBACK_CANDIDATE"] = 1

    def before_tick(self, enemy):
        state = self.states.get(enemy.enemy_index)
        if not state or state["switch_at"] is None or self.b.time + 1e-9 < state["switch_at"]:
            return
        state["switch_at"] = None
        enemy.mode_index = 1
        enemy.apply_way = ApplyWay.MELEE.value
        enemy.range_radius = -1.
        # FormulaItem MULTIPLIER: +200% of the base MOVE_SPEED. The ordinary
        # stage multiplier and control effects are still applied by Battle.
        enemy.move_speed = state["base_speed"] * (1 + state["board"]["MoveSpeedUp.move_speed"])
        enemy.attack_target = None
        enemy.attack_hit_frame = None
        enemy.cached_operator_target = None
        enemy.attack_cd = 0.
        enemy.attack_cd_frames = 0
        enemy.attack_animation_end_frame = None
        enemy.attack_rearm_frame = None
        enemy.attack_move_resume_frame = None
        enemy.pause_until = self.b.time
        self.set_animation(enemy, "Attack_2")
        self.emit("wd_rpg_mode_switch", enemy, mode=1, moveSpeed=enemy.move_speed)

    def fire_attack(self, enemy, target, *, will_hit):
        state = self.states.get(enemy.enemy_index)
        if not state:
            return False
        ranged = enemy.mode_index == 0
        if ranged:
            if state["fired"]:
                return True
            state["fired"] = True
            # AttachBuffOnEvent ON_SPELL_ON -> delayed one-shot mode buff.
            state["switch_at"] = self.b.time + 1.1399999856948853
            self.emit("wd_rpg_grenade", enemy, target=target.deployment_index,
                      switchAt=round(state["switch_at"], 4))
        context = dict(projectile=enemy.projectile_key if ranged else None,
                       damage_type=DamageType.PHYSICAL,
                       apply_way=ApplyWay.RANGED if ranged else ApplyWay.MELEE,
                       penetration=0., skill=None,
                       wdex_scale=1. if ranged else state["board"]["Combat.attack@atk_scale"])
        self.b._fire_single_enemy_attack(enemy, target, will_hit=will_hit, attack_context=context)
        return True

    def attack_scale(self, enemy, context):
        if enemy.enemy_index not in self.states:
            return 1.
        if context is not None and "wdex_scale" in context:
            return context["wdex_scale"]
        state = self.states[enemy.enemy_index]
        return state["board"]["Combat.attack@atk_scale"] if enemy.mode_index == 1 else 1.

    def projectile_lifecycle(self, key):
        if self.enabled and key == "projectile_enemy_merrpg":
            return dict(lifetime=10., force_reached_when_timeup=True,
                        always_hit_in_end=True, stop_when_source_invalid=False)
        return {}

    def snapshot(self):
        return {str(k): dict(mode=s["entity"].mode_index, fired=s["fired"], switchAt=s["switch_at"])
                for k, s in self.states.items() if not s["entity"].dead and not s["entity"].leaked}
