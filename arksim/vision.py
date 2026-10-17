"""Source-owned blind tiles; centre-ray geometry remains a candidate."""
from __future__ import annotations

SIGHT_BLOCKERS = frozenset({"trap_218_fttree", "trap_219_fttreant", "trap_220_ftbone"})


def segment_crosses_tile(start, end, cell):
    lo, hi = 0.0, 1.0
    for axis in (0, 1):
        delta = end[axis] - start[axis]
        lower, upper = cell[axis] - .5, cell[axis] + .5
        if abs(delta) <= 1e-12:
            if not lower < start[axis] < upper:
                return False
            continue
        a, z = (lower - start[axis]) / delta, (upper - start[axis]) / delta
        lo, hi = max(lo, min(a, z)), min(hi, max(a, z))
        if hi - lo <= 1e-12:
            return False
    return hi > 0 and lo < 1


def blind_tile(battle, source, destination):
    start = source.tile
    live = [unit for unit in (*battle.operators, *battle.enemies)
            if getattr(unit, "char_id", getattr(unit, "key", None)) in SIGHT_BLOCKERS]
    blockers = {unit.tile() if callable(unit.tile) else unit.tile for unit in live
                if not unit.dead and not getattr(unit, "leaked", False) and not getattr(unit, "disappeared", False)}
    # Some stage devices are not instantiated by the current stage adapters.
    # Preserve their declared initial obstruction, with an explicit limitation.
    represented = {unit.tile() if callable(unit.tile) else unit.tile for unit in live}
    for inst in battle.level.get("predefines", {}).get("tokenInsts", []):
        if inst.get("inst", {}).get("characterKey") not in SIGHT_BLOCKERS:
            continue
        pos = inst["position"];cell = (pos["row"], pos["col"])
        if cell not in represented:
            blockers.add(cell)
            battle.mechanic_diagnostics["VISION_PREDEFINED_DEVICE_LIFECYCLE_UNIMPLEMENTED"] = 1
    if blockers:
        battle.mechanic_diagnostics["VISION_CENTER_RAY_GEOMETRY_CANDIDATE"] = 1
    return any(cell not in {start, destination} and segment_crosses_tile(start, destination, cell)
               for cell in blockers)
