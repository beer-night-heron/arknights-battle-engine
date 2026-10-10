"""Temporary ground navigation for fear and placed lures.

The cardinal-cell route is a candidate approximation of client navigation;
the normal route/checkpoint index is never overwritten by a temporary route.
"""
from collections import deque
import math


def paths_from(battle, start, limit=10):
    paths = {start: [start]}
    queue = deque([start])
    while queue:
        row, col = queue.popleft()
        if len(paths[(row, col)]) - 1 >= limit:
            continue
        for cell in ((row + 1, col), (row, col + 1), (row - 1, col), (row, col - 1)):
            if cell not in paths and battle._ground_passable_cell(*cell):
                paths[cell] = paths[(row, col)] + [cell]
                queue.append(cell)
    return paths


def temporary_move(battle, enemy, destination=None, *, source=None):
    state = getattr(enemy, "induced_path", None)
    signature = (destination, source)
    if state is None or state[0] != signature or not state[1]:
        start = enemy.tile()
        paths = paths_from(battle, start)
        if destination is None:
            origin = source if source is not None else enemy.position()
            vector = (enemy.position_row - origin[0], enemy.position_col - origin[1])
            candidates = []
            for cell, path in paths.items():
                delta = (cell[0] - enemy.position_row, cell[1] - enemy.position_col)
                distance = math.hypot(*delta)
                length = math.hypot(*vector)
                if distance <= 10 and len(path) - 1 <= 5 and distance > 0 and length > 0:
                    cosine = (delta[0] * vector[0] + delta[1] * vector[1]) / (distance * length)
                    if cosine >= math.cos(math.pi / 4):
                        candidates.append(cell)
            destination = battle.choose_random_target(sorted(candidates), mechanism="fear_destination", source=enemy) if candidates else start
            offset = (battle.random_uniform(-.25, .25, mechanism="fear_offset_row", source=enemy),
                      battle.random_uniform(-.25, .25, mechanism="fear_offset_col", source=enemy))
        else:
            offset = (0, 0)
        route = [tuple(map(float, cell)) for cell in paths.get(destination, [start])[1:]]
        route.append((destination[0] + offset[0], destination[1] + offset[1]))
        state = (signature, route)
        enemy.induced_path = state
    route = state[1]
    budget = battle._effective_enemy_move_speed(enemy) * battle.dt
    while route and budget > 0:
        goal = route[0]
        delta = (goal[0] - enemy.position_row, goal[1] - enemy.position_col)
        distance = math.hypot(*delta)
        if distance <= budget + 1e-9:
            enemy.position_row, enemy.position_col = goal
            budget -= distance
            route.pop(0)
        else:
            enemy.position_row += delta[0] / distance * budget
            enemy.position_col += delta[1] / distance * budget
            budget = 0
    return not route
