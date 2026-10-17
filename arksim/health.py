"""Ordinary HP and opt-in negative HP, shared by modifier and recovery paths."""
from __future__ import annotations


def minus_hp_limit(actor):
    return max(0.0, float(getattr(actor, "max_minus_hp_ratio", 0))) * actor.max_hp


def extra_hp_limit(actor):
    return max(0.0, float(getattr(actor, "max_extra_hp_ratio", 0))) * actor.max_hp


def set_hp(actor, value, *, force=False):
    """Reset the regular bar; return whether the active HP pool is exhausted."""
    limit = minus_hp_limit(actor)
    from .mechanics import Status
    statuses = getattr(actor, "statuses", None)
    undead = bool(statuses and statuses.has(Status.UNDEADABLE))
    if force or limit <= 0:
        actor.hp = min(actor.max_hp, max(1.0 if undead else 0.0, value))
        return actor.hp <= 0
    current = actor.hp
    debt = max(0.0, float(getattr(actor, "minus_hp", 0)))
    if value > current:
        paid = min(debt, value - current)
        debt -= paid
        value -= paid
    elif value <= 0:
        debt = (debt if current <= 1 else 0.0) + max(0.0, -value)
        value = 1.0
    actor.minus_hp = min(debt, max(0.0, limit - 1.0) if undead else limit)
    actor.hp = min(actor.max_hp, max(1.0, value))
    actor.negative_hp_heal_free = actor.minus_hp > 0
    return not undead and debt >= limit


def gain_hp(actor, amount, *, packet=None):
    if actor.dead or amount <= 0:
        return 0.0
    received = min(amount, max(0.0, actor.max_hp - actor.hp)
                   + max(0.0, float(getattr(actor, "minus_hp", 0))))
    exhausted = set_hp(actor, actor.hp + received)
    if exhausted:
        handler = getattr(actor, "lethal_handler", None)
        if callable(handler):
            handler(packet)
        actor.hp = 0.0
        actor.dead = True
    elif packet is not None and getattr(packet, "heal_can_general_shield", False):
        stored = min(max(0.0, amount - received),
                     max(0.0, extra_hp_limit(actor) - getattr(actor, "extra_hp", 0)))
        if stored > 0:
            actor.extra_hp = getattr(actor, "extra_hp", 0) + stored
            received += stored
    return received


def lose_hp(actor, amount):
    """Only modifier application consumes the extra pool, before regular HP."""
    extra = max(0.0, float(getattr(actor, "extra_hp", 0)))
    spent = min(extra, max(0.0, amount)) if extra_hp_limit(actor) > 0 else 0.0
    if spent:
        actor.extra_hp = extra - spent
    return set_hp(actor, actor.hp - max(0.0, amount - spent))
