from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import re

RAW_PER_SCORE = 10
MIN_RAW_FAVORABILITY = -1000
MAX_RAW_FAVORABILITY = 1000

MIN_FAVORABILITY = MIN_RAW_FAVORABILITY // RAW_PER_SCORE
MAX_FAVORABILITY = MAX_RAW_FAVORABILITY // RAW_PER_SCORE

MAX_DELTA_PER_TURN_SCORE = 30
DEFAULT_DAILY_CAP = 7.0
DEFAULT_DAILY_BYPASS_LIMIT = 10.0
DEFAULT_BANK_LIMIT = 70.0
PENALTY_COOLDOWN_SECONDS = 1800
MIN_PENALTY_COOLDOWN_FACTOR = 0.25

_APOLOGY_PATTERN = re.compile(r"(道歉|认错|抱歉|对不起|补偿)", re.IGNORECASE)
_BYPASS_PATTERN = re.compile(r"(生日|周年|纪念|节日|跨年|新年|圣诞|补偿|活动奖励|大事件|#bypass)", re.IGNORECASE)


@dataclass(frozen=True)
class FavorabilityTransition:
    old_score: int
    old_raw: int
    requested_change: int
    requested_change_raw: int
    applied_change: int
    applied_change_raw: int
    new_score: int
    new_raw: int
    state_before: str
    state_after: str
    daily_gain_used_before: float
    daily_gain_used_after: float
    daily_bypass_used_before: float
    daily_bypass_used_after: float
    daily_gain_bank_before: float
    daily_gain_bank_after: float
    daily_cap_before: float
    daily_cap_after: float
    cap_reset_at_after: datetime
    apology_counts_after: dict[str, int]
    last_interact_at: datetime
    notes: tuple[str, ...] = ()


def clamp_raw_favorability(raw_score: int) -> int:
    return max(MIN_RAW_FAVORABILITY, min(MAX_RAW_FAVORABILITY, int(raw_score)))


def clamp_favorability(score: int) -> int:
    return max(MIN_FAVORABILITY, min(MAX_FAVORABILITY, int(score)))


def raw_to_score(raw_score: int) -> int:
    return clamp_favorability(int(round(clamp_raw_favorability(raw_score) / RAW_PER_SCORE)))


def score_to_raw(score: int) -> int:
    return clamp_raw_favorability(clamp_favorability(score) * RAW_PER_SCORE)


def get_favorability_state(raw_score: int) -> str:
    r = clamp_raw_favorability(raw_score)
    if r <= -800:
        return "broken"
    if r <= -500:
        return "distressed"
    if r <= -200:
        return "upset"
    if r < 200:
        return "normal"
    if r < 500:
        return "happy"
    if r < 750:
        return "affectionate"
    if r < 900:
        return "enamored"
    return "love"


def status_desc_from_raw(raw_score: int) -> str:
    state = get_favorability_state(raw_score)
    if state in {"broken", "distressed"}:
        return "厌恶/仇视"
    if state == "upset":
        return "冷淡/防备"
    if state == "normal":
        return "陌生/普通"
    if state == "happy":
        return "友善/熟人"
    if state in {"affectionate", "enamored"}:
        return "亲密/死党"
    return "恋人/依赖"


def status_desc_from_score(score: int) -> str:
    return status_desc_from_raw(score_to_raw(score))


_GAIN_MULTIPLIER = {
    "broken": 1.45,
    "distressed": 1.30,
    "upset": 1.15,
    "normal": 1.00,
    "happy": 0.90,
    "affectionate": 0.78,
    "enamored": 0.65,
    "love": 0.55,
}

_LOSE_MULTIPLIER = {
    "broken": 0.65,
    "distressed": 0.80,
    "upset": 0.90,
    "normal": 1.00,
    "happy": 1.12,
    "affectionate": 1.25,
    "enamored": 1.40,
    "love": 1.60,
}


def _reason_is_apology(reason: str) -> bool:
    return bool(_APOLOGY_PATTERN.search(reason or ""))


def _reason_is_bypass(reason: str) -> bool:
    return bool(_BYPASS_PATTERN.search(reason or ""))


def _apology_key(reason: str) -> str | None:
    if not _reason_is_apology(reason):
        return None
    text = reason or ""
    if re.search(r"(失约|爽约|放鸽子|不回|冷落|消失)", text):
        return "apology_absence"
    if re.search(r"(骂|侮辱|冒犯|嘴臭)", text):
        return "apology_insult"
    return "apology_generic"


def _apply_apology_diminishing(value: float, reason: str, counts: dict[str, int], notes: list[str]) -> tuple[float, dict[str, int]]:
    key = _apology_key(reason)
    if not key:
        return value, dict(counts)

    new_counts = dict(counts)
    used = int(new_counts.get(key, 0))
    new_counts[key] = used + 1

    if used == 0:
        notes.append("apology_first")
        return value, new_counts
    if used == 1:
        notes.append("apology_second_half")
        return value * 0.5, new_counts

    notes.append("apology_exhausted")
    return 0.0, new_counts


def _safe_float(value: float | int | None, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _normalize_daily(
    now: datetime,
    cap_reset_at: datetime | None,
    daily_gain_used: float | None,
    daily_bypass_used: float | None,
    daily_cap: float | None,
    notes: list[str],
) -> tuple[float, float, float, datetime]:
    cap_input = _safe_float(daily_cap, DEFAULT_DAILY_CAP)
    cap = cap_input if cap_input > 0 else DEFAULT_DAILY_CAP
    gain_used = max(0.0, _safe_float(daily_gain_used))
    bypass_used = max(0.0, _safe_float(daily_bypass_used))
    reset_at = cap_reset_at or now

    if reset_at.date() != now.date():
        gain_used = 0.0
        bypass_used = 0.0
        cap = DEFAULT_DAILY_CAP
        reset_at = now
        notes.append("daily_reset")

    return gain_used, bypass_used, cap, reset_at


def apply_monika_favorability_change(
    *,
    old_score: int,
    old_raw: int | None,
    requested_change: int,
    reason: str,
    now: datetime,
    daily_gain_used: float | None,
    daily_bypass_used: float | None,
    daily_gain_bank: float | None,
    daily_cap: float | None,
    cap_reset_at: datetime | None,
    apology_counts: dict[str, int] | None,
    last_penalty_at: datetime | None = None,
) -> FavorabilityTransition:
    base_raw = clamp_raw_favorability(old_raw if old_raw is not None else score_to_raw(old_score))
    base_score = raw_to_score(base_raw)
    req_score = int(requested_change)

    notes: list[str] = []
    counts_before = dict(apology_counts or {})
    counts_after = counts_before

    gain_used, bypass_used, cap_value, reset_at = _normalize_daily(
        now=now,
        cap_reset_at=cap_reset_at,
        daily_gain_used=daily_gain_used,
        daily_bypass_used=daily_bypass_used,
        daily_cap=daily_cap,
        notes=notes,
    )
    bank_value = max(0.0, _safe_float(daily_gain_bank))

    if req_score > MAX_DELTA_PER_TURN_SCORE:
        req_score = MAX_DELTA_PER_TURN_SCORE
        notes.append("delta_capped_positive")
    elif req_score < -MAX_DELTA_PER_TURN_SCORE:
        req_score = -MAX_DELTA_PER_TURN_SCORE
        notes.append("delta_capped_negative")

    req_raw = req_score * RAW_PER_SCORE
    state_before = get_favorability_state(base_raw)

    if req_score == 0:
        return FavorabilityTransition(
            old_score=base_score,
            old_raw=base_raw,
            requested_change=0,
            requested_change_raw=0,
            applied_change=0,
            applied_change_raw=0,
            new_score=base_score,
            new_raw=base_raw,
            state_before=state_before,
            state_after=state_before,
            daily_gain_used_before=gain_used,
            daily_gain_used_after=gain_used,
            daily_bypass_used_before=bypass_used,
            daily_bypass_used_after=bypass_used,
            daily_gain_bank_before=bank_value,
            daily_gain_bank_after=bank_value,
            daily_cap_before=cap_value,
            daily_cap_after=cap_value,
            cap_reset_at_after=reset_at,
            apology_counts_after=counts_after,
            last_interact_at=now,
            notes=tuple(notes),
        )

    if req_score > 0:
        headroom_raw = MAX_RAW_FAVORABILITY - base_raw
        saturation = max(0.20, min(1.0, headroom_raw / (55 * RAW_PER_SCORE)))
        proposed_raw = req_raw * _GAIN_MULTIPLIER[state_before] * saturation
        if base_raw < -600:
            proposed_raw += 2 * RAW_PER_SCORE
            notes.append("redemption_bonus")
        proposed_raw, counts_after = _apply_apology_diminishing(proposed_raw, reason, counts_before, notes)

        change_score = max(0.0, proposed_raw / RAW_PER_SCORE)
        if _reason_is_bypass(reason):
            bypass_avail = max(0.0, DEFAULT_DAILY_BYPASS_LIMIT - bypass_used)
            bypass_gain = min(change_score, bypass_avail)
            overflow = max(0.0, change_score - bypass_gain)
            bank_avail = max(0.0, DEFAULT_BANK_LIMIT - bank_value)
            bank_add = min(overflow, bank_avail)
            change_score = bypass_gain
            bypass_used += bypass_gain
            bank_value += bank_add
            if overflow > 0:
                notes.append("bypass_overflow")
            if bank_add > 0:
                notes.append("banked_gain")
            if bypass_gain > 0:
                notes.append("bypass_gain")
        else:
            gain_avail = max(0.0, cap_value - gain_used)
            if change_score > gain_avail:
                notes.append("daily_cap_limited")
            change_score = min(change_score, gain_avail)
            gain_used += change_score

        change_raw = change_score * RAW_PER_SCORE
    else:
        floorroom_raw = base_raw - MIN_RAW_FAVORABILITY
        saturation = max(0.20, min(1.0, floorroom_raw / (55 * RAW_PER_SCORE)))
        proposed_raw = req_raw * _LOSE_MULTIPLIER[state_before] * saturation
        if base_raw > 800:
            proposed_raw -= 2 * RAW_PER_SCORE
            notes.append("high_trust_fragile")

        lose_score = max(0.0, -proposed_raw / RAW_PER_SCORE)
        if last_penalty_at is not None:
            elapsed = (now - last_penalty_at).total_seconds()
            if elapsed < PENALTY_COOLDOWN_SECONDS:
                ratio = max(0.0, elapsed / PENALTY_COOLDOWN_SECONDS)
                cooldown_factor = max(MIN_PENALTY_COOLDOWN_FACTOR, ratio)
                lose_score *= cooldown_factor
                notes.append(f"penalty_cooldown({int(elapsed)}s)")
        if bank_value > 0:
            base_lose = lose_score * 0.4
            bank_lose = lose_score - base_lose
            if bank_value < bank_lose:
                bank_lose = bank_value
                base_lose = lose_score - bank_lose
            else:
                bank_lose = min(bank_lose * 1.25, bank_value)
            lose_score = base_lose + bank_lose
            bank_value -= bank_lose
            if bank_lose > 0:
                notes.append("bank_penalty")

        change_raw = -lose_score * RAW_PER_SCORE

    applied_raw = int(round(change_raw))
    if req_raw > 0 and applied_raw <= 0 and change_raw > 0:
        applied_raw = 1
    elif req_raw < 0 and applied_raw >= 0 and change_raw < 0:
        applied_raw = -1

    new_raw = clamp_raw_favorability(base_raw + applied_raw)
    applied_raw = new_raw - base_raw
    new_score = raw_to_score(new_raw)
    applied_score = new_score - base_score
    state_after = get_favorability_state(new_raw)
    cap_before = _safe_float(daily_cap, DEFAULT_DAILY_CAP)
    if cap_before <= 0:
        cap_before = DEFAULT_DAILY_CAP

    return FavorabilityTransition(
        old_score=base_score,
        old_raw=base_raw,
        requested_change=req_score,
        requested_change_raw=req_raw,
        applied_change=applied_score,
        applied_change_raw=applied_raw,
        new_score=new_score,
        new_raw=new_raw,
        state_before=state_before,
        state_after=state_after,
        daily_gain_used_before=max(0.0, _safe_float(daily_gain_used)),
        daily_gain_used_after=gain_used,
        daily_bypass_used_before=max(0.0, _safe_float(daily_bypass_used)),
        daily_bypass_used_after=bypass_used,
        daily_gain_bank_before=max(0.0, _safe_float(daily_gain_bank)),
        daily_gain_bank_after=bank_value,
        daily_cap_before=cap_before,
        daily_cap_after=cap_value,
        cap_reset_at_after=reset_at,
        apology_counts_after=counts_after,
        last_interact_at=now,
        notes=tuple(notes),
    )


def apply_favorability_change(old_score: int, requested_change: int) -> FavorabilityTransition:
    now = datetime.now()
    return apply_monika_favorability_change(
        old_score=old_score,
        old_raw=None,
        requested_change=requested_change,
        reason="",
        now=now,
        daily_gain_used=0.0,
        daily_bypass_used=0.0,
        daily_gain_bank=0.0,
        daily_cap=999.0,
        cap_reset_at=now,
        apology_counts={},
    )
