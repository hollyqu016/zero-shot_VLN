import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence


def normalize_angle(angle: float) -> float:
    return float((float(angle) + math.pi) % (2.0 * math.pi) - math.pi)


def turn_direction(theta: Optional[float], large_turn_threshold: float) -> Optional[int]:
    if theta is None or abs(theta) < large_turn_threshold:
        return None
    return 1 if theta > 0 else -1


def action_is_turn(action: Any) -> bool:
    return getattr(action, "type", None) in {"turn", "turn left", "turn right", "turn around"}


def action_theta(action: Any) -> Optional[float]:
    if not action_is_turn(action):
        return None
    try:
        return float(getattr(action, "theta"))
    except Exception:
        return None


@dataclass
class ControlStabilityConfig:
    enable_arv_v21_control_stability: bool = True
    enable_turn_oscillation_detector: bool = True
    enable_safe_verification_turn: bool = True
    enable_repair_cooldown: bool = True
    enable_safe_fallback_recovery: bool = True
    oscillation_window: int = 6
    oscillation_min_reversals: int = 2
    oscillation_displacement_threshold: float = 0.35
    large_turn_threshold_rad: float = math.radians(135.0)
    small_turn_rad: float = math.radians(45.0)
    cooldown_steps: int = 3

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ControlStabilityConfig":
        raw = {}
        if isinstance(config, dict):
            raw.update(config.get("arv_v21", {}) or {})
            raw.update(config.get("control_stability", {}) or {})
            for key in cls.__dataclass_fields__:
                if key in config:
                    raw[key] = config[key]
        defaults = cls()
        values = {key: raw.get(key, getattr(defaults, key)) for key in cls.__dataclass_fields__}
        return cls(**values)

    def enabled(self) -> bool:
        return bool(self.enable_arv_v21_control_stability)


class TurnOscillationDetector:
    """Detect repeated large heading reversals with little translation."""

    def __init__(self, cfg: ControlStabilityConfig):
        self.cfg = cfg
        self.records: List[Dict[str, Any]] = []
        self.last_detection: Optional[Dict[str, Any]] = None
        self.trigger_count = 0

    def add_record(
        self,
        step: int,
        action: Any,
        heading_before: float,
        heading_after: float,
        position_before: Sequence[float],
        position_after: Sequence[float],
        repair_operator: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        theta = action_theta(action)
        record = {
            "step": int(step),
            "action_type": getattr(action, "type", None),
            "theta": theta,
            "turn_direction": turn_direction(theta, self.cfg.large_turn_threshold_rad),
            "heading_before": float(heading_before),
            "heading_after": float(heading_after),
            "position_before": self._pos2(position_before),
            "position_after": self._pos2(position_after),
            "repair_operator": repair_operator,
        }
        self.records.append(record)
        keep = max(self.cfg.oscillation_window, 2) * 3
        if len(self.records) > keep:
            self.records = self.records[-keep:]

        detection = self.detect()
        if detection:
            self.last_detection = detection
            self.trigger_count += 1
        return detection

    def detect(self) -> Optional[Dict[str, Any]]:
        if not self.cfg.enabled() or not self.cfg.enable_turn_oscillation_detector:
            return None
        window = self.records[-max(self.cfg.oscillation_window, 2):]
        large_turns = [r for r in window if r.get("turn_direction") is not None]
        if len(large_turns) < 3:
            return None

        reversals = 0
        for prev, curr in zip(large_turns, large_turns[1:]):
            delta = abs(normalize_angle(curr["heading_after"] - prev["heading_after"]))
            opposite_turns = prev["turn_direction"] * curr["turn_direction"] < 0
            repeated_half_turn = (
                abs(float(prev.get("theta") or 0.0)) >= self.cfg.large_turn_threshold_rad
                and abs(float(curr.get("theta") or 0.0)) >= self.cfg.large_turn_threshold_rad
                and delta >= self.cfg.large_turn_threshold_rad * 0.75
            )
            if (opposite_turns and delta >= self.cfg.large_turn_threshold_rad * 0.75) or repeated_half_turn:
                reversals += 1

        if reversals < self.cfg.oscillation_min_reversals:
            return None

        positions = [r["position_before"] for r in window if r.get("position_before") is not None]
        if window and window[-1].get("position_after") is not None:
            positions.append(window[-1]["position_after"])
        if len(positions) < 2:
            return None
        net_displacement = _distance2(positions[-1], positions[0])
        if net_displacement >= self.cfg.oscillation_displacement_threshold:
            return None

        return {
            "failure_type": "TURN_OSCILLATION",
            "detector_source": "trajectory/control",
            "step": int(window[-1]["step"]),
            "recent_actions": [{"type": r.get("action_type"), "theta": r.get("theta")} for r in window],
            "recent_headings": [r.get("heading_after") for r in window],
            "recent_positions": [r.get("position_after") for r in window],
            "net_displacement": net_displacement,
            "oscillation_window": len(window),
            "oscillation_reversal_count": int(reversals),
            "oscillation_net_displacement": net_displacement,
            "reason": (
                f"{reversals} large heading reversals within {len(window)} actions "
                f"with net displacement {net_displacement:.3f}m"
            ),
        }

    def recent_reversal_count(self) -> int:
        detection = self.detect()
        if detection:
            return int(detection.get("oscillation_reversal_count", 0))
        window = self.records[-max(self.cfg.oscillation_window, 2):]
        dirs = [r.get("turn_direction") for r in window if r.get("turn_direction") is not None]
        return sum(1 for a, b in zip(dirs, dirs[1:]) if a * b < 0)

    def recent_displacement(self) -> Optional[float]:
        window = self.records[-max(self.cfg.oscillation_window, 2):]
        if not window:
            return None
        first = window[0].get("position_before")
        last = window[-1].get("position_after")
        if first is None or last is None:
            return None
        return _distance2(last, first)

    @staticmethod
    def _pos2(pos: Sequence[float]) -> Optional[List[float]]:
        try:
            values = list(pos)
            if len(values) < 2:
                return None
            return [float(values[0]), float(values[1])]
        except Exception:
            return None


def _distance2(a: Sequence[float], b: Sequence[float]) -> float:
    return float(math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1])))


class RepairCooldown:
    def __init__(self, cfg: ControlStabilityConfig):
        self.cfg = cfg
        self.last_repair_operator: Optional[str] = None
        self.last_repair_step: Optional[int] = None
        self.last_repair_turn_direction: Optional[int] = None

    def record(self, operator: str, step: int, action: Any) -> None:
        if not self.cfg.enabled() or not self.cfg.enable_repair_cooldown:
            return
        direction = turn_direction(action_theta(action), self.cfg.large_turn_threshold_rad)
        self.last_repair_operator = operator
        self.last_repair_step = int(step)
        self.last_repair_turn_direction = direction

    def blocks(self, step: int, action: Any) -> bool:
        if not self.cfg.enabled() or not self.cfg.enable_repair_cooldown:
            return False
        if self.last_repair_step is None or self.last_repair_turn_direction is None:
            return False
        if int(step) - int(self.last_repair_step) > int(self.cfg.cooldown_steps):
            return False
        direction = turn_direction(action_theta(action), self.cfg.large_turn_threshold_rad)
        return direction is not None and direction * self.last_repair_turn_direction < 0

    def to_dict(self, step: int, action: Any = None) -> Dict[str, Any]:
        active = (
            self.last_repair_step is not None
            and int(step) - int(self.last_repair_step) <= int(self.cfg.cooldown_steps)
        )
        return {
            "cooldown_active": bool(active),
            "last_repair_operator": self.last_repair_operator,
            "last_repair_step": self.last_repair_step,
            "last_repair_turn_direction": self.last_repair_turn_direction,
            "would_block_requested_action": self.blocks(step, action) if action is not None else False,
        }
