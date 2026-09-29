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
    enable_arv_v211_api_recovery: bool = True
    enable_rotation_stall_detection: bool = True
    enable_observation_sweep: bool = True
    enable_direction_commitment: bool = True
    enable_repair_outcome_verification: bool = True
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
    rotation_stall_window: int = 10
    rotation_stall_angle_rad: float = 2.0 * math.pi
    rotation_stall_displacement_threshold: float = 0.25
    observation_sweep_step_rad: float = math.radians(45.0)
    observation_sweep_max_angle_rad: float = 2.0 * math.pi
    commitment_horizon: int = 4
    repair_verification_horizon: int = 4
    repair_min_displacement: float = 0.25

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
            "position_before": horizontal_pos2(position_before),
            "position_after": horizontal_pos2(position_after),
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

def _distance2(a: Sequence[float], b: Sequence[float]) -> float:
    return float(math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1])))


def horizontal_pos2(pos: Sequence[float]) -> Optional[List[float]]:
    """Project raw Habitat position [x, y, z] to horizontal [x, z]."""
    try:
        values = list(pos)
        if len(values) >= 3:
            return [float(values[0]), float(values[2])]
        if len(values) >= 2:
            return [float(values[0]), float(values[1])]
        return None
    except Exception:
        return None


class RotationStallDetector:
    """Detect full in-place rotation without requiring turn-direction reversal."""

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
            "abs_rotation": abs(float(theta or 0.0)) if action_is_turn(action) else 0.0,
            "heading_before": float(heading_before),
            "heading_after": float(heading_after),
            "position_before": horizontal_pos2(position_before),
            "position_after": horizontal_pos2(position_after),
            "repair_operator": repair_operator,
        }
        self.records.append(record)
        keep = max(int(self.cfg.rotation_stall_window), 2) * 2
        if len(self.records) > keep:
            self.records = self.records[-keep:]
        detection = self.detect()
        if detection:
            self.last_detection = detection
            self.trigger_count += 1
        return detection

    def detect(self) -> Optional[Dict[str, Any]]:
        if not self.cfg.enabled() or not self.cfg.enable_rotation_stall_detection:
            return None
        window = self.records[-max(int(self.cfg.rotation_stall_window), 2):]
        if not window:
            return None
        accumulated = sum(float(r.get("abs_rotation") or 0.0) for r in window)
        if accumulated < float(self.cfg.rotation_stall_angle_rad):
            return None
        positions = [r["position_before"] for r in window if r.get("position_before") is not None]
        if window[-1].get("position_after") is not None:
            positions.append(window[-1]["position_after"])
        if len(positions) < 2:
            return None
        net_displacement = _distance2(positions[-1], positions[0])
        if net_displacement >= float(self.cfg.rotation_stall_displacement_threshold):
            return None
        return {
            "failure_type": "ROTATION_STALL",
            "detector_source": "trajectory/control",
            "step": int(window[-1]["step"]),
            "recent_actions": [{"type": r.get("action_type"), "theta": r.get("theta")} for r in window],
            "recent_headings": [r.get("heading_after") for r in window],
            "recent_positions": [r.get("position_after") for r in window],
            "cumulative_rotation": accumulated,
            "rotation_window_displacement": net_displacement,
            "reason": (
                f"cumulative absolute rotation {accumulated:.3f}rad with "
                f"net horizontal displacement {net_displacement:.3f}m"
            ),
        }

    def recent_rotation(self) -> float:
        window = self.records[-max(int(self.cfg.rotation_stall_window), 2):]
        return float(sum(float(r.get("abs_rotation") or 0.0) for r in window))

    def recent_displacement(self) -> Optional[float]:
        window = self.records[-max(int(self.cfg.rotation_stall_window), 2):]
        if not window:
            return None
        first = window[0].get("position_before")
        last = window[-1].get("position_after")
        if first is None or last is None:
            return None
        return _distance2(last, first)


class ObservationSweep:
    def __init__(self, cfg: ControlStabilityConfig):
        self.cfg = cfg
        self.active = False
        self.completed = False
        self.failure_type: Optional[str] = None
        self.accumulated_angle = 0.0
        self.views: List[Dict[str, Any]] = []
        self.started_step: Optional[int] = None
        self.forced_decision_required = False

    def start(self, step: int, failure_type: str) -> None:
        if not self.cfg.enabled() or not self.cfg.enable_observation_sweep:
            return
        self.active = True
        self.completed = False
        self.failure_type = failure_type
        self.accumulated_angle = 0.0
        self.views = []
        self.started_step = int(step)
        self.forced_decision_required = False

    def next_action(self, action_factory, view_summary: Optional[Dict[str, Any]] = None):
        if not self.active or self.completed:
            return None
        if view_summary is not None:
            self.views.append(dict(view_summary))
        remaining = float(self.cfg.observation_sweep_max_angle_rad) - self.accumulated_angle
        if remaining <= 1e-6:
            self.active = False
            self.completed = True
            self.forced_decision_required = True
            return None
        theta = min(abs(float(self.cfg.observation_sweep_step_rad)), remaining)
        self.accumulated_angle += theta
        if self.accumulated_angle >= float(self.cfg.observation_sweep_max_angle_rad) - 1e-6:
            self.active = False
            self.completed = True
            self.forced_decision_required = True
        return action_factory(theta)

    def trace(self) -> Dict[str, Any]:
        return {
            "observation_sweep_started": self.started_step is not None,
            "observation_sweep_active": self.active,
            "observation_sweep_completed": self.completed,
            "observation_sweep_views": list(self.views),
            "observation_sweep_angle": self.accumulated_angle,
            "forced_direction_decision_required": self.forced_decision_required,
        }


class DirectionCommitment:
    def __init__(self, cfg: ControlStabilityConfig):
        self.cfg = cfg
        self.selected_direction: Optional[str] = None
        self.remaining_steps = 0
        self.candidates: List[Dict[str, Any]] = []
        self.repair_operator: Optional[str] = None
        self.repair_start_step: Optional[int] = None
        self.repair_start_position: Optional[List[float]] = None
        self.repair_start_rotation: float = 0.0
        self.repair_status: Optional[str] = None

    def activate(self, selected_direction: str, candidates: List[Dict[str, Any]], step: int,
                 position: Sequence[float], cumulative_rotation: float = 0.0) -> None:
        if not self.cfg.enabled() or not self.cfg.enable_direction_commitment:
            return
        self.selected_direction = selected_direction
        self.candidates = list(candidates or [])
        self.remaining_steps = int(self.cfg.commitment_horizon)
        self.repair_operator = "FORCED_DIRECTION_DECISION"
        self.repair_start_step = int(step)
        self.repair_start_position = horizontal_pos2(position)
        self.repair_start_rotation = float(cumulative_rotation or 0.0)
        self.repair_status = "PENDING"

    def blocks_reverse(self, requested_direction: str, strong_contradiction: bool = False) -> bool:
        if not self.cfg.enabled() or not self.cfg.enable_direction_commitment:
            return False
        if not self.selected_direction or self.remaining_steps <= 0 or strong_contradiction:
            return False
        opposites = {("L", "R"), ("R", "L"), ("LEFT", "RIGHT"), ("RIGHT", "LEFT")}
        return (self.selected_direction.upper(), str(requested_direction).upper()) in opposites

    def tick(self) -> None:
        if self.remaining_steps > 0:
            self.remaining_steps -= 1

    def verify(self, step: int, position: Sequence[float], cumulative_rotation: float = 0.0) -> Dict[str, Any]:
        if not self.cfg.enabled() or not self.cfg.enable_repair_outcome_verification:
            return {"repair_status": self.repair_status}
        if self.repair_start_step is None or self.repair_start_position is None:
            return {"repair_status": self.repair_status}
        elapsed = int(step) - int(self.repair_start_step)
        current = horizontal_pos2(position)
        displacement = _distance2(current, self.repair_start_position) if current is not None else 0.0
        new_rotation = max(0.0, float(cumulative_rotation or 0.0) - float(self.repair_start_rotation))
        if elapsed >= int(self.cfg.repair_verification_horizon):
            if displacement >= float(self.cfg.repair_min_displacement) and new_rotation < float(self.cfg.rotation_stall_angle_rad):
                self.repair_status = "VERIFIED"
            else:
                self.repair_status = "FAILED"
        return {
            "repair_status": self.repair_status,
            "displacement_after_repair": displacement,
            "new_rotation_after_repair": new_rotation,
        }

    def trace(self) -> Dict[str, Any]:
        return {
            "commitment_active": bool(self.selected_direction and self.remaining_steps > 0),
            "commitment_remaining_steps": int(self.remaining_steps),
            "forced_direction_candidates": list(self.candidates),
            "forced_selected_direction": self.selected_direction,
            "repair_operator": self.repair_operator,
            "repair_start_step": self.repair_start_step,
            "repair_status": self.repair_status,
        }


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
