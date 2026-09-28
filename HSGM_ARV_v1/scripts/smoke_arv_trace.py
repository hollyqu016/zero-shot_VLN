#!/usr/bin/env python
"""Non-Habitat checks for ARV state/controller wiring and v2.1 control stability."""

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agent.belief import ARVState, resolve_controller_decision  # noqa: E402
from agent.control_stability import ControlStabilityConfig, RepairCooldown, TurnOscillationDetector  # noqa: E402


class MockController:
    """Small runtime harness for the gate shared with the real agent."""

    def __init__(self, subtask_index=0, is_final=False):
        self.arv_state = ARVState()
        self.current_subtask_index = subtask_index
        self.is_final = is_final
        self.semantic_completion = False
        self.termination_reason = None
        self.committed_validation_target = None
        self.committed_validation_source = None

    def handle(self, parsed_json, step):
        if parsed_json is None:
            return resolve_controller_decision(None, self.arv_state, self.current_subtask_index)
        action_key = str(parsed_json["action"]).upper()
        self.arv_state.observe_decision(parsed_json, step, self.current_subtask_index, action_key)
        branch = resolve_controller_decision(action_key, self.arv_state, self.current_subtask_index)
        if branch == "stop":
            completed_index = self.current_subtask_index
            self.committed_validation_target = self.arv_state.completion_validation_target
            self.committed_validation_source = self.arv_state.completion_validation_source
            self.arv_state.commit_completion(completed_index, [0, 0, 0], step)
            if self.is_final:
                self.semantic_completion = True
                self.termination_reason = "verified_completion"
            else:
                self.current_subtask_index += 1
        return branch


def decision(active_claim, belief_type="ROUTE_TRANSITION", observed=None,
             missing=None, contradictions=None, diagnosis=None, repair=None,
             validation=None, action=1):
    return {
        "evidence": {
            "observed": observed or [],
            "missing_expected": missing or [],
            "contradictions": contradictions or [],
        },
        "belief": {
            "belief_type": belief_type,
            "active_claim": active_claim,
            "confidence": 0.7,
        },
        "diagnosis": diagnosis or {"error_detected": False},
        "repair": repair or {},
        "validation": validation or {},
        "action": action,
    }


def route_error(replacement="alternative corridor reaches dining area"):
    return decision(
        "corridor A leads to dining area",
        observed=["corridor continues"],
        contradictions=["formal dining table absent"],
        diagnosis={
            "error_detected": True,
            "failed_belief_type": "ROUTE_TRANSITION",
            "failed_belief": "corridor A leads to dining area",
            "contradicting_evidence": ["formal dining table absent"],
        },
        repair={
            "required": True,
            "replacement_belief": replacement,
            "replacement_belief_type": "ROUTE_TRANSITION",
            "repair_evidence": ["formal dining table absent"],
            "validation_target": "sofa on right",
            "validation_target_stage": 2,
            "validation_target_source": "next_subtask",
            "validation_target_instruction_span": "with a sofa on your right",
        },
    )


def test_attribution():
    state = ARVState()
    trace = state.observe_decision(route_error(), 1, 1, 1)
    assert trace["failed_belief_id"]
    assert trace["failed_belief_type"] == "ROUTE_TRANSITION"
    assert trace["state"]["failed_belief"]["subtask_index"] == 1


def test_localized_revision():
    state = ARVState()
    state.commit_completion(0, [0, 0, 0], 0)
    verified_before = state.to_dict()["verified_beliefs"][0]
    trace = state.observe_decision(route_error(), 1, 1, 1)
    assert trace["repair_status"] == "provisional"
    assert trace["state"]["replacement_belief"]["proposition"] != trace["state"]["failed_belief"]["proposition"]
    assert state.to_dict()["verified_beliefs"][0] == verified_before

    no_repair = ARVState()
    trace_same = no_repair.observe_decision(
        route_error("corridor A leads to dining area"), 1, 1, 1
    )
    assert trace_same["repair_status"] == "diagnosed_not_repaired"


def test_evidence_leakage():
    state = ARVState()
    state.observe_decision(route_error(), 1, 1, 1)
    unrelated = decision(
        "alternative corridor reaches dining area",
        observed=["lamp"],
        validation={"target": "sofa on right", "target_stage": 2, "observed_support": ["lamp"], "validated": True},
    )
    state.observe_decision(unrelated, 2, 2, 1)
    assert state.repair_status == "provisional"

    missing = decision(
        "alternative corridor reaches dining area",
        observed=[],
        missing=["sofa on right"],
        validation={"target": "sofa on right", "target_stage": 2, "observed_support": [], "validated": True},
    )
    state.observe_decision(missing, 3, 2, 1)
    assert state.repair_status == "provisional"

    valid = decision(
        "alternative corridor reaches dining area",
        observed=["sofa on right"],
        validation={"target": "sofa on right", "target_stage": 2, "observed_support": ["sofa on right"], "validated": True},
    )
    state.observe_decision(valid, 4, 2, 1)
    assert state.repair_status == "verified"


def test_stop_gating():
    controller = MockController(subtask_index=2)
    state = controller.arv_state
    first = decision(
        "agent is at bar corner", belief_type="COMPLETION",
        observed=["bar sign"],
        validation={"target": "bar counter alignment", "target_stage": 2,
                    "target_source": "current_subtask",
                    "target_instruction_span": "stop at the bar counter corner",
                    "observed_support": ["bar sign"]},
        action=-1,
    )
    assert controller.handle(first, 5) == "verify_completion"
    assert state.completion_status == "proposed"
    assert not state.is_completion_verified(2)
    assert controller.current_subtask_index == 2
    assert controller.semantic_completion is False

    second = decision(
        "agent is at bar corner", belief_type="COMPLETION",
        observed=["bar counter alignment"],
        validation={"target": "different target chosen after observation",
                    "target_source": "full_instruction",
                    "target_instruction_span": "invented later",
                    "observed_support": ["bar counter alignment"]},
        action=-1,
    )
    assert controller.handle(second, 6) == "stop"
    assert controller.committed_validation_target == "bar counter alignment"
    assert controller.committed_validation_source == "current_subtask"
    assert controller.current_subtask_index == 3
    assert controller.semantic_completion is False
    assert len([b for b in state.beliefs if b.belief_type == "PROGRESS_STAGE"]) == 1


def test_invalid_json_recovery():
    controller = MockController(subtask_index=1)
    state = controller.arv_state
    state.record_invalid_decision()
    assert state.summary()["invalid_decision_count"] == 1
    assert controller.handle(None, 3) == "invalid_fallback"
    assert controller.current_subtask_index == 1
    assert controller.semantic_completion is False
    assert controller.termination_reason is None


def test_rollback():
    state = ARVState()
    state.commit_completion(0, [0, 0, 0], 0)
    state.observe_decision(route_error(), 1, 1, 1)
    state.observe_decision(decision("future landmark", "LANDMARK_BINDING"), 2, 2, 1)
    event = state.rollback_to_stage(1)
    stage0 = [b for b in state.beliefs if b.subtask_index == 0][0]
    assert stage0.status == "verified"
    assert event["beliefs_preserved"] == [stage0.belief_id]
    assert all(b.status not in {"active", "provisional"} for b in state.beliefs if (b.subtask_index or 0) >= 1)


def test_completion_index():
    state = ARVState()
    state.commit_completion(1, [1, 2, 3], 8)
    progress = [b for b in state.beliefs if b.belief_type == "PROGRESS_STAGE"][-1]
    assert progress.subtask_index == 1
    assert state.summary()["final_verified_stage"] == 1


class DummyAction:
    def __init__(self, theta, action_type="turn"):
        self.theta = theta
        self.r = 0
        self.type = action_type


def test_v21_turn_oscillation_detector():
    cfg = ControlStabilityConfig(
        oscillation_window=4,
        oscillation_min_reversals=2,
        oscillation_displacement_threshold=0.2,
    )
    detector = TurnOscillationDetector(cfg)
    headings = [0.0, 3.14159, 0.0, 3.14159]
    last = None
    for step, heading in enumerate(headings):
        theta = 3.14159 if step % 2 == 0 else -3.14159
        last = detector.add_record(
            step=step,
            action=DummyAction(theta),
            heading_before=headings[step - 1] if step else 0.0,
            heading_after=heading,
            position_before=[0.0, 0.0],
            position_after=[0.02 * step, 0.0],
            repair_operator="VERIFY_COMPLETION_OBSERVE",
        )
    assert last is not None
    assert last["failure_type"] == "TURN_OSCILLATION"

    same_direction_detector = TurnOscillationDetector(cfg)
    same_direction_last = None
    for step, heading in enumerate(headings):
        same_direction_last = same_direction_detector.add_record(
            step=step,
            action=DummyAction(3.14159),
            heading_before=headings[step - 1] if step else 0.0,
            heading_after=heading,
            position_before=[0.0, 0.0],
            position_after=[0.01 * step, 0.0],
            repair_operator="VERIFY_COMPLETION_OBSERVE",
        )
    assert same_direction_last is not None
    assert same_direction_last["failure_type"] == "TURN_OSCILLATION"


def test_v21_normal_turn_forward_no_oscillation():
    cfg = ControlStabilityConfig(
        oscillation_window=4,
        oscillation_min_reversals=2,
        oscillation_displacement_threshold=0.2,
    )
    detector = TurnOscillationDetector(cfg)
    sequence = [
        (DummyAction(1.5708), 0.0, 1.5708, [0.0, 0.0], [0.0, 0.0]),
        (DummyAction(0.0, "move_forward"), 1.5708, 1.5708, [0.0, 0.0], [0.0, 0.8]),
        (DummyAction(-1.5708), 1.5708, 0.0, [0.0, 0.8], [0.0, 0.8]),
        (DummyAction(0.0, "move_forward"), 0.0, 0.0, [0.0, 0.8], [0.8, 0.8]),
    ]
    detections = [
        detector.add_record(i, action, hb, ha, pb, pa)
        for i, (action, hb, ha, pb, pa) in enumerate(sequence)
    ]
    assert all(item is None for item in detections)


def test_v21_repair_cooldown_blocks_reverse_large_turn():
    cfg = ControlStabilityConfig(cooldown_steps=3)
    cooldown = RepairCooldown(cfg)
    cooldown.record("VERIFY_COMPLETION_OBSERVE", 10, DummyAction(3.14159))
    assert cooldown.blocks(11, DummyAction(-3.14159))
    assert not cooldown.blocks(14, DummyAction(-3.14159))


def test_v21_global_disable_legacy_path():
    cfg = ControlStabilityConfig(enable_arv_v21_control_stability=False)
    detector = TurnOscillationDetector(cfg)
    cooldown = RepairCooldown(cfg)
    assert detector.add_record(0, DummyAction(3.14159), 0.0, 3.14159, [0, 0], [0, 0]) is None
    assert cooldown.blocks(1, DummyAction(-3.14159)) is False


def main():
    tests = [
        test_attribution,
        test_localized_revision,
        test_evidence_leakage,
        test_stop_gating,
        test_invalid_json_recovery,
        test_rollback,
        test_completion_index,
        test_v21_turn_oscillation_detector,
        test_v21_normal_turn_forward_no_oscillation,
        test_v21_repair_cooldown_blocks_reverse_large_turn,
        test_v21_global_disable_legacy_path,
    ]
    for test in tests:
        test()
    print(json.dumps({"ok": True, "tests": [test.__name__ for test in tests]}, indent=2))


if __name__ == "__main__":
    main()
