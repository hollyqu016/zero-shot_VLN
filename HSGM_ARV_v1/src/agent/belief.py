from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


ERROR_TYPES = {"LANDMARK_BINDING", "PROGRESS_STAGE", "ROUTE_TRANSITION", "COMPLETION"}
VALIDATION_TARGET_SOURCES = {"current_subtask", "next_subtask", "full_instruction"}


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def _text(value: Any) -> str:
    return str(value or "").strip()


def _matches_target(evidence: Any, target: str) -> bool:
    evidence_text = _text(evidence).casefold()
    target_text = _text(target).casefold()
    return bool(evidence_text and target_text and (
        evidence_text in target_text or target_text in evidence_text
    ))


def resolve_controller_decision(action_key: Any, arv_state: "ARVState",
                                subtask_index: Optional[int]) -> str:
    """Return the controller branch after applying ARV completion gating."""
    if action_key is None:
        return "invalid_fallback"
    if _text(action_key).upper() == "-1":
        return "stop" if arv_state.is_completion_verified(subtask_index) else "verify_completion"
    return "action"


@dataclass
class BeliefRecord:
    belief_id: str
    belief_type: str
    subtask_index: Optional[int]
    proposition: str
    status: str = "hypothesized"
    confidence: float = 0.0
    supporting_evidence: List[Any] = field(default_factory=list)
    contradicting_evidence: List[Any] = field(default_factory=list)
    created_step: int = 0
    last_updated_step: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return vars(self).copy()


class ARVState:
    """Attribution, localized revision, and independent validation state."""

    def __init__(self) -> None:
        self.beliefs: List[BeliefRecord] = []
        self.failed_belief_id: Optional[str] = None
        self.replacement_belief_id: Optional[str] = None
        self.last_verified_pose: Optional[Any] = None
        self.repair_evidence: List[Any] = []
        self.validation_evidence: List[Any] = []
        self.repair_status = "none"
        self.repair_step: Optional[int] = None
        self.repair_subtask_index: Optional[int] = None
        self.validation_target: Optional[str] = None
        self.validation_target_stage: Optional[int] = None
        self.validation_target_source: Optional[str] = None
        self.validation_target_instruction_span: Optional[str] = None
        self.completion_status = "none"
        self.completion_step: Optional[int] = None
        self.completion_subtask_index: Optional[int] = None
        self.completion_evidence: List[Any] = []
        self.completion_validation_evidence: List[Any] = []
        self.completion_validation_target: Optional[str] = None
        self.completion_validation_stage: Optional[int] = None
        self.completion_validation_source: Optional[str] = None
        self.completion_validation_instruction_span: Optional[str] = None
        self.invalid_decision_count = 0
        self.rollback_count = 0
        self.rollback_events: List[Dict[str, Any]] = []
        self.num_attributions = 0
        self.num_repairs = 0
        self.num_verified_repairs = 0
        self.num_rejected_repairs = 0
        self.num_completion_proposals = 0
        self.num_verified_completions = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "beliefs": [belief.to_dict() for belief in self.beliefs],
            "active_beliefs": [belief.to_dict() for belief in self.beliefs if belief.status in {"hypothesized", "active", "provisional"}],
            "verified_beliefs": [belief.to_dict() for belief in self.beliefs if belief.status == "verified"],
            "failed_belief": self._belief_to_dict(self.failed_belief_id),
            "replacement_belief": self._belief_to_dict(self.replacement_belief_id),
            "last_verified_pose": self.last_verified_pose,
            "repair_evidence": list(self.repair_evidence),
            "validation_evidence": list(self.validation_evidence),
            "repair_status": self.repair_status,
            "repair_step": self.repair_step,
            "repair_subtask_index": self.repair_subtask_index,
            "validation_target": self.validation_target,
            "validation_target_stage": self.validation_target_stage,
            "validation_target_source": self.validation_target_source,
            "validation_target_instruction_span": self.validation_target_instruction_span,
            "completion_status": self.completion_status,
            "completion_evidence": list(self.completion_evidence),
            "completion_validation_evidence": list(self.completion_validation_evidence),
            "completion_validation_target": self.completion_validation_target,
            "completion_validation_stage": self.completion_validation_stage,
            "completion_validation_source": self.completion_validation_source,
            "completion_validation_instruction_span": self.completion_validation_instruction_span,
            "rollback_events": list(self.rollback_events),
        }

    def summary(self) -> Dict[str, Any]:
        verified_stages = [
            belief.subtask_index for belief in self.beliefs
            if belief.status == "verified" and belief.subtask_index is not None
        ]
        return {
            "num_attributions": self.num_attributions,
            "num_repairs": self.num_repairs,
            "num_verified_repairs": self.num_verified_repairs,
            "num_rejected_repairs": self.num_rejected_repairs,
            "num_completion_proposals": self.num_completion_proposals,
            "num_verified_completions": self.num_verified_completions,
            "invalid_decision_count": self.invalid_decision_count,
            "rollback_count": self.rollback_count,
            "final_verified_stage": max(verified_stages) if verified_stages else None,
            "repair_status": self.repair_status,
            "completion_status": self.completion_status,
        }

    def observe_decision(self, parsed_json: Dict[str, Any], step: int,
                         subtask_index: Optional[int], selected_action: Any) -> Dict[str, Any]:
        evidence = parsed_json.get("evidence") if isinstance(parsed_json.get("evidence"), dict) else {}
        belief = parsed_json.get("belief") if isinstance(parsed_json.get("belief"), dict) else {}
        diagnosis = parsed_json.get("diagnosis") if isinstance(parsed_json.get("diagnosis"), dict) else {}
        repair = parsed_json.get("repair") if isinstance(parsed_json.get("repair"), dict) else {}
        validation = parsed_json.get("validation") if isinstance(parsed_json.get("validation"), dict) else {}

        observed = _as_list(evidence.get("observed"))
        missing_expected = _as_list(evidence.get("missing_expected"))
        contradictions = _as_list(evidence.get("contradictions"))
        active_claim = _text(belief.get("active_claim") or parsed_json.get("plan"))
        belief_type = self._normalize_type(belief.get("belief_type") or diagnosis.get("failed_belief_type"))
        confidence = self._safe_confidence(belief.get("confidence"))

        active_record = None
        if active_claim:
            active_record = self._upsert_active_belief(
                belief_type, subtask_index, active_claim, observed, confidence, step
            )

        error_detected = bool(diagnosis.get("error_detected")) or bool(contradictions)
        if error_detected:
            self._attribute_and_repair(
                diagnosis, repair, contradictions, observed, active_record,
                belief_type, subtask_index, confidence, step,
            )
        else:
            self._validate_repair(validation, observed, step, subtask_index)

        if _text(selected_action).upper() == "-1":
            self._observe_completion_proposal(
                validation, observed, missing_expected, contradictions,
                subtask_index, confidence, step,
            )

        return {
            "observed_evidence": observed,
            "missing_expected": missing_expected,
            "contradiction": contradictions or _as_list(diagnosis.get("contradicting_evidence")),
            "diagnosis": diagnosis,
            "failed_belief_id": self.failed_belief_id,
            "failed_belief_type": self._belief_type(self.failed_belief_id),
            "repair_status": self.repair_status,
            "replacement_belief_id": self.replacement_belief_id,
            "preserved_verified_beliefs": [b.belief_id for b in self.beliefs if b.status == "verified"],
            "repair_evidence": list(self.repair_evidence),
            "validation_target": self.validation_target,
            "validation_target_stage": self.validation_target_stage,
            "validation_target_source": self.validation_target_source,
            "validation_target_instruction_span": self.validation_target_instruction_span,
            "validation_evidence": list(self.validation_evidence),
            "validation_status": "verified" if self.repair_status == "verified" else "pending",
            "completion_status": self.completion_status,
            "state": self.to_dict(),
        }

    def _attribute_and_repair(self, diagnosis: Dict[str, Any], repair: Dict[str, Any],
                              contradictions: List[Any], observed: List[Any],
                              active_record: Optional[BeliefRecord], belief_type: str,
                              subtask_index: Optional[int], confidence: float, step: int) -> None:
        failed_type = self._normalize_type(diagnosis.get("failed_belief_type") or belief_type)
        failed_prop = _text(diagnosis.get("failed_belief") or (active_record.proposition if active_record else "unspecified belief"))
        failed = self._find_matching_belief(failed_type, subtask_index, failed_prop) or active_record
        if failed is None:
            failed = self._new_belief(failed_type, subtask_index, failed_prop, step)
            self.beliefs.append(failed)
        failed.status = "contradicted"
        failed.last_updated_step = step
        failed.contradicting_evidence.extend(_as_list(diagnosis.get("contradicting_evidence")) + contradictions)
        self.failed_belief_id = failed.belief_id
        self.num_attributions += 1
        self.repair_evidence = _as_list(repair.get("repair_evidence")) or (
            _as_list(diagnosis.get("contradicting_evidence")) + contradictions
        )
        self.validation_evidence = []
        self.repair_step = step
        self.repair_subtask_index = subtask_index
        self.validation_target = _text(repair.get("validation_target")) or None
        self.validation_target_stage = self._safe_int(repair.get("validation_target_stage"))
        self.validation_target_source = self._valid_target_source(repair.get("validation_target_source"))
        self.validation_target_instruction_span = _text(repair.get("validation_target_instruction_span")) or None
        if not self.validation_target_source or not self.validation_target_instruction_span:
            self.validation_target = None

        replacement_prop = _text(repair.get("replacement_belief"))
        if not replacement_prop or replacement_prop.casefold() == failed.proposition.casefold():
            self.repair_status = "diagnosed_not_repaired"
            self.replacement_belief_id = None
            return

        replacement_type = self._normalize_type(repair.get("replacement_belief_type") or failed_type)
        replacement = self._new_belief(replacement_type, subtask_index, replacement_prop, step)
        replacement.status = "provisional"
        replacement.supporting_evidence.extend(observed)
        replacement.confidence = confidence
        self.beliefs.append(replacement)
        self.replacement_belief_id = replacement.belief_id
        self.repair_status = "provisional"
        self.num_repairs += 1

    def _validate_repair(self, validation: Dict[str, Any], observed: List[Any],
                         step: int, subtask_index: Optional[int]) -> None:
        if self.repair_status != "provisional" or self.repair_step is None:
            return
        target = self.validation_target
        target_stage = self.validation_target_stage
        support = _as_list(validation.get("observed_support"))
        valid_support = [
            item for item in support
            if item in observed and item not in self.repair_evidence and _matches_target(item, target)
        ]
        later_stage = target_stage is None or (
            subtask_index is not None and subtask_index >= target_stage
        )
        if step > self.repair_step and later_stage and valid_support:
            self.validation_evidence.extend(valid_support)
            self.repair_status = "verified"
            replacement = self._belief_by_id(self.replacement_belief_id)
            if replacement is not None:
                replacement.status = "verified"
                replacement.supporting_evidence.extend(valid_support)
                replacement.last_updated_step = step
            self.num_verified_repairs += 1

    def _observe_completion_proposal(self, validation: Dict[str, Any], observed: List[Any],
                                     missing_expected: List[Any], contradictions: List[Any],
                                     subtask_index: Optional[int], confidence: float, step: int) -> None:
        proposed_target = _text(validation.get("target"))
        proposed_stage = self._safe_int(validation.get("target_stage"))
        proposed_source = self._valid_target_source(validation.get("target_source"))
        proposed_span = _text(validation.get("target_instruction_span")) or None
        target = self.completion_validation_target
        support = _as_list(validation.get("observed_support"))
        valid_support = [item for item in support if item in observed and _matches_target(item, target)]
        is_follow_up = (
            self.completion_status in {"proposed", "verifying"}
            and self.completion_step is not None
            and step > self.completion_step
            and subtask_index == self.completion_subtask_index
        )
        independent = [item for item in valid_support if item not in self.completion_evidence]
        if is_follow_up and independent and not missing_expected and not contradictions:
            self.completion_status = "verified"
            self.completion_validation_evidence.extend(independent)
            self.num_verified_completions += 1
            for record in reversed(self.beliefs):
                if record.belief_type == "COMPLETION" and record.status == "provisional":
                    record.status = "verified"
                    record.supporting_evidence.extend(independent)
                    record.last_updated_step = step
                    break
            return

        completion = self._new_belief("COMPLETION", subtask_index, "VLM proposed subtask completion", step)
        completion.status = "provisional"
        completion.supporting_evidence.extend(observed)
        completion.confidence = confidence
        self.beliefs.append(completion)
        self.completion_status = "verifying" if is_follow_up else "proposed"
        if not is_follow_up:
            self.completion_step = step
            self.completion_subtask_index = subtask_index
            self.completion_evidence = list(observed)
            self.completion_validation_evidence = []
            if proposed_target and proposed_source and proposed_span:
                self.completion_validation_target = proposed_target
                self.completion_validation_stage = proposed_stage
                self.completion_validation_source = proposed_source
                self.completion_validation_instruction_span = proposed_span
        self.num_completion_proposals += 1

    def is_completion_verified(self, subtask_index: Optional[int]) -> bool:
        return self.completion_status == "verified" and self.completion_subtask_index == subtask_index

    def commit_completion(self, subtask_index: Optional[int], pose: Any, step: int) -> None:
        self.last_verified_pose = pose
        self.completion_status = "none"
        self.completion_step = None
        self.completion_subtask_index = None
        self.completion_evidence = []
        self.completion_validation_evidence = []
        self.completion_validation_target = None
        self.completion_validation_stage = None
        self.completion_validation_source = None
        self.completion_validation_instruction_span = None
        record = self._new_belief("PROGRESS_STAGE", subtask_index, "Subtask progress verified by independent completion evidence", step)
        record.status = "verified"
        record.supporting_evidence.append({"source": "verified_completion"})
        self.beliefs.append(record)

    def record_invalid_decision(self) -> None:
        self.invalid_decision_count += 1

    def rollback_to_stage(self, target_index: int) -> Dict[str, Any]:
        preserved = []
        invalidated = []
        for belief in self.beliefs:
            stage = belief.subtask_index
            if stage is None or stage < target_index:
                if belief.status == "verified":
                    preserved.append(belief.belief_id)
                continue
            if belief.status in {"active", "hypothesized", "provisional"}:
                belief.status = "rejected"
                invalidated.append(belief.belief_id)
        if self.repair_subtask_index is not None and self.repair_subtask_index >= target_index:
            if self.repair_status == "provisional":
                self.num_rejected_repairs += 1
            self._clear_repair()
        if self.completion_subtask_index is not None and self.completion_subtask_index >= target_index:
            self.completion_status = "none"
            self.completion_step = None
            self.completion_subtask_index = None
            self.completion_evidence = []
            self.completion_validation_evidence = []
            self.completion_validation_target = None
            self.completion_validation_stage = None
            self.completion_validation_source = None
            self.completion_validation_instruction_span = None
        event = {"rollback_target_stage": target_index,
                 "beliefs_preserved": preserved, "beliefs_invalidated": invalidated}
        self.rollback_events.append(event)
        self.rollback_count += 1
        return event

    def _clear_repair(self) -> None:
        self.failed_belief_id = None
        self.replacement_belief_id = None
        self.repair_evidence = []
        self.validation_evidence = []
        self.repair_status = "none"
        self.repair_step = None
        self.repair_subtask_index = None
        self.validation_target = None
        self.validation_target_stage = None
        self.validation_target_source = None
        self.validation_target_instruction_span = None

    def _new_belief(self, belief_type: str, subtask_index: Optional[int], proposition: str, step: int) -> BeliefRecord:
        return BeliefRecord(belief_id=f"b{len(self.beliefs) + 1:04d}",
                            belief_type=self._normalize_type(belief_type),
                            subtask_index=subtask_index, proposition=proposition,
                            created_step=step, last_updated_step=step)

    def _upsert_active_belief(self, belief_type: str, subtask_index: Optional[int], proposition: str,
                              evidence: List[Any], confidence: float, step: int) -> BeliefRecord:
        existing = self._find_matching_belief(belief_type, subtask_index, proposition)
        if existing is None:
            existing = self._new_belief(belief_type, subtask_index, proposition, step)
            self.beliefs.append(existing)
        if existing.status not in {"verified", "contradicted", "rejected"}:
            existing.status = "active"
        existing.confidence = confidence
        existing.supporting_evidence.extend(evidence)
        existing.last_updated_step = step
        return existing

    def _find_matching_belief(self, belief_type: str, subtask_index: Optional[int], proposition: str) -> Optional[BeliefRecord]:
        belief_type = self._normalize_type(belief_type)
        for belief in reversed(self.beliefs):
            if belief.belief_type == belief_type and belief.subtask_index == subtask_index:
                if not proposition or belief.proposition == proposition:
                    return belief
        return None

    def _belief_by_id(self, belief_id: Optional[str]) -> Optional[BeliefRecord]:
        return next((belief for belief in self.beliefs if belief.belief_id == belief_id), None)

    def _belief_to_dict(self, belief_id: Optional[str]) -> Optional[Dict[str, Any]]:
        belief = self._belief_by_id(belief_id)
        return belief.to_dict() if belief is not None else None

    def _belief_type(self, belief_id: Optional[str]) -> Optional[str]:
        belief = self._belief_by_id(belief_id)
        return belief.belief_type if belief is not None else None

    def _normalize_type(self, value: Any) -> str:
        normalized = _text(value or "PROGRESS_STAGE").upper()
        return normalized if normalized in ERROR_TYPES else "PROGRESS_STAGE"

    def _safe_confidence(self, value: Any) -> float:
        try:
            return max(0.0, min(1.0, float(value)))
        except Exception:
            return 0.0

    def _safe_int(self, value: Any) -> Optional[int]:
        try:
            return int(value) if value is not None else None
        except Exception:
            return None

    def _valid_target_source(self, value: Any) -> Optional[str]:
        source = _text(value).casefold()
        return source if source in VALIDATION_TARGET_SOURCES else None
