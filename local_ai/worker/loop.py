"""Agent loop for the one running task. The slot calls run_task. This module does not import dispatcher."""

import hashlib
import json
import logging
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from local_ai.blackboard.pins import Blackboard
from local_ai.contracts import (
    MAX_IDENTICAL_FAILURES,
    MAX_ROUNDS_PER_STEP,
    AccessGrant,
    Completion,
    ReasoningNote,
    PlanStep,
    StepDraft,
    TaskPin,
    ToolCall,
    ToolEnvelope,
)
from local_ai.llm.port import LlmPort, split_note
from local_ai.plan.steps import PlanStore
from local_ai.prompt.assemble import PromptAssembler
from local_ai.tools.shell import ShellTool


logger = logging.getLogger(__name__)

_NUMBERED_STEP = re.compile(r"^(\d+)\.\s+(\S.*?)\s*$")
_NAMED_OUTPUT = re.compile(r"\b[\w][\w.-]*\.(?:md|txt|py|pdf)\b")
_TEXT_SUFFIXES = frozenset({".md", ".txt", ".py"})
_COMMAND_CLIP = 180
_NOTE_CLIP = 400
_MAX_SHELL_CONTINUATIONS = 3
_SHELL_PLAN_NOT_THE_GOAL = "shell_plan_not_the_goal"
_MIN_STEP_ORDER = 1


class Worker:
    """
    Plans, runs the shell, and revises the plan after a failed command.

    Collaborators: PromptAssembler, LlmPort, ShellTool, PlanStore, Blackboard.
    The slot owns the thread. Worker returns "completed" or "failed".
    Slot then calls DispatcherService.finish.
    """

    split_note = staticmethod(split_note)

    def __init__(
        self,
        llm: LlmPort,
        shell: ShellTool,
        prompt: PromptAssembler,
        plan: PlanStore,
        blackboard: Blackboard,
        connection: sqlite3.Connection,
    ) -> None:
        """
        Store collaborators. Do not start a task here.

        :param llm: model port as LlmPort. Example: ScriptedLLM(())
        :param shell: granted shell as ShellTool. Example: ShellTool(GrantChecker())
        :param prompt: prompt builder as PromptAssembler. Example: PromptAssembler()
        :param plan: step store as PlanStore. Example: PlanStore(database)
        :param blackboard: pin store as Blackboard. Example: Blackboard(database)
        :param connection: shared sqlite connection as sqlite3.Connection. Example: the Database connection
        """
        self._llm = llm
        self._shell = shell
        self._prompt = prompt
        self._plan = plan
        self._blackboard = blackboard
        self._connection = connection
        self._last_envelope: ToolEnvelope | None = None
        self._last_command: str | None = None
        self._round_had_shell = False
        self._step_envelope: ToolEnvelope | None = None
        self._failed_command: str | None = None
        self._failed_error = ""
        self._identical: dict[str, tuple[str, int]] = {}
        self._seed_command: str | None = None
        self._shell_only_plan = False
        self.failure_reason = ""
        self.trace: list[dict[str, str]] = []

    def run_task(self, task_id: str) -> str:
        """
        Run one pin until it completes or fails.

        :param task_id: pin id as str. Example: "7c1e0a2e-3b4d-4f5a-9c8d-1a2b3c4d5e6f"
        :return: final status as str. Example: "completed"
        :raises ValueError: when task_id is blank or the pin is missing. Example: ""
        """
        logger.info("Starting run_task task_id=%s", task_id)
        _require_task_id(task_id)
        self.failure_reason = ""
        self._shell_only_plan = False
        if not self._store_planned_steps(task_id):
            self.failure_reason = "empty_plan"
            return "failed"
        outcome = self._run_pending_steps(task_id, self._require_pin(task_id).model)
        if outcome == "completed" and self._shell_only_plan:
            outcome = self._continue_after_shell(task_id)
        outcome = self._reject_if_result_missing(task_id, outcome)
        logger.info("worker.task | finished | task_id=%s status=%s", task_id, outcome)
        return outcome

    def parse_plan(self, completion: Completion) -> tuple[StepDraft, ...]:
        """
        Read step drafts from one planning completion.

        :param completion: planning turn as Completion. Example: Completion(content="1. List chapter files")
        :return: drafts as tuple[StepDraft, ...]. Example: (StepDraft(1, "List chapter files"),)
        :raises ValueError: when the completion names no step. Example: Completion()
        :raises TypeError: when completion is not a Completion. Example: None
        """
        drafts = _drafts_from_completion(completion)
        if not drafts:
            logger.info("Worker.parse_plan rejected reason=empty")
            raise ValueError("completion names no step. Example: Completion()")
        logger.info("Worker.parse_plan count=%s", len(drafts))
        return tuple(drafts)

    def run_round(self, task_id: str, order: int, prompt: str, model_name: str) -> Completion:
        """
        One model turn, then at most one shell call.

        :param task_id: pin id as str. Example: "7c1e0a2e-3b4d-4f5a-9c8d-1a2b3c4d5e6f"
        :param order: step order as int. Example: 2
        :param prompt: assembled prompt as str. Example: "Foundation\\nShell\\nTask"
        :param model_name: logical size as str. Example: "qwen-14b"
        :return: the model turn as Completion. Example: Completion(final_answer="Found ch-01.md")
        :raises ValueError: when task_id, order, or model_name is unusable. Example: ""
        :raises TypeError: when prompt is not a str. Example: None
        """
        logger.info("Starting run_round task_id=%s step=%s", task_id, order)
        _require_round_args(task_id, order, prompt, model_name)
        completion = self._llm.complete(prompt, model_name)
        self._append_note(task_id, order, completion)
        self._run_requested_shell(task_id, order, completion)
        logger.info("run_round finished task_id=%s step=%s", task_id, order)
        return completion

    def run_step(self, task_id: str, order: int, model_name: str) -> str:
        """
        Drive one step for at most MAX_ROUNDS_PER_STEP rounds.

        :param task_id: pin id as str. Example: "7c1e0a2e-3b4d-4f5a-9c8d-1a2b3c4d5e6f"
        :param order: step order as int. Example: 2
        :param model_name: logical size as str. Example: "qwen-14b"
        :return: step outcome as str. Example: "completed"
        :raises ValueError: when task_id, order, or model_name is unusable. Example: ""
        """
        logger.info("Starting run_step task_id=%s step=%s", task_id, order)
        _require_step_args(task_id, order, model_name)
        self._open_step(task_id, order)
        seeded = self._run_seed_command(task_id, order)
        if seeded is not None:
            return seeded
        for _round_index in range(_ROUNDS):
            outcome = self._one_step_round(task_id, order, model_name)
            if outcome is not None:
                return outcome
        return self._fail_open_step(task_id, order, "max rounds without a final answer")

    def force_final(self, task_id: str, order: int, model_name: str) -> str:
        """
        Ask once for a final answer. An empty answer fails the pin.

        :param task_id: pin id as str. Example: "7c1e0a2e-3b4d-4f5a-9c8d-1a2b3c4d5e6f"
        :param order: step order as int. Example: 2
        :param model_name: logical size as str. Example: "qwen-8b"
        :return: "completed" or "failed" as str. Example: "failed"
        :raises ValueError: when task_id, order, or model_name is unusable. Example: ""
        """
        logger.info("Starting force_final task_id=%s step=%s", task_id, order)
        _require_step_args(task_id, order, model_name)
        completion = self._llm.complete(self._final_prompt(task_id, order), model_name)
        return self._outcome_from_final(task_id, order, completion)

    def adapt_after_failure(self, task_id: str, order: int, model_name: str, error: str) -> None:
        """
        Leave the failed step failed and insert one pending replacement.

        :param task_id: pin id as str. Example: "7c1e0a2e-3b4d-4f5a-9c8d-1a2b3c4d5e6f"
        :param order: failed step order as int. Example: 2
        :param model_name: logical size as str. Example: "qwen-24b"
        :param error: tool error as str. Example: "weasyprint: CSS path missing"
        :return: None. Example: None
        :raises ValueError: when task_id, order, or model_name is unusable, or the action is empty. Example: ""
        :raises TypeError: when error is not a str. Example: None
        """
        logger.info("Starting adapt_after_failure task_id=%s step=%s", task_id, order)
        _require_adapt_args(task_id, order, model_name, error)
        self._publish_phase(task_id, "recovering")
        new_order = self._insert_replacement(task_id, order, model_name, error)
        if new_order is None:
            return
        self._publish_phase(task_id, "acting")
        logger.info("worker.adapt | inserted | task_id=%s step=%s", task_id, new_order)

    def note_identical_failure(self, task_id: str, order: int, command: str) -> bool:
        """
        Count identical command failures. The third one stops the pin.

        :param task_id: pin id as str. Example: "7c1e0a2e-3b4d-4f5a-9c8d-1a2b3c4d5e6f"
        :param order: step order as int. Example: 2
        :param command: shell command as str. Example: "weasyprint book.html book.pdf"
        :return: True when the pin must stop as bool. Example: False
        :raises ValueError: when task_id, order, or command is unusable. Example: ""
        """
        logger.info("Starting note_identical_failure task_id=%s step=%s", task_id, order)
        _require_failure_args(task_id, order, command)
        count = self._next_failure_count(task_id, command)
        _log_failure_count(task_id, order, command, count)
        return count >= _IDENTICAL

    def _append_note(self, task_id: str, order: int, completion: Completion) -> None:
        """
        Store a non-empty reasoning note on the pin.

        :param task_id: pin id as str. Example: "7c1e0a2e-3b4d-4f5a-9c8d-1a2b3c4d5e6f"
        :param order: step order as int. Example: 2
        :param completion: model turn as Completion. Example: Completion(thinking="Writing the marker.")
        :return: None. Example: None
        """
        note = split_note(completion)
        if not note.strip():
            logger.info("run_round note empty task_id=%s step=%s", task_id, order)
            return
        self._blackboard.append_reasoning(
            self._connection,
            task_id,
            ReasoningNote(_utc_now(), order, note),
        )

    def _run_requested_shell(self, task_id: str, order: int, completion: Completion) -> None:
        """
        Run one shell command when the turn asks for it.

        :param task_id: pin id as str. Example: "7c1e0a2e-3b4d-4f5a-9c8d-1a2b3c4d5e6f"
        :param order: step order as int. Example: 2
        :param completion: model turn as Completion. Example: Completion(tool_calls=(ToolCall("shell", "printf marker"),))
        :return: None. Example: None
        """
        command = _shell_command(completion)
        self._round_had_shell = command is not None
        if command is None:
            return
        self._execute_shell(task_id, order, command)

    def _execute_shell(self, task_id: str, order: int, command: str) -> None:
        """
        Run the command in the job directory and keep the envelope.

        :param task_id: pin id as str. Example: "7c1e0a2e-3b4d-4f5a-9c8d-1a2b3c4d5e6f"
        :param order: step order as int. Example: 2
        :param command: shell command as str. Example: "printf marker"
        :return: None. Example: None
        """
        pin = self._require_pin(task_id)
        job_dir = _job_dir(self._connection, task_id)
        envelope = self._shell.run(_shell_grant(pin, job_dir), command, job_dir)
        self._last_envelope = envelope
        self._step_envelope = envelope
        self._last_command = command
        self._record_shell_note(task_id, order, command, envelope)
        self.trace.append({"tool": "shell"})
        logger.info("worker.step | acting | task_id=%s step=%s tool=shell", task_id, order)

    def _require_pin(self, task_id: str) -> TaskPin:
        """
        Load the pin or reject a missing id.

        :param task_id: pin id as str. Example: "7c1e0a2e-3b4d-4f5a-9c8d-1a2b3c4d5e6f"
        :return: pin as TaskPin. Example: a pin with status "running"
        :raises ValueError: when the pin is missing. Example: "missing-id"
        """
        pin = self._blackboard.get_pin(self._connection, task_id)
        if pin is None:
            raise ValueError(f"pin not found task_id={task_id}")
        return pin

    def _open_step(self, task_id: str, order: int) -> None:
        """
        Mark the step running and publish the acting phase.

        :param task_id: pin id as str. Example: "7c1e0a2e-3b4d-4f5a-9c8d-1a2b3c4d5e6f"
        :param order: step order as int. Example: 2
        :return: None. Example: None
        """
        self._failed_command = None
        self._failed_error = ""
        self._step_envelope = None
        self._plan.mark_running(self._connection, task_id, order)
        self._publish_phase(task_id, "acting")

    def _one_step_round(self, task_id: str, order: int, model_name: str) -> str | None:
        """
        Take one round. Return a terminal outcome, or None to continue.

        :param task_id: pin id as str. Example: "7c1e0a2e-3b4d-4f5a-9c8d-1a2b3c4d5e6f"
        :param order: step order as int. Example: 2
        :param model_name: logical size as str. Example: "qwen-14b"
        :return: outcome or None as str | None. Example: "completed"
        """
        completion = self.run_round(task_id, order, self._step_prompt(task_id, order), model_name)
        if _shell_failed(self._round_had_shell, self._last_envelope):
            error = self._last_envelope.error if self._last_envelope is not None else ""
            return self._fail_open_step(task_id, order, error)
        if completion.final_answer.strip():
            return self._complete_open_step(task_id, order, completion.final_answer.strip())
        if not self._round_had_shell:
            return self.force_final(task_id, order, model_name)
        return None

    def _final_prompt(self, task_id: str, order: int) -> str:
        """
        Ask the model for a final answer on top of the step prompt.

        :param task_id: pin id as str. Example: "7c1e0a2e-3b4d-4f5a-9c8d-1a2b3c4d5e6f"
        :param order: step order as int. Example: 2
        :return: prompt as str. Example: "Foundation\\n\\nReply with a final answer for this step."
        """
        return self._step_prompt(task_id, order) + "\n\nReply with a final answer for this step."

    def _outcome_from_final(self, task_id: str, order: int, completion: Completion) -> str:
        """
        Complete the step when the extra turn has a final answer.

        :param task_id: pin id as str. Example: "7c1e0a2e-3b4d-4f5a-9c8d-1a2b3c4d5e6f"
        :param order: step order as int. Example: 2
        :param completion: extra turn as Completion. Example: Completion(final_answer="Found ch-01.md")
        :return: "completed" or "failed" as str. Example: "failed"
        """
        answer = completion.final_answer.strip()
        if answer:
            logger.info("worker.force_final | completed | task_id=%s step=%s", task_id, order)
            return self._complete_open_step(task_id, order, answer)
        logger.info("worker.force_final | failed | task_id=%s step=%s", task_id, order)
        return "failed"

    def _step_prompt(self, task_id: str, order: int) -> str:
        """
        Assemble the step prompt, including the previous synthesis.

        :param task_id: pin id as str. Example: "7c1e0a2e-3b4d-4f5a-9c8d-1a2b3c4d5e6f"
        :param order: step order as int. Example: 2
        :return: prompt as str. Example: "Foundation\\nShell\\nTask"
        """
        pin = self._require_pin(task_id)
        steps = self._plan.list_steps(self._connection, task_id)
        return self._prompt.assemble(
            pin.goal,
            pin.done_when,
            pin.access,
            (),
            self._plan.render_plan(steps),
            _prior_synthesis(steps, order),
        )

    def _publish_phase(self, task_id: str, phase: str) -> None:
        """
        Write the phase and the readable plan onto the pin.

        :param task_id: pin id as str. Example: "7c1e0a2e-3b4d-4f5a-9c8d-1a2b3c4d5e6f"
        :param phase: phase word as str. Example: "acting"
        :return: None. Example: None
        """
        plan_text = self._plan.render_plan(self._plan.list_steps(self._connection, task_id))
        self._blackboard.set_phase(self._connection, task_id, phase, plan_text)
        if phase == "planning":
            self.trace.append({"workflow": "planning"})

    def _refresh_plan(self, task_id: str) -> None:
        """
        Rewrite current_plan from the step rows without changing the phase.

        :param task_id: pin id as str. Example: "7c1e0a2e-3b4d-4f5a-9c8d-1a2b3c4d5e6f"
        :return: None. Example: None
        """
        self._publish_phase(task_id, self._require_pin(task_id).phase)

    def _record_shell_note(self, task_id: str, order: int, command: str, envelope: ToolEnvelope) -> None:
        """
        Put the command and its output on the pin.

        :param task_id: pin id as str. Example: "7c1e0a2e-3b4d-4f5a-9c8d-1a2b3c4d5e6f"
        :param order: step order as int. Example: 1
        :param command: shell command as str. Example: "pdftotext book.pdf -"
        :param envelope: shell result as ToolEnvelope. Example: ToolEnvelope(False, None, "missing pdf")
        :return: None. Example: None
        """
        self._blackboard.append_reasoning(
            self._connection,
            task_id,
            ReasoningNote(_utc_now(), order, _shell_note(order, command, envelope)),
        )

    def _reject_if_result_missing(self, task_id: str, outcome: str) -> str:
        """
        Fail a completed run when a file named in done_when is missing or blank.

        :param task_id: pin id as str. Example: "7c1e0a2e-3b4d-4f5a-9c8d-1a2b3c4d5e6f"
        :param outcome: step-loop status as str. Example: "completed"
        :return: status as str. Example: "failed"
        """
        if outcome != "completed":
            return outcome
        missing = _missing_named_output(self._require_pin(task_id).done_when, _job_dir(self._connection, task_id))
        if not missing:
            return outcome
        self.failure_reason = missing
        self._blackboard.append_reasoning(self._connection, task_id, ReasoningNote(_utc_now(), 0, missing))
        logger.info("worker.task | result missing | task_id=%s", task_id)
        return "failed"

    def _complete_open_step(self, task_id: str, order: int, synthesis: str) -> str:
        """
        Store the handoff and record the completed step.

        :param task_id: pin id as str. Example: "7c1e0a2e-3b4d-4f5a-9c8d-1a2b3c4d5e6f"
        :param order: step order as int. Example: 2
        :param synthesis: next-step handoff as str. Example: "Found ch-01.md"
        :return: "completed" as str. Example: "completed"
        """
        self._plan.complete_step(
            self._connection,
            task_id,
            order,
            synthesis,
            _envelope_json(self._step_envelope),
        )
        self._refresh_plan(task_id)
        self.trace.append({"effect": "step_completed"})
        logger.info("worker.step | completed | task_id=%s step=%s", task_id, order)
        return "completed"

    def _fail_open_step(self, task_id: str, order: int, error: str) -> str:
        """
        Mark the running step failed. A failed shell remembers the command.

        :param task_id: pin id as str. Example: "7c1e0a2e-3b4d-4f5a-9c8d-1a2b3c4d5e6f"
        :param order: step order as int. Example: 2
        :param error: step error as str. Example: "boom"
        :return: "failed" as str. Example: "failed"
        """
        self._plan.fail_step(self._connection, task_id, order, error)
        if _shell_failed(self._round_had_shell, self._last_envelope) and self._last_command is not None:
            self._failed_command = self._last_command
            self._failed_error = error
        self._refresh_plan(task_id)
        logger.info("worker.step | failed | task_id=%s step=%s", task_id, order)
        return "failed"

    def _store_planned_steps(self, task_id: str) -> bool:
        """
        Ask for a plan and insert it. An empty plan fails the pin.

        :param task_id: pin id as str. Example: "7c1e0a2e-3b4d-4f5a-9c8d-1a2b3c4d5e6f"
        :return: True when steps were stored as bool. Example: True
        """
        pin = self._require_pin(task_id)
        self._publish_phase(task_id, "planning")
        completion = self._llm.complete(self._planning_prompt(pin), pin.model)
        if not self._insert_parsed_plan(task_id, completion):
            return False
        self._publish_phase(task_id, "planning")
        return True

    def _insert_parsed_plan(self, task_id: str, completion: Completion) -> bool:
        """
        Parse and insert drafts. ValueError means the pin stops.

        :param task_id: pin id as str. Example: "7c1e0a2e-3b4d-4f5a-9c8d-1a2b3c4d5e6f"
        :param completion: planning turn as Completion. Example: Completion(content="1. List chapter files")
        :return: True when rows were inserted as bool. Example: True
        """
        try:
            drafts = self.parse_plan(completion)
        except ValueError:
            drafts = self._draft_from_shell(completion)
            if drafts is None:
                logger.info("worker.task | failed | task_id=%s reason=empty_plan", task_id)
                return False
        self._plan.insert_steps(self._connection, task_id, drafts)
        return True

    def _draft_from_shell(self, completion: Completion) -> tuple[StepDraft, ...] | None:
        """
        Turn a planning tool call into one step when the model skipped the numbered plan.

        :param completion: planning turn as Completion. Example: Completion(tool_calls=(ToolCall("shell", "printf marker"),))
        :return: one draft or None as tuple[StepDraft, ...] | None. Example: (StepDraft(1, "Run the shell command from the planning turn"),)
        """
        command = _shell_command(completion)
        if not command:
            return None
        self._seed_command = command
        self._shell_only_plan = True
        logger.info("worker.task | plan_from_shell")
        return (StepDraft(1, "Run the shell command from the planning turn"),)

    def _run_seed_command(self, task_id: str, order: int) -> str | None:
        """
        Run the shell command captured during planning, then finish that step.

        :param task_id: pin id as str. Example: "7c1e0a2e-3b4d-4f5a-9c8d-1a2b3c4d5e6f"
        :param order: step order as int. Example: 1
        :return: step outcome or None as str | None. Example: "completed"
        """
        command = self._seed_command
        if not command:
            return None
        self._seed_command = None
        self._run_requested_shell(
            task_id,
            order,
            Completion(tool_calls=(ToolCall("shell", command),)),
        )
        if _shell_failed(self._round_had_shell, self._last_envelope):
            error = self._last_envelope.error if self._last_envelope is not None else ""
            return self._fail_open_step(task_id, order, error)
        return self._complete_open_step(task_id, order, "Ran the shell command from the planning turn")

    def _continue_after_shell(self, task_id: str) -> str:
        """
        Ask for the rest of the goal after a shell-only plan.

        :param task_id: pin id as str. Example: "7c1e0a2e-3b4d-4f5a-9c8d-1a2b3c4d5e6f"
        :return: status as str. Example: "failed"
        """
        model_name = self._require_pin(task_id).model
        for _index in range(_MAX_SHELL_CONTINUATIONS):
            outcome = self._follow_shell_plan(task_id, model_name)
            if outcome != "again":
                return outcome
        return self._stop_unfinished_shell_plan(task_id)

    def _follow_shell_plan(self, task_id: str, model_name: str) -> str:
        """
        Take one follow-up turn. A further shell command means ask again.

        :param task_id: pin id as str. Example: "7c1e0a2e-3b4d-4f5a-9c8d-1a2b3c4d5e6f"
        :param model_name: logical size as str. Example: "qwen-14b"
        :return: status or "again" as str. Example: "again"
        """
        completion = self._llm.complete(self._continuation_prompt(task_id), model_name)
        kind = self._store_follow_up(task_id, completion)
        if kind == "none":
            return self._close_or_stop_shell_plan(task_id, completion)
        outcome = self._run_pending_steps(task_id, model_name)
        if outcome != "completed" or kind == "numbered":
            return outcome
        return "again"

    def _store_follow_up(self, task_id: str, completion: Completion) -> str:
        """
        Insert the remaining plan, or one more shell step.

        :param task_id: pin id as str. Example: "7c1e0a2e-3b4d-4f5a-9c8d-1a2b3c4d5e6f"
        :param completion: follow-up turn as Completion. Example: Completion(content="1. Write the thesis")
        :return: "numbered", "shell", or "none" as str. Example: "numbered"
        """
        drafts = _drafts_from_completion(completion)
        if drafts:
            self._shell_only_plan = False
            self._insert_follow_up(task_id, _shifted_drafts(drafts, self._next_order(task_id)))
            return "numbered"
        command = _shell_command(completion)
        if not command:
            return "none"
        self._seed_command = command
        self._shell_only_plan = True
        self._insert_follow_up(task_id, (StepDraft(self._next_order(task_id), "Run the follow-up shell command"),))
        return "shell"

    def _insert_follow_up(self, task_id: str, drafts: tuple[StepDraft, ...]) -> None:
        """
        Store follow-up steps and publish the plan.

        :param task_id: pin id as str. Example: "7c1e0a2e-3b4d-4f5a-9c8d-1a2b3c4d5e6f"
        :param drafts: new steps as tuple[StepDraft, ...]. Example: (StepDraft(2, "Write the thesis"),)
        :return: None. Example: None
        """
        self._plan.insert_steps(self._connection, task_id, drafts)
        self._publish_phase(task_id, "planning")
        logger.info("worker.task | shell_follow_up | task_id=%s count=%s", task_id, len(drafts))

    def _close_or_stop_shell_plan(self, task_id: str, completion: Completion) -> str:
        """
        Accept a final answer. An empty follow-up means the goal is still open.

        :param task_id: pin id as str. Example: "7c1e0a2e-3b4d-4f5a-9c8d-1a2b3c4d5e6f"
        :param completion: follow-up turn as Completion. Example: Completion(final_answer="The thesis is written.")
        :return: status as str. Example: "completed"
        """
        if completion.final_answer.strip():
            logger.info("worker.task | shell_plan_closed | task_id=%s", task_id)
            return "completed"
        return self._stop_unfinished_shell_plan(task_id)

    def _stop_unfinished_shell_plan(self, task_id: str) -> str:
        """
        Fail the pin because a finished command is not the goal.

        :param task_id: pin id as str. Example: "7c1e0a2e-3b4d-4f5a-9c8d-1a2b3c4d5e6f"
        :return: "failed" as str. Example: "failed"
        """
        self.failure_reason = _SHELL_PLAN_NOT_THE_GOAL
        self._blackboard.append_reasoning(
            self._connection,
            task_id,
            ReasoningNote(_utc_now(), 0, "The shell command finished. The goal is not met."),
        )
        logger.info("worker.task | shell_plan_open | task_id=%s", task_id)
        return "failed"

    def _next_order(self, task_id: str) -> int:
        """
        Return the order for the next inserted step.

        :param task_id: pin id as str. Example: "7c1e0a2e-3b4d-4f5a-9c8d-1a2b3c4d5e6f"
        :return: order as int. Example: 2
        """
        steps = self._plan.list_steps(self._connection, task_id)
        if not steps:
            return _MIN_STEP_ORDER
        return steps[-1].order + 1

    def _continuation_prompt(self, task_id: str) -> str:
        """
        Tell the model the shell command finished and the goal is still open.

        :param task_id: pin id as str. Example: "7c1e0a2e-3b4d-4f5a-9c8d-1a2b3c4d5e6f"
        :return: prompt as str. Example: "Foundation\\n\\nThe planning turn was only a shell command."
        """
        prompt = self._planning_prompt(self._require_pin(task_id))
        return (
            prompt
            + "\n\nThe planning turn was only a shell command, and that command has finished. "
            + "It is not the goal. Reply with the remaining numbered steps. "
            + "Reply with a final answer only when the goal is already met."
        )

    def _planning_prompt(self, pin: TaskPin) -> str:
        """
        Assemble the planning prompt from the pin.

        :param pin: live pin as TaskPin. Example: a pin with goal "Write a marker file in the job folder."
        :return: prompt as str. Example: "Foundation\\nShell\\nTask"
        """
        return self._prompt.assemble(
            pin.goal,
            pin.done_when,
            pin.access,
            (),
            pin.current_plan,
            "",
        )

    def _run_pending_steps(self, task_id: str, model_name: str) -> str:
        """
        Run each pending step. Stop when one cannot be recovered.

        :param task_id: pin id as str. Example: "7c1e0a2e-3b4d-4f5a-9c8d-1a2b3c4d5e6f"
        :param model_name: logical size as str. Example: "qwen-14b"
        :return: "completed" or "failed" as str. Example: "completed"
        """
        while True:
            pending = self._next_pending(task_id)
            if pending is None:
                return "completed"
            if self._run_or_adapt(task_id, pending.order, model_name) == "failed":
                return "failed"

    def _next_pending(self, task_id: str) -> PlanStep | None:
        """
        Return the lowest-order pending step.

        :param task_id: pin id as str. Example: "7c1e0a2e-3b4d-4f5a-9c8d-1a2b3c4d5e6f"
        :return: step or None as PlanStep | None. Example: None
        """
        for step in self._plan.list_steps(self._connection, task_id):
            if step.status == "pending":
                return step
        return None

    def _run_or_adapt(self, task_id: str, order: int, model_name: str) -> str:
        """
        Run one step. A shell failure adapts unless the command has failed three times.

        :param task_id: pin id as str. Example: "7c1e0a2e-3b4d-4f5a-9c8d-1a2b3c4d5e6f"
        :param order: step order as int. Example: 1
        :param model_name: logical size as str. Example: "qwen-14b"
        :return: "completed", "failed", or "adapted" as str. Example: "completed"
        """
        if self.run_step(task_id, order, model_name) == "completed":
            return "completed"
        command = self._failed_command
        if command is None or self.note_identical_failure(task_id, order, command):
            return "failed"
        self.adapt_after_failure(task_id, order, model_name, self._failed_error)
        if self.failure_reason:
            return "failed"
        return "adapted"

    def _insert_replacement(self, task_id: str, order: int, model_name: str, error: str) -> int | None:
        """
        Ask for one replacement sentence and insert it after the failed step.

        :param task_id: pin id as str. Example: "7c1e0a2e-3b4d-4f5a-9c8d-1a2b3c4d5e6f"
        :param order: failed step order as int. Example: 2
        :param model_name: logical size as str. Example: "qwen-24b"
        :param error: tool error as str. Example: "weasyprint: CSS path missing"
        :return: new order, or None when there is nothing to run. Example: 3
        """
        completion = self._llm.complete(self._recovery_prompt(task_id, order, error), model_name)
        action = _replacement_action(completion, self)
        if action is None:
            self.failure_reason = error.strip() or "empty_plan"
            logger.info("worker.adapt | no replacement | task_id=%s", task_id)
            return None
        return self._plan.insert_after(self._connection, task_id, order, action)

    def _recovery_prompt(self, task_id: str, order: int, error: str) -> str:
        """
        Ask for one replacement action. The error is in the prompt, not the log.

        :param task_id: pin id as str. Example: "7c1e0a2e-3b4d-4f5a-9c8d-1a2b3c4d5e6f"
        :param order: failed step order as int. Example: 2
        :param error: tool error as str. Example: "weasyprint: CSS path missing"
        :return: prompt as str. Example: "Foundation\\n\\nThe command failed."
        """
        logger.info(
            "worker.adapt | recovering | task_id=%s step=%s error_len=%s",
            task_id,
            order,
            len(error),
        )
        return (
            self._step_prompt(task_id, order)
            + "\n\nThe command failed.\n"
            + error
            + "\nReply with one replacement action."
        )

    def _next_failure_count(self, task_id: str, command: str) -> int:
        """
        Increment the count when the command matches. A new command starts at one.

        :param task_id: pin id as str. Example: "7c1e0a2e-3b4d-4f5a-9c8d-1a2b3c4d5e6f"
        :param command: shell command as str. Example: "weasyprint book.html book.pdf"
        :return: count as int. Example: 3
        """
        previous, count = self._identical.get(task_id, ("", 0))
        count = count + 1 if command == previous else 1
        self._identical[task_id] = (command, count)
        return count


_ROUNDS = MAX_ROUNDS_PER_STEP
_IDENTICAL = MAX_IDENTICAL_FAILURES


def _require_task_id(task_id: str) -> None:
    """
    Reject a blank pin id.

    :param task_id: pin id as str. Example: "7c1e0a2e-3b4d-4f5a-9c8d-1a2b3c4d5e6f"
    :return: None. Example: None
    :raises ValueError: when task_id is blank. Example: ""
    :raises TypeError: when task_id is not a str. Example: None
    """
    if not isinstance(task_id, str):
        raise TypeError("task_id must be str. Example: '7c1e0a2e-3b4d-4f5a-9c8d-1a2b3c4d5e6f'")
    if not task_id.strip():
        raise ValueError("task_id must be a non-empty str. Example: '7c1e0a2e-3b4d-4f5a-9c8d-1a2b3c4d5e6f'")


def _require_adapt_args(task_id: str, order: int, model_name: str, error: str) -> None:
    """
    Reject an adapt call the plan store cannot apply.

    :param task_id: pin id as str. Example: "7c1e0a2e-3b4d-4f5a-9c8d-1a2b3c4d5e6f"
    :param order: failed step order as int. Example: 2
    :param model_name: logical size as str. Example: "qwen-24b"
    :param error: tool error as str. Example: "weasyprint: CSS path missing"
    :return: None. Example: None
    :raises ValueError: when task_id, order, or model_name is unusable. Example: ""
    :raises TypeError: when error is not a str. Example: None
    """
    _require_step_args(task_id, order, model_name)
    if not isinstance(error, str):
        raise TypeError("error must be str. Example: 'weasyprint: CSS path missing'")


def _shell_note(order: int, command: str, envelope: ToolEnvelope) -> str:
    """
    Format one shell outcome so a read of the pin shows what ran.

    :param order: step order as int. Example: 1
    :param command: shell command as str. Example: "pdftotext book.pdf -"
    :param envelope: shell result as ToolEnvelope. Example: ToolEnvelope(False, None, "missing pdf")
    :return: note text as str. Example: "step 1 failed: pdftotext book.pdf -\\nmissing pdf"
    """
    head = _clip(command.strip(), _COMMAND_CLIP)
    if envelope.success:
        body = f"step {order}: {head}\nexit 0\n{envelope.result or ''}"
    else:
        body = f"step {order} failed: {head}\n{envelope.error}"
    return _clip(body.strip(), _NOTE_CLIP)


def _shifted_drafts(drafts: list[StepDraft], start: int) -> tuple[StepDraft, ...]:
    """
    Renumber drafts so they follow steps that are already stored.

    :param drafts: parsed drafts as list[StepDraft]. Example: [StepDraft(1, "Write the thesis")]
    :param start: first new order as int. Example: 2
    :return: drafts as tuple[StepDraft, ...]. Example: (StepDraft(2, "Write the thesis"),)
    """
    return tuple(StepDraft(start + index, draft.action) for index, draft in enumerate(drafts))


def _clip(text: str, limit: int) -> str:
    """
    Keep a note short enough to read on the pin.

    :param text: raw text as str. Example: "missing pdf"
    :param limit: max characters as int. Example: 400
    :return: clipped text as str. Example: "missing pdf"
    """
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


def _missing_named_output(done_when: str, job_dir: Path) -> str:
    """
    Name the first done_when file that is absent or blank.

    :param done_when: completion text as str. Example: "marker.txt exists"
    :param job_dir: job directory as Path. Example: Path("/tmp/jobs/id")
    :return: failure text or empty as str. Example: "marker.txt is missing or empty"
    """
    for name in _named_outputs(done_when):
        if _output_missing(job_dir / name):
            return f"{name} is missing or empty"
    return ""


def _named_outputs(done_when: str) -> tuple[str, ...]:
    """
    Pull file names out of a done_when sentence.

    :param done_when: completion text as str. Example: "summary.md exists in the job folder"
    :return: names as tuple[str, ...]. Example: ("summary.md",)
    """
    seen: list[str] = []
    for name in _NAMED_OUTPUT.findall(done_when):
        if name not in seen:
            seen.append(name)
    return tuple(seen)


def _output_missing(path: Path) -> bool:
    """
    True when the promised file is absent, zero-length, or only whitespace.

    :param path: file in the job directory as Path. Example: Path("/tmp/jobs/id/marker.txt")
    :return: missing as bool. Example: True
    """
    if not path.is_file() or path.stat().st_size == 0:
        return True
    if path.suffix.lower() not in _TEXT_SUFFIXES:
        return False
    return path.read_text(encoding="utf-8").strip() == ""


def _replacement_action(completion: Completion, worker: Worker) -> str | None:
    """
    Use the sentence, or the shell command when that is all the model returned.

    :param completion: recovery turn as Completion. Example: Completion(final_answer="Fix the CSS path")
    :param worker: worker that stores a seed command as Worker. Example: Worker(...)
    :return: action or None as str | None. Example: "Fix the CSS path"
    """
    command = _shell_command(completion)
    action = _action_sentence(completion)
    if command:
        worker._seed_command = command
        return action or "Run the replacement shell command"
    return action or None


def _action_sentence(completion: Completion) -> str:
    """
    Take the first sentence from the final answer, or from content.

    :param completion: recovery turn as Completion. Example: Completion(final_answer="Fix the CSS path")
    :return: action as str. Example: "Fix the CSS path"
    """
    text = completion.final_answer.strip() or completion.content.strip()
    line = next((item.strip() for item in text.splitlines() if item.strip()), "")
    match = _NUMBERED_STEP.match(line)
    if match is None:
        return line
    return match.group(2)


def _require_failure_args(task_id: str, order: int, command: str) -> None:
    """
    Reject a failure note that cannot be counted.

    :param task_id: pin id as str. Example: "7c1e0a2e-3b4d-4f5a-9c8d-1a2b3c4d5e6f"
    :param order: step order as int. Example: 2
    :param command: shell command as str. Example: "weasyprint book.html book.pdf"
    :return: None. Example: None
    :raises ValueError: when task_id, order, or command is unusable. Example: ""
    """
    _require_step_args(task_id, order, "qwen-14b")
    if not isinstance(command, str) or not command.strip():
        raise ValueError("command must be a non-empty str. Example: 'weasyprint book.html book.pdf'")


def _log_failure_count(task_id: str, order: int, command: str, count: int) -> None:
    """
    Log the count, the length, and a short hash. The command text stays out.

    :param task_id: pin id as str. Example: "7c1e0a2e-3b4d-4f5a-9c8d-1a2b3c4d5e6f"
    :param order: step order as int. Example: 2
    :param command: shell command as str. Example: "weasyprint book.html book.pdf"
    :param count: identical-failure count as int. Example: 3
    :return: None. Example: None
    """
    digest = hashlib.sha256(command.encode("utf-8")).hexdigest()[:12]
    logger.info(
        "worker.failure | identical | task_id=%s step=%s count=%s command_len=%s command_hash=%s",
        task_id,
        order,
        count,
        len(command),
        digest,
    )


def _require_step_args(task_id: str, order: int, model_name: str) -> None:
    """
    Reject a step address the plan store cannot mark running.

    :param task_id: pin id as str. Example: "7c1e0a2e-3b4d-4f5a-9c8d-1a2b3c4d5e6f"
    :param order: step order as int. Example: 2
    :param model_name: logical size as str. Example: "qwen-14b"
    :return: None. Example: None
    :raises ValueError: when task_id, order, or model_name is unusable. Example: ""
    """
    _require_round_args(task_id, order, "", model_name)


def _shell_failed(had_shell: bool, envelope: ToolEnvelope | None) -> bool:
    """
    Return whether this round's shell command failed.

    :param had_shell: whether the round ran the shell as bool. Example: True
    :param envelope: tool envelope as ToolEnvelope | None. Example: ToolEnvelope(False, "", "boom")
    :return: True when the shell failed as bool. Example: True
    """
    return had_shell and envelope is not None and not envelope.success


def _prior_synthesis(steps: tuple[PlanStep, ...], order: int) -> str:
    """
    Use the latest handoff from a step before this one.

    :param steps: ordered steps as tuple[PlanStep, ...]. Example: ()
    :param order: current step order as int. Example: 2
    :return: synthesis as str. Example: "Found ch-01.md"
    """
    prior = ""
    for step in steps:
        if step.order < order and step.llm_synthesis:
            prior = step.llm_synthesis
    return prior


def _envelope_json(envelope: ToolEnvelope | None) -> str:
    """
    Encode a tool envelope for the step row. None becomes an empty string.

    :param envelope: tool envelope as ToolEnvelope | None. Example: ToolEnvelope(True, "marker", "")
    :return: JSON or empty as str. Example: '{"success": true, "result": "marker", "error": ""}'
    """
    if envelope is None:
        return ""
    return json.dumps({"success": envelope.success, "result": envelope.result, "error": envelope.error})


def _require_round_args(task_id: str, order: int, prompt: str, model_name: str) -> None:
    """
    Reject a round the model and the shell cannot run.

    :param task_id: pin id as str. Example: "7c1e0a2e-3b4d-4f5a-9c8d-1a2b3c4d5e6f"
    :param order: step order as int. Example: 2
    :param prompt: assembled prompt as str. Example: "Foundation\\nShell\\nTask"
    :param model_name: logical size as str. Example: "qwen-14b"
    :return: None. Example: None
    :raises ValueError: when task_id, order, or model_name is unusable. Example: ""
    :raises TypeError: when prompt is not a str. Example: None
    """
    if not isinstance(task_id, str) or not task_id.strip():
        raise ValueError("task_id must be a non-empty str. Example: '7c1e0a2e-3b4d-4f5a-9c8d-1a2b3c4d5e6f'")
    if isinstance(order, bool) or not isinstance(order, int) or order < _MIN_STEP_ORDER:
        raise ValueError("order must be an int >= 1. Example: 2")
    if not isinstance(prompt, str):
        raise TypeError("prompt must be str. Example: 'Foundation\\nShell\\nTask'")
    if not isinstance(model_name, str) or not model_name:
        raise ValueError("model_name must be a non-empty str. Example: 'qwen-14b'")


def _shell_command(completion: Completion) -> str | None:
    """
    Return the first shell command on the turn.

    :param completion: model turn as Completion. Example: Completion(tool_calls=(ToolCall("shell", "printf marker"),))
    :return: command or None as str | None. Example: "printf marker"
    """
    for call in completion.tool_calls:
        if call.name == "shell":
            return call.command
    return None


def _shell_grant(pin: TaskPin, job_dir: Path) -> AccessGrant:
    """
    Use the job directory as the root when the pin grant is job.

    :param pin: live pin as TaskPin. Example: a pin with access mode job
    :param job_dir: job folder as Path. Example: Path("data/jobs/7c1e0a2e")
    :return: grant as AccessGrant. Example: AccessGrant(mode="job", paths=("data/jobs/7c1e0a2e",))
    """
    if pin.access.mode == "job":
        return AccessGrant(mode="job", paths=(str(job_dir),))
    return pin.access


def _job_dir(connection: sqlite3.Connection, task_id: str) -> Path:
    """
    Place the job next to the sqlite file, under jobs/task_id.

    :param connection: open connection as sqlite3.Connection. Example: the Database connection
    :param task_id: pin id as str. Example: "7c1e0a2e-3b4d-4f5a-9c8d-1a2b3c4d5e6f"
    :return: job directory as Path. Example: Path("data/jobs/7c1e0a2e-3b4d-4f5a-9c8d-1a2b3c4d5e6f")
    """
    return _database_file(connection).parent / "jobs" / task_id


def _database_file(connection: sqlite3.Connection) -> Path:
    """
    Read the main sqlite file path from the open connection.

    :param connection: open connection as sqlite3.Connection. Example: the Database connection
    :return: database path as Path. Example: Path("data/local-ai.sqlite")
    :raises ValueError: when sqlite has no file path. Example: an unnamed connection
    """
    row = connection.execute("PRAGMA database_list").fetchone()
    if row is None or not row[2]:
        raise ValueError("sqlite database path is missing")
    return Path(str(row[2]))


def _utc_now() -> str:
    """
    Return the current UTC time as a timestamp string.

    :return: timestamp as str. Example: "2026-10-04T18:40:01Z"
    """
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _drafts_from_completion(completion: Completion) -> list[StepDraft]:
    """
    Read numbered lines from the final answer, then from content.

    :param completion: planning turn as Completion. Example: Completion(content="1. List chapter files")
    :return: drafts as list[StepDraft]. Example: [StepDraft(1, "List chapter files")]
    :raises TypeError: when completion is not a Completion. Example: None
    """
    if not isinstance(completion, Completion):
        raise TypeError(
            "completion must be Completion. Example: Completion(content='1. List chapter files')"
        )
    from_answer = _numbered_drafts(completion.final_answer)
    if from_answer:
        return from_answer
    return _numbered_drafts(completion.content)


def _numbered_drafts(text: str) -> list[StepDraft]:
    """
    Keep lines that start with an order and a sentence.

    :param text: model text as str. Example: "1. List chapter files"
    :return: drafts as list[StepDraft]. Example: [StepDraft(1, "List chapter files")]
    """
    drafts: list[StepDraft] = []
    for line in text.splitlines():
        draft = _draft_from_line(line)
        if draft is not None:
            drafts.append(draft)
    return drafts


def _draft_from_line(line: str) -> StepDraft | None:
    """
    Parse one numbered line. Other lines are not steps.

    :param line: one line as str. Example: "1. List chapter files"
    :return: draft or None as StepDraft | None. Example: StepDraft(1, "List chapter files")
    """
    match = _NUMBERED_STEP.match(line.strip())
    if match is None:
        return None
    order = int(match.group(1))
    if order < _MIN_STEP_ORDER:
        return None
    return StepDraft(order, match.group(2))
