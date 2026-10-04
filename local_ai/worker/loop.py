"""Agent loop for the one running task. The slot calls run_task. This module does not import dispatcher."""

import logging
import re
import sqlite3

from local_ai.blackboard.pins import Blackboard
from local_ai.contracts import MAX_IDENTICAL_FAILURES, MAX_ROUNDS_PER_STEP, Completion, StepDraft
from local_ai.llm.port import LlmPort, split_note
from local_ai.plan.steps import PlanStore
from local_ai.prompt.assemble import PromptAssembler
from local_ai.tools.shell import ShellTool


logger = logging.getLogger(__name__)

_NUMBERED_STEP = re.compile(r"^(\d+)\.\s+(\S.*?)\s*$")
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

    def run_task(self, task_id: str) -> str:
        """
        Run one pin until it completes or fails.

        :param task_id: pin id as str. Example: "7c1e0a2e-3b4d-4f5a-9c8d-1a2b3c4d5e6f"
        :return: final status as str. Example: "completed"
        """
        # Load the pin from Blackboard. Log entry with task_id.
        # Set phase to planning. Assemble the prompt. Ask LlmPort for the plan.
        # parse_plan, then PlanStore.insert_steps, then Blackboard.set_phase with the readable plan.
        # For each pending step, run_step. A failed step calls adapt_after_failure
        # unless note_identical_failure says the pin must stop.
        # When no pending steps remain, return "completed".
        # The caller (Slot) passes that word to DispatcherService.finish.
        raise NotImplementedError

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
        """
        # Call LlmPort.complete.
        # split_note. If the note is non-empty, Blackboard.append_reasoning.
        # If the turn has a shell tool call, ShellTool.run and keep the envelope on the turn.
        # Tool errors stay in the envelope. Do not raise them.
        # Log worker.step acting with task_id, step, and tool=shell at INFO when the shell runs.
        raise NotImplementedError

    def run_step(self, task_id: str, order: int, model_name: str) -> str:
        """
        Drive one step for at most MAX_ROUNDS_PER_STEP rounds.

        :param task_id: pin id as str. Example: "7c1e0a2e-3b4d-4f5a-9c8d-1a2b3c4d5e6f"
        :param order: step order as int. Example: 2
        :param model_name: logical size as str. Example: "qwen-14b"
        :return: step outcome as str. Example: "completed"
        """
        # PlanStore.mark_running. Blackboard phase acting.
        # Loop at most MAX_ROUNDS_PER_STEP (12) times:
        #   PromptAssembler.assemble including the previous step's llm_synthesis.
        #   run_round.
        #   A final_answer calls PlanStore.complete_step and returns "completed".
        #   A failed shell returns "failed" after PlanStore.fail_step.
        #   No tool and no final answer calls force_final once.
        # If the loop ends with no final answer, fail the step and return "failed".
        raise NotImplementedError

    def force_final(self, task_id: str, order: int, model_name: str) -> str:
        """
        Ask once for a final answer. An empty answer fails the pin.

        :param task_id: pin id as str. Example: "7c1e0a2e-3b4d-4f5a-9c8d-1a2b3c4d5e6f"
        :param order: step order as int. Example: 2
        :param model_name: logical size as str. Example: "qwen-8b"
        :return: "completed" or "failed" as str. Example: "failed"
        """
        # One more LlmPort.complete that asks for a final answer.
        # A non-empty final_answer completes the step.
        # An empty answer returns "failed" so run_task can stop the pin.
        # Log the branch at INFO.
        raise NotImplementedError

    def adapt_after_failure(self, task_id: str, order: int, model_name: str, error: str) -> None:
        """
        Leave the failed step failed and insert one pending replacement.

        :param task_id: pin id as str. Example: "7c1e0a2e-3b4d-4f5a-9c8d-1a2b3c4d5e6f"
        :param order: failed step order as int. Example: 2
        :param model_name: logical size as str. Example: "qwen-24b"
        :param error: tool error as str. Example: "weasyprint: CSS path missing"
        :return: None. Example: None
        """
        # Set phase recovering.
        # Ask LlmPort for a replacement action. PlanStore.insert_after.
        # The failed row stays failed.
        # Refresh current_plan on the blackboard.
        # Log task_id and the new order at INFO.
        raise NotImplementedError

    def note_identical_failure(self, task_id: str, order: int, command: str) -> bool:
        """
        Count identical command failures. The third one stops the pin.

        :param task_id: pin id as str. Example: "7c1e0a2e-3b4d-4f5a-9c8d-1a2b3c4d5e6f"
        :param order: step order as int. Example: 2
        :param command: shell command as str. Example: "weasyprint book.html book.pdf"
        :return: True when the pin must stop as bool. Example: False
        """
        # Compare command to the previous failure on this step.
        # At MAX_IDENTICAL_FAILURES (3), return True. Otherwise return False.
        # A different command resets the count.
        # Log the count at INFO. Do not log the command text beyond a short hash or length.
        raise NotImplementedError


_ROUNDS = MAX_ROUNDS_PER_STEP
_IDENTICAL = MAX_IDENTICAL_FAILURES


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
