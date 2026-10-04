"""Three prompt layers. Worker calls PromptAssembler before every model turn."""

import logging

from local_ai.contracts import AccessGrant

logger = logging.getLogger(__name__)

_FOUNDATION_TEXT = (
    "You are the local worker for one task. "
    "Plan, then act one step at a time. "
    "After a failed command, revise the remaining plan. "
    "Emit a short reasoning note before each command."
)
_SHELL_BRIEF_TEXT = (
    "python3, curl, pandoc, weasyprint, and mmdc are on PATH. "
    "Use python3 for files, curl to fetch, pandoc for markdown, "
    "weasyprint for PDF, and mmdc for Mermaid. "
    "Stay inside the access grant. "
    "Declare output files when a step produces them."
)


class PromptAssembler:
    """Builds foundation, shell brief, then the task layer, in that order."""

    def assemble(
        self,
        goal: str,
        done_when: str,
        grant: AccessGrant,
        inputs: tuple[str, ...],
        plan_text: str,
        prior_synthesis: str,
    ) -> str:
        """
        Assemble the worker prompt for one model call.

        :param goal: task goal as str. Example: "Turn the notes into a one-page brief."
        :param done_when: completion condition as str. Example: "brief.md exists in the job folder."
        :param grant: access grant as AccessGrant. Example: AccessGrant(mode="job")
        :param inputs: input paths as tuple[str, ...]. Example: ()
        :param plan_text: readable plan as str. Example: "1. List chapter files (pending)."
        :param prior_synthesis: previous step handoff as str. Example: "Found ch-01.md through ch-12.md"
        :return: prompt as str. Example: "You are the local worker for one task.\\n\\npython3, curl, pandoc, weasyprint, and mmdc are on PATH.\\n\\nGoal: Turn the notes into a one-page brief."
        :raises TypeError: when an argument has the wrong type. Example: grant={"mode": "job"}
        """
        self._validate_task_inputs(goal, done_when, grant, inputs, plan_text, prior_synthesis)
        empty_prior = prior_synthesis == ""
        logger.info(f"PromptAssembler.assemble grant_mode={grant.mode} prior_synthesis_empty={empty_prior}")
        task = self._task(goal, done_when, grant, inputs, plan_text, prior_synthesis)
        return "\n\n".join((self._foundation(), self._shell_brief(), task))

    def _foundation(self) -> str:
        """
        Identity and recovery rule for the local worker.

        :return: foundation text as str. Example: "You are the local worker for one task."
        """
        logger.info("PromptAssembler._foundation")
        return _FOUNDATION_TEXT

    def _shell_brief(self) -> str:
        """
        Binaries the shell may use, and the grant reminder.

        :return: brief as str. Example: "python3, curl, pandoc, weasyprint, and mmdc are on PATH."
        """
        logger.info("PromptAssembler._shell_brief")
        return _SHELL_BRIEF_TEXT

    def _task(
        self,
        goal: str,
        done_when: str,
        grant: AccessGrant,
        inputs: tuple[str, ...],
        plan_text: str,
        prior_synthesis: str,
    ) -> str:
        """
        The pin-specific layer.

        :param goal: task goal as str. Example: "Turn the notes into a one-page brief."
        :param done_when: completion condition as str. Example: "brief.md exists"
        :param grant: access grant as AccessGrant. Example: AccessGrant(mode="home")
        :param inputs: input paths as tuple[str, ...]. Example: ("/Users/denispetelin/books/draft",)
        :param plan_text: readable plan as str. Example: ""
        :param prior_synthesis: previous synthesis as str. Example: ""
        :return: task block as str. Example: "Goal: Turn the notes into a one-page brief."
        :raises TypeError: when an argument has the wrong type. Example: grant={"mode": "job"}
        """
        self._validate_task_inputs(goal, done_when, grant, inputs, plan_text, prior_synthesis)
        logger.info(
            f"PromptAssembler._task grant_mode={grant.mode} "
            f"input_count={len(inputs)} prior_synthesis_empty={prior_synthesis == ''}"
        )
        return _format_task(goal, done_when, grant, inputs, plan_text, prior_synthesis)

    def _validate_task_inputs(
        self,
        goal: str,
        done_when: str,
        grant: AccessGrant,
        inputs: tuple[str, ...],
        plan_text: str,
        prior_synthesis: str,
    ) -> None:
        """
        Reject arguments that are not the pin fields the task layer prints.

        :param goal: task goal as str. Example: "Turn the notes into a one-page brief."
        :param done_when: completion condition as str. Example: "brief.md exists"
        :param grant: access grant as AccessGrant. Example: AccessGrant(mode="home")
        :param inputs: input paths as tuple[str, ...]. Example: ("/Users/denispetelin/books/draft",)
        :param plan_text: readable plan as str. Example: ""
        :param prior_synthesis: previous synthesis as str. Example: ""
        :return: None. Example: None
        :raises TypeError: when an argument has the wrong type. Example: grant={"mode": "job"}
        """
        if not isinstance(goal, str) or not isinstance(done_when, str):
            raise TypeError("goal and done_when must be str")
        if not isinstance(grant, AccessGrant):
            raise TypeError(f"grant must be AccessGrant, got {type(grant).__name__}")
        if not isinstance(inputs, tuple) or any(not isinstance(item, str) for item in inputs):
            raise TypeError("inputs must be tuple[str, ...]")
        if not isinstance(plan_text, str) or not isinstance(prior_synthesis, str):
            raise TypeError("plan_text and prior_synthesis must be str")


def _format_task(
    goal: str,
    done_when: str,
    grant: AccessGrant,
    inputs: tuple[str, ...],
    plan_text: str,
    prior_synthesis: str,
) -> str:
    """
    Render the pin fields, including the grant object's repr.

    :param goal: task goal as str. Example: "Turn the notes into a one-page brief."
    :param done_when: completion condition as str. Example: "brief.md exists"
    :param grant: access grant as AccessGrant. Example: AccessGrant(mode="home")
    :param inputs: input paths as tuple[str, ...]. Example: ("/Users/denispetelin/books/draft",)
    :param plan_text: readable plan as str. Example: ""
    :param prior_synthesis: previous synthesis as str. Example: ""
    :return: task block as str. Example: "Goal: Turn the notes into a one-page brief."
    """
    return "\n".join(
        (
            f"Goal: {goal}",
            f"Done when: {done_when}",
            f"Inputs: {inputs!r}",
            f"Grant: {grant!r}",
            f"Plan: {plan_text}",
            f"Prior synthesis: {prior_synthesis}",
        )
    )
