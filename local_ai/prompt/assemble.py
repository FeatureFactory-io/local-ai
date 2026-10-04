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
        :return: prompt as str. Example: "Foundation\\nShell\\nTask"
        """
        # Call _foundation, then _shell_brief, then _task.
        # Join the three blocks with a blank line.
        # Log the grant mode and whether prior_synthesis is empty at INFO. Do not log the goal body.
        raise NotImplementedError

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
        # Name python3, curl, pandoc, weasyprint, and mmdc.
        # Tell the model to stay inside the access grant and to declare output files.
        raise NotImplementedError

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
        """
        # Include goal, done_when, inputs, the grant object, plan_text, and prior_synthesis.
        raise NotImplementedError
