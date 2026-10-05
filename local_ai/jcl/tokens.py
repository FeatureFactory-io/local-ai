"""Bearer tokens that cap how good a JCL priority may be."""

import logging
from collections.abc import Mapping
from dataclasses import dataclass

from local_ai.contracts import InvalidPriority, validate_priority


logger = logging.getLogger(__name__)

TOKEN_ENV_PREFIX = "LOCAL_AI_JCL_TOKEN_"
_EXAMPLE_NAME = "LOCAL_AI_JCL_TOKEN_50"
_EXAMPLE_SECRET = "lane-50-secret"


class TokenTableError(ValueError):
    """Raised when the JCL token environment cannot be loaded."""


class PriorityForbidden(Exception):
    """Raised when a caller asks for a better priority than its token allows."""


@dataclass(frozen=True)
class TokenTable:
    """
    Secrets read once at startup.

    :param grants: secret and best priority as tuple[tuple[str, int], ...]. Example: (("lane-50-secret", 50),)
    """

    grants: tuple[tuple[str, int], ...]

    def ceiling(self, token: str) -> int | None:
        """
        Best priority this secret may set.

        :param token: bearer secret as str. Example: "lane-50-secret"
        :return: priority as int | None. Example: 50
        :raises TypeError: when token is not a str. Example: None
        """
        logger.info("ceiling | entry")
        _require_token_text(token)
        found = _ceiling_or_none(self.grants, token)
        logger.info("ceiling | exit | known=%s", found is not None)
        return found

    def registration_rows(self) -> tuple[dict[str, object], ...]:
        """
        Priority and secret pairs for a catalog register body.

        :return: rows as tuple[dict[str, object], ...]. Example: ({"priority": 50, "token": "lane-50-secret"},)
        """
        logger.info("registration_rows | entry | count=%s", len(self.grants))
        rows = tuple({"priority": priority, "token": token} for token, priority in self.grants)
        logger.info("registration_rows | exit | count=%s", len(rows))
        return rows


def load_token_table(env: Mapping[str, str]) -> TokenTable:
    """
    Read LOCAL_AI_JCL_TOKEN_<priority> from env.

    :param env: environment as Mapping[str, str]. Example: {"LOCAL_AI_JCL_TOKEN_50": "lane-50-secret"}
    :return: table as TokenTable. Example: TokenTable((("lane-50-secret", 50),))
    :raises TypeError: when env is not a mapping. Example: []
    :raises TokenTableError: when no token is set, a secret is empty, a name is not a priority, or two names share a secret. Example: {}
    :raises InvalidPriority: when the priority suffix is outside 1..100. Example: "LOCAL_AI_JCL_TOKEN_0"
    """
    logger.info("load_token_table | entry")
    _require_mapping(env)
    grants = _grants_from_env(env)
    _require_grants(grants)
    _require_unique_secrets(grants)
    table = TokenTable(tuple(sorted(grants, key=_grant_order)))
    logger.info("load_token_table | exit | priorities=%s", _priority_text(table))
    return table


def priority_within_ceiling(ceiling: int, requested: int | None) -> int:
    """
    Choose the priority a token may store.

    An omitted request uses the ceiling. A present request must be the ceiling
    or worse, which is a larger integer.

    :param ceiling: best priority the token may set as int. Example: 50
    :param requested: caller priority as int | None. Example: 80
    :return: priority to store as int. Example: 80
    :raises InvalidPriority: when ceiling or requested is outside 1..100. Example: 0
    :raises PriorityForbidden: when requested is better than ceiling. Example: 10
    :raises TypeError: when requested is not an int or None. Example: "50"
    """
    logger.info("priority_within_ceiling | entry | ceiling=%s requested=%s", ceiling, requested)
    checked = _require_ceiling(ceiling)
    chosen = _chosen_priority(checked, requested)
    logger.info("priority_within_ceiling | exit | priority=%s", chosen)
    return chosen


def _require_mapping(env: object) -> None:
    """
    Reject anything that is not a mapping.

    :param env: candidate environment as object. Example: {}
    :return: None. Example: None
    :raises TypeError: when env is not a mapping. Example: []
    """
    if not isinstance(env, Mapping):
        raise TypeError("env must be a mapping. Example: {'LOCAL_AI_JCL_TOKEN_50': 'lane-50-secret'}")


def _grants_from_env(env: Mapping[str, str]) -> list[tuple[str, int]]:
    """
    Collect token variables. Other names are left alone.

    :param env: environment as Mapping[str, str]. Example: {"LOCAL_AI_JCL_TOKEN_50": "lane-50-secret"}
    :return: secret and priority pairs as list[tuple[str, int]]. Example: [("lane-50-secret", 50)]
    :raises TokenTableError: when a token name or secret is unusable. Example: "LOCAL_AI_JCL_TOKEN_high"
    :raises InvalidPriority: when a suffix is outside 1..100. Example: "LOCAL_AI_JCL_TOKEN_0"
    """
    return [_one_grant(str(name), raw) for name, raw in env.items() if _is_token_name(name)]


def _is_token_name(name: object) -> bool:
    """
    True when the name is a JCL token variable.

    :param name: environment name as object. Example: "LOCAL_AI_JCL_TOKEN_50"
    :return: whether to parse it as bool. Example: True
    """
    return isinstance(name, str) and name.startswith(TOKEN_ENV_PREFIX)


def _one_grant(name: str, raw: object) -> tuple[str, int]:
    """
    Parse one token variable into a secret and a ceiling.

    :param name: variable name as str. Example: "LOCAL_AI_JCL_TOKEN_50"
    :param raw: variable value as object. Example: "lane-50-secret"
    :return: secret and priority as tuple[str, int]. Example: ("lane-50-secret", 50)
    :raises TokenTableError: when the suffix or secret is unusable. Example: "LOCAL_AI_JCL_TOKEN_high"
    :raises InvalidPriority: when the suffix is outside 1..100. Example: "LOCAL_AI_JCL_TOKEN_0"
    """
    return _secret_text(raw, name), _priority_suffix(name)


def _priority_suffix(name: str) -> int:
    """
    Read the priority from the variable suffix.

    :param name: variable name as str. Example: "LOCAL_AI_JCL_TOKEN_50"
    :return: priority as int. Example: 50
    :raises TokenTableError: when the suffix is not digits. Example: "LOCAL_AI_JCL_TOKEN_high"
    :raises InvalidPriority: when the suffix is outside 1..100. Example: "LOCAL_AI_JCL_TOKEN_0"
    """
    suffix = name[len(TOKEN_ENV_PREFIX) :]
    if not suffix.isdigit():
        raise TokenTableError(f"{name} must end with a priority. Example: {_EXAMPLE_NAME}")
    return validate_priority(int(suffix))


def _secret_text(raw: object, name: str) -> str:
    """
    Require a non-empty string secret.

    :param raw: variable value as object. Example: "lane-50-secret"
    :param name: variable name as str. Example: "LOCAL_AI_JCL_TOKEN_50"
    :return: stripped secret as str. Example: "lane-50-secret"
    :raises TokenTableError: when raw is not a non-empty string. Example: "   "
    """
    if not isinstance(raw, str):
        raise TokenTableError(f"{name} must be a string. Example: {_EXAMPLE_SECRET}")
    secret = raw.strip()
    if not secret:
        raise TokenTableError(f"{name} must be a non-empty secret. Example: {_EXAMPLE_SECRET}")
    return secret


def _require_grants(grants: list[tuple[str, int]]) -> None:
    """
    Refuse an empty table so the process does not bind without a token.

    :param grants: parsed pairs as list[tuple[str, int]]. Example: []
    :return: None. Example: None
    :raises TokenTableError: when grants is empty. Example: []
    """
    if grants:
        return
    raise TokenTableError(f"no JCL tokens are set. Example: {_EXAMPLE_NAME}={_EXAMPLE_SECRET}")


def _require_unique_secrets(grants: list[tuple[str, int]]) -> None:
    """
    One secret maps to one ceiling.

    :param grants: parsed pairs as list[tuple[str, int]]. Example: [("lane-50-secret", 10), ("lane-50-secret", 50)]
    :return: None. Example: None
    :raises TokenTableError: when the same secret appears twice. Example: ("lane-50-secret", 10) and ("lane-50-secret", 50)
    """
    seen: set[str] = set()
    for secret, _priority in grants:
        if secret in seen:
            raise TokenTableError("two JCL tokens share one secret. Each secret must map to one priority.")
        seen.add(secret)


def _grant_order(grant: tuple[str, int]) -> tuple[int, str]:
    """
    Sort by priority, then by secret, so registration rows stay stable.

    :param grant: secret and priority as tuple[str, int]. Example: ("lane-50-secret", 50)
    :return: sort key as tuple[int, str]. Example: (50, "lane-50-secret")
    """
    secret, priority = grant
    return priority, secret


def _priority_text(table: TokenTable) -> str:
    """
    Comma-separated ceilings for the load log line.

    :param table: loaded table as TokenTable. Example: TokenTable((("lane-50-secret", 50),))
    :return: priorities as str. Example: "10,50"
    """
    return ",".join(str(priority) for _token, priority in table.grants)


def _require_token_text(token: object) -> None:
    """
    Reject a non-string bearer token before lookup.

    :param token: candidate secret as object. Example: "lane-50-secret"
    :return: None. Example: None
    :raises TypeError: when token is not a str. Example: None
    """
    if not isinstance(token, str):
        raise TypeError("token must be a string. Example: 'lane-50-secret'")


def _ceiling_or_none(grants: tuple[tuple[str, int], ...], token: str) -> int | None:
    """
    Find the ceiling for one secret.

    :param grants: stored pairs as tuple[tuple[str, int], ...]. Example: (("lane-50-secret", 50),)
    :param token: bearer secret as str. Example: "lane-50-secret"
    :return: priority as int | None. Example: 50
    """
    for secret, priority in grants:
        if secret == token:
            return priority
    return None


def _require_ceiling(ceiling: object) -> int:
    """
    Accept a ceiling inside 1..100.

    :param ceiling: candidate ceiling as object. Example: 50
    :return: ceiling as int. Example: 50
    :raises InvalidPriority: when ceiling is outside 1..100. Example: 0
    """
    if type(ceiling) is not int:
        raise InvalidPriority(f"priority must be an int, got {type(ceiling).__name__}")
    return validate_priority(ceiling)


def _chosen_priority(ceiling: int, requested: int | None) -> int:
    """
    Apply the ceiling. Omitted means the token's own lane.

    :param ceiling: best priority as int. Example: 50
    :param requested: caller priority as int | None. Example: 80
    :return: priority to store as int. Example: 80
    :raises PriorityForbidden: when requested is better than ceiling. Example: 10
    :raises TypeError: when requested is not an int or None. Example: "50"
    :raises InvalidPriority: when requested is outside 1..100. Example: 0
    """
    if requested is None:
        logger.info("priority_within_ceiling | branch=omitted | priority=%s", ceiling)
        return ceiling
    if type(requested) is not int:
        raise TypeError("requested priority must be an int or None. Example: 80")
    chosen = validate_priority(requested)
    if chosen < ceiling:
        logger.info(
            "priority_within_ceiling | branch=forbidden | ceiling=%s requested=%s",
            ceiling,
            chosen,
        )
        raise PriorityForbidden(f"priority {chosen} is better than the token ceiling {ceiling}")
    return chosen
