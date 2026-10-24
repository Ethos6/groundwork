"""Terms check for any model whose outputs become training data.

Fine-tuning a model on another model's outputs is only allowed when that model's terms
permit it. Several major API providers explicitly forbid using their services or outputs
to develop competing models, so Groundwork refuses to use them as data sources. Any other
third-party model must be declared with a link to its terms and an explicit confirmation
that the terms allow it. Every data source is recorded in ``<work_dir>/provenance.jsonl``.

This applies only to data generation (traces, probes, restyling) and LLM judges. Using any
model through the runtime layer or evaluating it is not restricted here.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from .config import EndpointConfig
from .io import append_jsonl


class ComplianceError(RuntimeError):
    pass


@dataclass(frozen=True)
class Restriction:
    provider: str
    terms_url: str
    clause: str


_ANTHROPIC = Restriction(
    "Anthropic",
    "https://www.anthropic.com/legal/commercial-terms",
    "Commercial Terms D.4: customers may not access the Services to build a competing "
    "product or service, including to train competing AI models.",
)
_OPENAI = Restriction(
    "OpenAI",
    "https://openai.com/policies/terms-of-use/",
    "Terms of Use: you may not use Output to develop models that compete with OpenAI.",
)
_GOOGLE = Restriction(
    "Google (Gemini API)",
    "https://ai.google.dev/terms",
    "Gemini API Additional Terms: you may not use the Services to develop models that "
    "compete with the Services.",
)

_BLOCKED_HOSTS: dict[str, Restriction] = {
    "api.anthropic.com": _ANTHROPIC,
    "api.openai.com": _OPENAI,
    "generativelanguage.googleapis.com": _GOOGLE,
}

# Model names served through gateways or cloud marketplaces. Open-weight releases from the
# same companies (e.g. gpt-oss, Gemma) are not matched.
_BLOCKED_MODEL_PATTERNS: list[tuple[re.Pattern, Restriction]] = [
    (re.compile(r"claude|anthropic", re.I), _ANTHROPIC),
    (re.compile(r"gemini", re.I), _GOOGLE),
    (re.compile(r"(?:^|[/:.\s])(?:gpt-(?!oss)|chatgpt|o\d(?:-|$))", re.I), _OPENAI),
]


def find_restriction(endpoint: EndpointConfig) -> Restriction | None:
    host = (urlparse(endpoint.base_url).hostname or "").lower() if endpoint.kind == "openai" else ""
    for blocked, restriction in _BLOCKED_HOSTS.items():
        if host == blocked or host.endswith("." + blocked):
            return restriction
    name = endpoint.model if endpoint.kind == "openai" else (endpoint.path or endpoint.model)
    for pattern, restriction in _BLOCKED_MODEL_PATTERNS:
        if name and pattern.search(name):
            return restriction
    return None


def check_data_source(endpoint: EndpointConfig, role: str) -> None:
    """Raise ComplianceError if ``endpoint`` may not be used to create training data."""
    restriction = find_restriction(endpoint)
    label = endpoint.model or endpoint.path or endpoint.base_url
    if restriction is not None:
        raise ComplianceError(
            f"The {role} endpoint '{label}' is provided by {restriction.provider}, whose terms do not "
            f"allow using it to train other models:\n  {restriction.clause}\n  {restriction.terms_url}\n"
            "Use your own model (self-improvement), an open-weight model whose license allows "
            "distillation, or an API whose terms explicitly permit it."
        )
    if endpoint.ownership not in ("self", "third_party"):
        raise ComplianceError(f"{role}.ownership must be 'self' or 'third_party', got {endpoint.ownership!r}")
    if endpoint.ownership == "third_party":
        missing = []
        if not endpoint.provider:
            missing.append("provider")
        if not endpoint.terms_url:
            missing.append("terms_url")
        if not endpoint.distillation_permitted:
            missing.append("distillation_permitted: true")
        if missing:
            raise ComplianceError(
                f"The {role} endpoint '{label}' is a third-party model. Before its outputs can be "
                f"used as training data, set {', '.join(missing)} in the config after confirming "
                "that its license/terms allow training other models on its outputs."
            )


def record_provenance(work_dir: str | Path, endpoint: EndpointConfig, role: str, purpose: str) -> None:
    host = urlparse(endpoint.base_url).hostname if endpoint.kind == "openai" else "local"
    append_jsonl(
        Path(work_dir) / "provenance.jsonl",
        {
            "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "role": role,
            "purpose": purpose,
            "kind": endpoint.kind,
            "model": endpoint.model or endpoint.path,
            "host": host,
            "ownership": endpoint.ownership,
            "provider": endpoint.provider,
            "terms_url": endpoint.terms_url,
        },
    )
