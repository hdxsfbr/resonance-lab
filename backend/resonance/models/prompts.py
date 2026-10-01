"""Prompt templates live in `<project>/prompts/*.txt`; none are inlined in code.

Placeholders are `{identifier}` (lowercase letters, digits, underscore). Any
other brace text — e.g. a literal JSON example `{"pattern_id": 2}` — is left
untouched, so templates need no brace escaping. Rendering fails loudly if a
template uses a placeholder that was not supplied.
"""

from __future__ import annotations

import re
from pathlib import Path

_PLACEHOLDER_RE = re.compile(r"\{([a-z_][a-z0-9_]*)\}")

REQUIRED_TEMPLATES: tuple[str, ...] = (
    "choose_pattern.system",
    "choose_pattern.user",
    "choose_motif.system",
    "choose_motif.user",
    "propose_phrase.system",
    "propose_phrase.user",
    "narrative.system",
    "narrative.user",
)


class PromptError(KeyError):
    pass


class PromptLibrary:
    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        self._cache: dict[str, str] = {}

    def get(self, name: str) -> str:
        if name not in self._cache:
            path = self.directory / f"{name}.txt"
            if not path.is_file():
                raise PromptError(f"prompt template not found: {path}")
            self._cache[name] = path.read_text(encoding="utf-8")
        return self._cache[name]

    def placeholders(self, name: str) -> set[str]:
        return set(_PLACEHOLDER_RE.findall(self.get(name)))

    def render(self, name: str, **values: object) -> str:
        template = self.get(name)
        missing = self.placeholders(name) - set(values)
        if missing:
            raise PromptError(f"template {name!r} needs values for: {sorted(missing)}")
        return _PLACEHOLDER_RE.sub(lambda m: str(values[m.group(1)]), template).strip()

    def missing_templates(self) -> list[str]:
        return [n for n in REQUIRED_TEMPLATES if not (self.directory / f"{n}.txt").is_file()]
