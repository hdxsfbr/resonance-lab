"""Optional model-assisted layer for Resonance Lab.

Nothing in the simulation depends on this package: with `model.provider: none`
(the default) the local policy runs alone. When a provider is configured, a
`ModelPolicy` may make receiver/sender decisions from a SYMBOLIC FEATURE
SUMMARY (never audio), falling back to the local policy on any failure.
Model output is recorded verbatim (redacted) in `model_call` events; generated
narrative is labelled as such and is never fed back into state or learning.
"""

from resonance.models.base import ModelProvider, parse_json_object
from resonance.models.redact import redact, redact_obj

__all__ = ["ModelProvider", "parse_json_object", "redact", "redact_obj"]
