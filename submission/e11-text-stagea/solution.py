"""Codabench entry point for E11: text + prompt-hidden-state fusion.

Works under both harness versions. The current ingestion program calls
``are_robust(model_id, problems)`` positionally; the pending ``reasoning_effort`` version
inspects the signature and calls ``are_robust(model_id, reasoning_effort=..., problems=...)``.
"""

from e11_inference import predict_robustness


def are_robust(model_id: str, problems: list[str], reasoning_effort: str = "default") -> list[bool]:
    """One native bool per problem, in input order; True means robust.

    ``reasoning_effort`` is accepted for forward compatibility and not used: all training
    labels come from a single configuration.
    """
    return predict_robustness(model_id, problems)
