"""Codabench entry point for E11: text + prompt-hidden-state fusion.

Works under both harness versions. The current ingestion program calls
``are_robust(model_id, problems)`` positionally; the pending ``reasoning_effort`` version
inspects the signature and calls ``are_robust(model_id, reasoning_effort=..., problems=...)``.
"""

from e11_inference import predict_robustness


def are_robust(model_id: str, problems: list[str] | None = None, reasoning_effort: str = "default") -> list[bool]:
    """One native bool per problem, in input order; True means robust.

    ``reasoning_effort`` is accepted for forward compatibility and not used: all training
    labels come from a single configuration.
    """
    if problems is None:
        raise TypeError("are_robust() missing required argument: 'problems'")
    return predict_robustness(model_id, problems)
