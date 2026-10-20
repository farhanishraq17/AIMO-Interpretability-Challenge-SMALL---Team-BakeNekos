"""E58: answer-property variants of the official problems, in the sampler's pool format.

Finding 76: on olympiad problems the family that decides a label is `expert_no_solution`, a premise
that contradicts the rest, and its sibling `expert_perturbations` adds a consistent one. Writing either
by hand does not scale to the hundreds of problems a probe needs, so E58 builds both from the reference
answer N alone: a sentence stating a property of the answer that is false for N (no answer satisfies
the problem any more) or true for N (nothing changes). Any problem with a known integer answer gets the
same four variants, so the labeller extends to E55's 548 problems unchanged.

| category | family | added property |
| --- | --- | --- |
| contradict_parity | no_solution | the parity N does not have |
| contradict_bound | no_solution | greater than 2N + 10 |
| consistent_parity | consistent | the parity N has |
| consistent_bound | consistent | less than 2N + 10 |

A variant's `expected_answer` is N, so the sampler's `n_correct` counts samples that answered as if the
contradiction were not there; `train/eval_e58.py` grades contradiction samples with E52's grader instead.

Population: the 27 labelled small-track rows of `train-main-v2` whose problem has a reference answer
(E24a, `answer_key.py`): 14 problems. Writes data/e58/{pool,variants,rows}.parquet; `rows` lists the
(model, problem) pairs to sample and carries no label.

Run:  python datagen/perturb/e58_variants.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CHECK = ROOT / "data" / "official_check"
OUT = ROOT / "data" / "e58"
MODEL_SHORT = {"Qwen/Qwen3.5-4B": "qwen35", "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B": "deepseek",
               "Skywork/Skywork-OR1-Math-7B": "skywork", "allenai/Olmo-3-7B-Think": "olmo-think"}
SENTENCE = "\n\nIt is also known that the answer is {}."


def properties(answer: int) -> dict[str, tuple[str, str]]:
    """category -> (family, property phrase) for a non-negative integer answer."""
    parity, other = ("an even number", "an odd number") if answer % 2 == 0 else ("an odd number", "an even number")
    bound = 2 * answer + 10
    return {"contradict_parity": ("no_solution", other), "contradict_bound": ("no_solution", f"greater than {bound}"),
            "consistent_parity": ("consistent", parity), "consistent_bound": ("consistent", f"less than {bound}")}


def satisfied(phrase: str, answer: int) -> bool:
    """Whether `answer` has the property, read back from the phrase itself rather than from `properties`."""
    if phrase in ("an even number", "an odd number"):
        return answer % 2 == (phrase == "an odd number")
    relation, _, value = phrase.rpartition(" ")
    return {"greater than": answer > int(value), "less than": answer < int(value)}[relation]


def main() -> None:
    pool = pd.read_parquet(CHECK / "pool_keyed.parquet")
    assert (pool["answer"] >= 0).all() and pool["problem_key"].is_unique
    variants = []
    for row in pool.itertuples(index=False):
        for category, (family, prop) in properties(int(row.answer)).items():
            # A contradiction must be false for the reference answer and a consistent premise true.
            assert satisfied(prop, int(row.answer)) == (family == "consistent"), (row.pool_id, category)
            variants.append({"variant_id": f"e58:{category}:{row.pool_id}", "pool_id": row.pool_id,
                             "category": category, "family": family, "tier": row.tier,
                             "variant_problem": row.problem + SENTENCE.format(prop),
                             "compliance": "distinct", "expected_answer": int(row.answer)})
    rows = pd.concat([pd.read_parquet(CHECK / f"pool_{short}.parquet")[["pool_id"]].assign(model_id=model_id)
                      for model_id, short in MODEL_SHORT.items()])
    rows = rows.merge(pool[["pool_id", "problem_key"]], on="pool_id", validate="many_to_one")
    OUT.mkdir(parents=True, exist_ok=True)
    pool.to_parquet(OUT / "pool.parquet", index=False)
    pd.DataFrame(variants).to_parquet(OUT / "variants.parquet", index=False)
    rows[["model_id", "pool_id", "problem_key"]].to_parquet(OUT / "rows.parquet", index=False)
    print(f"{len(pool)} problems, {len(variants)} variants, {len(rows)} (model, problem) rows "
          f"{rows.model_id.map(MODEL_SHORT).value_counts().to_dict()} -> {OUT}")


if __name__ == "__main__":
    main()
