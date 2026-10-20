"""E60: premise-level contradictions written by one LLM, checked blind by another, through OpenRouter.

E58 and E59 (Findings 83, 84) built contradictions from the answer ("the answer is odd"), and our models read
them as a wrong hint. The organizers' `expert_no_solution` adds a condition about the problem's own objects
instead ("the system registers at least 21 departures" where at most 15 are possible). Writing those by hand
does not scale, so a strong model writes them under one fixed prompt and a model from another family checks
each, blind, before any of our models sees it:

1. **Generate.** `GENERATOR` gets the problem and its answer and returns two sentences, each stating a
   condition that contradicts a different part of the problem. It writes only the sentences; this script
   inserts each one before the problem's question (`insert`), so the original text is never retyped.
2. **Check.** `CHECKER` sees each variant and, as a control, the original, without being told anything was
   added, and ends with a verdict. A variant is kept iff the checker calls it contradictory and calls the
   original solvable.
3. **Retry once.** A problem left with fewer than two kept variants gets one more generation round with the
   same prompt; whatever is kept after that is used.

Every request and response is saved (`raw/*.jsonl`) for the report; the API key is read from `Secrets.env`
(`omarapi`) and never written anywhere. A model named `gemini-*` goes to the Gemini API instead (E62's
`--generator gemini-2.5-pro --checker gemini-2.5-flash`), its key sent as a header, never in a URL. Writes `<out>/variants.parquet` in the sampler's variants format
(category `contradict_premise`, `expected_answer` the original answer).

Run:  python datagen/perturb/premise_variants.py --pool data/e58/pool.parquet --out-dir data/e60
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

PERTURB = Path(__file__).resolve().parent
ROOT = PERTURB.parents[1]
sys.path.insert(0, str(PERTURB))

from transforms import _belongs_to_question, split_sentences  # noqa: E402

# Amendment 1: the paid pair (deepseek/deepseek-v4-pro, anthropic/claude-sonnet-5.5) had no credit to run on.
GENERATOR = "nvidia/nemotron-3-ultra-550b-a55b:free"
CHECKER = "minimax/minimax-m3.1-flash:free"
URL = "https://openrouter.ai/api/v1/chat/completions"
PACE_SECONDS = 4  # free models allow about 20 requests a minute
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
GEMINI_PACE_SECONDS = 13  # the free tier's tightest per-minute limit, gemini-2.5-pro's, is 5 requests


class DailyLimit(Exception):
    """The free models' daily request limit; rerun the script after it resets and it resumes."""

GENERATION_PROMPT = """Below is a competition math problem and its correct answer.

Write two NEW sentences. Each must state a condition about the problem's own objects (its quantities, counts, \
lengths, values or configuration) that contradicts the problem's other conditions, so that if the sentence is \
added to the problem, NO answer satisfies it. Rules:
- Each sentence must read like an ordinary given condition. It must not mention "the answer", must not say \
the problem is unsolvable, and must not ask anything.
- The contradiction must follow from the problem's stated conditions alone, and be checkable with a short \
argument.
- Changing a number so that the answer merely changes is NOT enough: the problem must become impossible.
- The two sentences must contradict different conditions of the problem.

Example of the style, for a problem about three airlines departing every 100, 120 and 150 days: "Over a full \
synchronized cycle of 600 days, the system registers a cumulative total of at least 21 individual departures." \
(At most 15 departures are possible in that cycle.)

Problem:
{problem}

Correct answer: {answer}

Return only JSON: {{"conditions": [{{"sentence": "...", "why_impossible": "..."}}, {{"sentence": "...", \
"why_impossible": "..."}}]}}"""

CHECK_PROMPT = """Here is a math problem. Decide whether it has a valid answer, or whether its conditions \
contradict each other so that no answer exists. Reason briefly, then end with exactly one line: \
"VERDICT: SOLVABLE" or "VERDICT: CONTRADICTORY".

Problem:
{problem}"""


def api_keys() -> list[str]:
    """Every OpenRouter key in Secrets.env (`omarapi` first); each account has its own free daily quota."""
    keys = []
    for line in (ROOT / "Secrets.env").read_text(encoding="utf-8-sig").splitlines():
        name, _, value = line.partition("=")
        value = value.strip().strip("\"'")
        if value.startswith("sk-or-"):
            keys.insert(0, value) if name.strip().lower() == "omarapi" else keys.append(value)
    if not keys:
        raise SystemExit("no OpenRouter key in Secrets.env")
    return keys


def gemini_key() -> str:
    """The Gemini key in Secrets.env: the first line whose name mentions gemini."""
    for line in (ROOT / "Secrets.env").read_text(encoding="utf-8-sig").splitlines():
        name, _, value = line.partition("=")
        if "gemini" in name.lower() and value.strip():
            return value.strip().strip("\"'")
    raise SystemExit("no Gemini key in Secrets.env")


def chat(keys: list[str], model: str, prompt: str, temperature: float, log) -> str:
    """One completion, logged without the key: Gemini for `gemini-*`, else OpenRouter on the first key with quota."""
    gemini = model.startswith("gemini-")
    if gemini:
        body = {"contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": temperature, "maxOutputTokens": 32000}}
    else:
        body = {"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": temperature,
                "max_tokens": 32000}
    for attempt in range(6):
        time.sleep(GEMINI_PACE_SECONDS if gemini else PACE_SECONDS)
        key = gemini_key() if gemini else keys[0]
        url, header = ((GEMINI_URL.format(model=model), {"x-goog-api-key": key}) if gemini
                       else (URL, {"Authorization": f"Bearer {key}"}))
        request = urllib.request.Request(url, data=json.dumps(body).encode(),
                                         headers={**header, "Content-Type": "application/json"})
        try:
            response = json.load(urllib.request.urlopen(request, timeout=900))
            if gemini:  # thought parts, if any, are not the answer
                text = "".join(part.get("text", "") for part in response["candidates"][0]["content"]["parts"]
                               if not part.get("thought"))
            else:
                text = response["choices"][0]["message"]["content"] or ""
            log.write(json.dumps({"model": model, "prompt": prompt, "temperature": temperature, "response": text,
                                  "usage": response.get("usageMetadata" if gemini else "usage")}) + "\n")
            log.flush()
            return text
        except urllib.error.HTTPError as exc:
            message = exc.read().decode(errors="replace").replace(key, "[REDACTED]")
            if gemini and exc.code == 429 and "perday" in message.lower():
                raise DailyLimit(message[:200]) from None
            if exc.code == 429 and "per-day" in message.lower():
                keys.pop(0)  # this account's free quota is spent for today; move to the next
                if not keys:
                    raise DailyLimit(message[:200]) from None
                print(f"  daily limit on one key; {len(keys)} left", flush=True)
                continue
            print(f"  {model}: HTTP {exc.code} {message[:160]}; retry {attempt + 1}", flush=True)
            time.sleep(30 * (attempt + 1))
        except (urllib.error.URLError, KeyError, IndexError, TimeoutError, json.JSONDecodeError) as exc:
            print(f"  {model}: {type(exc).__name__}, retry {attempt + 1}", flush=True)
            time.sleep(30 * (attempt + 1))
    raise SystemExit(f"{model} failed six times")


def parse_conditions(text: str) -> list[dict]:
    match = re.search(r"\{.*\}", text, re.S)
    try:
        conditions = json.loads(match.group(0))["conditions"] if match else []
    except (json.JSONDecodeError, KeyError, TypeError):
        return []
    return [c for c in conditions if isinstance(c, dict) and str(c.get("sentence", "")).strip()]


def insert(problem: str, sentence: str) -> str:
    """The problem with `sentence` placed before its question, as the organizers' example is, where the text
    allows: some premise must precede the question, and the spot must not be inside a quotation or a problem
    with display math or several paragraphs. Otherwise it is appended, after every object it can refer to."""
    sentence, problem = sentence.strip(), problem.strip()
    appended = f"{problem} {sentence}"
    if "\n" in problem or "\\[" in problem or "$$" in problem or "\\begin" in problem:
        return appended
    sentences = split_sentences(problem)
    start = len(sentences) - 1
    while start > 0 and _belongs_to_question(sentences[start - 1]):
        start -= 1
    before = " ".join(sentences[:start])
    if start == 0 or before.count("``") > before.count("''") or before.count('"') % 2:
        return appended
    return " ".join([before, sentence, *sentences[start:]])


def verdict(text: str) -> str | None:
    found = re.findall(r"VERDICT:\s*(SOLVABLE|CONTRADICTORY)", text.upper())
    return found[-1] if found else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pool", type=Path, default=ROOT / "data" / "e58" / "pool.parquet")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "data" / "e60")
    parser.add_argument("--per-problem", type=int, default=2)
    parser.add_argument("--generator", default=GENERATOR)
    parser.add_argument("--checker", default=CHECKER)
    args = parser.parse_args()

    keys = api_keys()
    pool = pd.read_parquet(args.pool)
    (args.out_dir / "raw").mkdir(parents=True, exist_ok=True)
    # One line per finished problem, so a run stopped by the daily limit resumes where it stopped.
    progress_path = args.out_dir / "raw" / "progress.jsonl"
    progress = [json.loads(line) for line in progress_path.read_text(encoding="utf-8").splitlines()
                if line.strip()] if progress_path.is_file() else []
    done = {p["pool_id"] for p in progress}
    try:
        with open(args.out_dir / "raw" / "generation.jsonl", "a", encoding="utf-8") as gen_log, \
             open(args.out_dir / "raw" / "check.jsonl", "a", encoding="utf-8") as check_log, \
             open(progress_path, "a", encoding="utf-8") as progress_log:
            for row in pool.itertuples(index=False):
                if row.pool_id in done:
                    continue
                original = verdict(chat(keys, args.checker, CHECK_PROMPT.format(problem=row.problem), 0.0, check_log))
                mine, audit = [], []
                for round_ in (1, 2):
                    text = chat(keys, args.generator, GENERATION_PROMPT.format(problem=row.problem, answer=int(row.answer)), 1.0, gen_log)
                    for condition in parse_conditions(text):
                        variant = insert(row.problem, condition["sentence"])
                        said = verdict(chat(keys, args.checker, CHECK_PROMPT.format(problem=variant), 0.0, check_log))
                        ok = said == "CONTRADICTORY" and original == "SOLVABLE"
                        audit.append({"pool_id": row.pool_id, "round": round_, "sentence": condition["sentence"],
                                      "why_impossible": condition.get("why_impossible", ""), "checker_variant": said,
                                      "checker_original": original, "kept": ok and len(mine) < args.per_problem})
                        if audit[-1]["kept"]:
                            mine.append({"variant_id": f"{args.out_dir.name}:contradict_premise:{row.pool_id}:{len(mine)}",
                                         "pool_id": row.pool_id, "category": "contradict_premise", "family": "no_solution",
                                         "tier": row.tier, "variant_problem": variant, "compliance": "distinct",
                                         "expected_answer": int(row.answer), "sentence": condition["sentence"]})
                    if len(mine) >= args.per_problem:
                        break
                progress_log.write(json.dumps({"pool_id": row.pool_id, "kept": mine, "audit": audit}) + "\n")
                progress_log.flush()
                progress.append({"pool_id": row.pool_id, "kept": mine, "audit": audit})
                print(f"{row.pool_id}: original {original}, kept {len(mine)} of {len(audit)} checked", flush=True)
    except DailyLimit as exc:
        print(f"daily request limit reached ({exc}); rerun after it resets to resume", flush=True)
    kept = [v for p in progress for v in p["kept"]]
    pd.DataFrame(kept).to_parquet(args.out_dir / "variants.parquet", index=False)
    pd.DataFrame([a for p in progress for a in p["audit"]]).to_parquet(args.out_dir / "audit.parquet", index=False)
    print(f"{len(progress)} of {len(pool)} problems done; {len(kept)} variants for "
          f"{len({k['pool_id'] for k in kept})} problems -> {args.out_dir}")


if __name__ == "__main__":
    main()
