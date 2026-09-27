# protocols/

A protocol is a **pre-registration**: the population, the procedure, the metric and the pass
threshold, written down *before* the run's output is looked at, and not edited afterwards.

Write one whenever a run's result will decide whether a lane continues. Skip it for plumbing.

Why this folder exists: the small track lost time twice to results that looked real and were
selection effects — a probe layer chosen on the data that scored it (its Finding 11), and a
threshold tuned on the 14 cases it was then evaluated on. A frozen gate is the cheapest defence,
and a gate that fails is a publishable result rather than a setback.

| File | Gate | Status |
| --- | --- | --- |
| `M3_PROTOCOL.md` | **G-M3** — do our own labels reproduce the organizers' 71? | Frozen 27 Sep 2026, not yet run |

Each protocol ends with a table of fields to record before the first sample (hardware, engine
version, checkpoint revisions, seeds). Fill it in then; leaving it blank afterwards is the same as
not having pre-registered.
