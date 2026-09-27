"""Regenerate the E37 result section from the preserved Kaggle output."""
import json
from pathlib import Path
from results_io import upsert_section

ROOT = Path(__file__).resolve().parents[1]
path = ROOT / "kernels/runs/e37-confirmation-Arifin/result.json"
result = json.loads(path.read_text())
ci = result["holdout_auc_ci95_problem_bootstrap"]
body = ["Frozen GSM-Plus confirmation; source: `kernels/runs/e37-confirmation-Arifin/result.json`.",
        "", "| Measurement | Value |", "| --- | --- |",
        f"| Held-out configurations | {result['holdout']['n']} |",
        f"| Canonical problems | {result['holdout']['problems']} |",
        f"| Robust / spurious | {result['holdout']['robust']} / {result['holdout']['spurious']} |",
        f"| ROC-AUC | {result['holdout_auc']:.4f} |",
        f"| Problem-bootstrap 95% interval | [{ci[0]:.4f}, {ci[1]:.4f}] |",
        f"| Half-rank balanced accuracy | {result['holdout_half_rank_balanced_accuracy']:.4f} |",
        f"| All preregistered gates passed | {result['passed']} |", "",
        "| Model | ROC-AUC |", "| --- | --- |"]
body += [f"| {name} | {value:.4f} |" for name, value in result["model_aucs"].items()]
body += ["", "This opens the unchanged official AIMO transfer; it does not yet authorize a submission."]
upsert_section(ROOT / "RESULTS.md", "E37: frozen GSM-Plus confirmation", "\n".join(body))
