"""Job intake: reads a training script, extracts features via an LLM,
predicts k with the saved model, and submits to the scheduler queue."""

import json
import time
import joblib
import pandas as pd
from pathlib import Path
from anthropic import Anthropic
from dotenv import load_dotenv
load_dotenv()
from scheduler.job import Job
from scheduler.logger import logger
from scheduler.queue import Queue

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "model"
FAMILIES_JSON = ROOT / "model_families.json"
SCALING_CSV = ROOT / "train_data" / "scaling_dataset.csv"
FAMILIES = list(json.loads(FAMILIES_JSON.read_text()).keys())

MAX_SOURCE_LINES = 60  # only need constants + class signatures, not full impl

LLM_PROMPT = """\
You are analyzing a PyTorch DDP training script. Extract these three fields:

1. **family**: one of {families}
2. **batch_size**: the training batch size (integer)
3. **param_count**: approximate total trainable parameter count as an integer. Look for comments, docstrings, or variable names that state the param count. Estimate from the model architecture. Do NOT try to compute it precisely.

Respond with ONLY a JSON object. No reasoning, no markdown, no explanation.
{{"family": "...", "batch_size": ..., "param_count": ...}}

Source code (truncated):
```python
{source}
```"""


class JobProfiler:
    def __init__(self, queue):
        self.queue = queue
        self.client = Anthropic()
        self.model = joblib.load(MODEL_DIR / "best_model.joblib")
        meta = json.loads((MODEL_DIR / "feature_columns.json").read_text())
        self.feature_cols = meta["feature_columns"]
        self.family_defaults = _compute_family_defaults(SCALING_CSV)

    def submit(self, script_path):
        """Read script, extract features via LLM, predict k, add to queue."""
        script_path = Path(script_path)
        logger.info(f"intake: reading {script_path.name}")
        source = script_path.read_text()
        logger.debug(f"intake: {script_path.name} is {len(source)} chars, {len(source.splitlines())} lines")

        features = self._extract_features(source)
        k = self._predict_k(features)

        job = Job(k, features, script_path)
        logger.info(f"profiled: {job}")
        self.queue.add_job(job)
        return job

    def _extract_features(self, source):
        """Call the LLM to pull family, batch_size, param_count from source."""
        prompt = LLM_PROMPT.format(
            families=", ".join(FAMILIES),
            source=source,
        )
        logger.debug("LLM: sending feature extraction request...")
        t0 = time.time()
        response = self.client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text.strip()
        logger.info(f"LLM: response in {time.time() - t0:.1f}s — {text}")
        try:
            features = json.loads(text)
        except json.JSONDecodeError as e:
            logger.error(f"LLM: failed to parse response as JSON: {e}")
            logger.error(f"LLM: raw response was: {text}")
            raise
        # Validate and fall back to family averages for missing/bad fields
        family = features.get("family", "Other")
        if family not in FAMILIES:
            family = "Other"
        defaults = self.family_defaults.get(family, self.family_defaults["_global"])

        for field in ("batch_size", "param_count"):
            val = features.get(field)
            if val is None or not isinstance(val, (int, float)) or val <= 0:
                logger.error(f"LLM: missing or invalid '{field}' (got {val!r}), "
                             f"using {family} average: {defaults[field]}")
                features[field] = defaults[field]

        logger.info(f"LLM: extracted family={features.get('family')}, "
                     f"batch_size={features.get('batch_size')}, "
                     f"param_count={features.get('param_count')}")
        return features

    def _predict_k(self, features):
        """Build the feature vector and run the saved model."""
        family = features.get("family", "Other")

        row = {col: 0 for col in self.feature_cols}
        row["batch_size"] = int(features["batch_size"])
        row["param_count"] = int(features["param_count"])
        row[f"family_{family}"] = 1

        X = pd.DataFrame([row])
        k = float(self.model.predict(X)[0])
        logger.info(f"predict_k: family={family}, batch_size={row['batch_size']}, "
                     f"param_count={row['param_count']:,} -> k={k:.4f}")
        return k


def _compute_family_defaults(csv_path):
    """Compute average batch_size and param_count per family from training data."""
    df = pd.read_csv(csv_path)
    family_cols = [c for c in df.columns if c.startswith("family_")]
    defaults = {}
    for col in family_cols:
        family = col.replace("family_", "")
        subset = df[df[col] == 1]
        if len(subset) > 0:
            defaults[family] = {
                "batch_size": int(subset["batch_size"].mean()),
                "param_count": int(subset["param_count"].mean()),
            }
    # Global fallback across all families
    defaults["_global"] = {
        "batch_size": int(df["batch_size"].mean()),
        "param_count": int(df["param_count"].mean()),
    }
    return defaults


def main():
    queue = Queue()
    intake = JobProfiler(queue)
    eval_data = ROOT / "eval_data" / "jobs"
    for py_file in eval_data.glob("*.py"):
        intake.submit(py_file)

if __name__ == "__main__":
    main()