"""Job intake: reads a training script, extracts features via an LLM,
predicts k with the saved model, and submits to the scheduler queue."""

import json
import joblib
import pandas as pd
from pathlib import Path
from anthropic import Anthropic
from dotenv import load_dotenv
load_dotenv()
from scheduler.job import Job
from scheduler.queue import Queue

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "model"
FAMILIES_JSON = ROOT / "model_families.json"
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


class Intake:
    def __init__(self, queue):
        self.queue = queue
        self.client = Anthropic()
        self.model = joblib.load(MODEL_DIR / "best_model.joblib")
        meta = json.loads((MODEL_DIR / "feature_columns.json").read_text())
        self.feature_cols = meta["feature_columns"]

    def submit(self, script_path):
        """Read script, extract features via LLM, predict k, add to queue."""
        script_path = Path(script_path)
        source = script_path.read_text()

        features = self._extract_features(source)
        k = self._predict_k(features)

        job = Job(k, features, script_path)
        print(job)
        self.queue.add_job(job)
        return job

    def _extract_features(self, source):
        """Call the LLM to pull family, batch_size, param_count from source."""
        prompt = LLM_PROMPT.format(
            families=", ".join(FAMILIES),
            source=source,
        )
        response = self.client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text.strip()
        print( " and resp is " + text + "END")
        return json.loads(text)

    def _predict_k(self, features):
        """Build the feature vector and run the saved model."""
        family = features.get("family", "Other")
        if family not in FAMILIES:
            family = "Other"

        row = {col: 0 for col in self.feature_cols}
        row["batch_size"] = int(features["batch_size"])
        row["param_count"] = int(features["param_count"])
        row[f"family_{family}"] = 1

        X = pd.DataFrame([row])
        k = float(self.model.predict(X)[0])
        return k

def main():
    queue = Queue()
    intake = Intake(queue)
    eval_data = ROOT / "eval_data" / "jobs"
    for py_file in eval_data.glob("*.py"):
        intake.submit(py_file)

if __name__ == "__main__":
    main()