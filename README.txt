# End-to-End Instructions (Flat Repo, No Subfolders in ZIP)

This ZIP is flat (no subfolders). It contains:
- requirements.txt
- rf_pipeline.py
- setup_ci.sh  (helper that *creates* the required .github/workflows path for GitHub Actions)
- README.txt (this file)

The pipeline uses a tiny, GitHub-hosted ransomware dataset (subiksha03/dataset) with labeled samples (benign=0, ransomware=1).

====================
1) Create the GitHub repository
====================
A) Go to https://github.com/new
B) Name it, e.g., `sme-ransomware-rf`
C) Leave it Public or Private as you prefer. Do not add any files yet.
D) Click **Create repository**.

====================
2) Add the files from this ZIP
====================
Option 1 — Web upload:
  - Click **Add file > Upload files** in your new repo.
  - Drag-and-drop the contents of this ZIP (flat files only).
  - Commit the changes.

Option 2 — Local git:
  ```bash
  mkdir sme-ransomware-rf && cd sme-ransomware-rf
  # unzip the ZIP contents here so the files sit in the repo root
  git init
  git remote add origin https://github.com/<your-username>/sme-ransomware-rf.git
  git add .
  git commit -m "Initial commit: RF baseline"
  git push -u origin main
  ```

====================
3) Run locally (optional but recommended)
====================
```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python rf_pipeline.py
```
Outputs in the repo root:
- dataset.csv
- metrics.csv
- confusion_matrix_run1.png ... confusion_matrix_run5.png

You can re-run with different settings:
```bash
python rf_pipeline.py --runs 10 --test-size 0.30
```

====================
4) Set up GitHub Actions CI (two choices)
====================
GitHub requires workflows to live at `.github/workflows/*.yml`.
Since you asked for a **flat ZIP with no subfolders**, use ONE of these:

Choice A — Use the helper script (fastest)
```bash
# from the repo root
bash setup_ci.sh
git add .github/workflows/rf-ci.yml
git commit -m "Add CI workflow"
git push
```
This creates `.github/workflows/rf-ci.yml` for you.

Choice B — Create workflow file in the GitHub UI
- In your repo, click **Actions** → **New workflow** → **set up a workflow yourself**
- Name it `rf-ci.yml` under `.github/workflows/`
- Paste the following YAML and commit:

---------------- COPY BELOW ----------------
name: rf-baseline

on:
  push:
  pull_request:
  workflow_dispatch:

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install deps
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt

      - name: Run RF baseline
        run: |
          python rf_pipeline.py

      - name: Upload artifacts
        uses: actions/upload-artifact@v4
        with:
          name: rf-results
          path: |
            dataset.csv
            metrics.csv
            confusion_matrix_run*.png
---------------- COPY ABOVE ----------------

====================
5) Trigger the CI and view results
====================
- Push a commit (or click **Run workflow** from the Actions tab).
- Open **Actions** → select the latest run → check logs.
- Download artifacts: `rf-results` (contains dataset.csv, metrics.csv, confusion matrices).

====================
6) Interpreting Results (what to look for)
====================
Each run prints a classification report (per-class precision/recall/F1), and we also save:
- metrics.csv — a table of accuracy, precision, recall, F1 for each randomized run
- confusion_matrix_runX.png — heatmap images of TP/TN/FP/FN

On this small dataset (~147 rows), expect metrics to vary between runs because:
- The train/test split changes which samples go to test.
- RF hyperparameters vary a bit per run (n_estimators, depth, leaf size).

Good outcomes for ransomware detection:
- Higher **recall** on class 1 (ransomware) → fewer misses.
- Balanced or high **precision** → fewer benign flagged incorrectly.
- A solid **F1** score balances both.

====================
7) Next steps (comparing other models)
====================
In `rf_pipeline.py`, swap RandomForest with other classifiers using the same loop/metrics:
- XGBoost (`from xgboost import XGBClassifier`)
- SVM (`from sklearn.svm import SVC`)
- Logistic Regression / Gradient Boosting

Keep the evaluation identical so scores are comparable.
