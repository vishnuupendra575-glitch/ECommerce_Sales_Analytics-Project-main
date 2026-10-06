"""
run_ml.py
=========
Entry-point script for the E-Commerce Sales Analytics ML pipeline.

Delegates entirely to the existing implementations in:
  src/data_loader.py  →  load_all_datasets()
  src/train_model.py  →  run_ml_pipeline()

What this script does
---------------------
1. Adds the project root to sys.path so that the src/ package is importable
   regardless of the working directory from which the script is invoked.
2. Calls load_all_datasets() to load and clean all five CSV files.
3. Calls run_ml_pipeline() which:
      a. Builds the 26-feature matrix (net_sales target, leakage-free)
      b. Splits data 80/20 (random_state=42)
      c. Trains Random Forest Regressor
      d. Trains XGBoost Regressor
      e. Evaluates both on the held-out test set (MAE, RMSE, R²)
      f. Runs 5-fold cross-validation on the training set
      g. Saves evaluation charts to  output/figures/
      h. Saves the champion model to  models/saved/best_model.pkl
      i. Saves model metadata to      models/saved/model_metadata.pkl
4. Prints a final summary of the champion model's metrics.

Usage
-----
    python run_ml.py

No arguments are required. All paths resolve relative to the project root.
"""

import pathlib
import sys
import time

# ── Ensure src/ is importable from any working directory ──────────────────────
_ROOT = pathlib.Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data_loader import load_all_datasets   # noqa: E402
from src.train_model import run_ml_pipeline     # noqa: E402


def main() -> None:
    """Load data then execute the full ML pipeline."""

    print()
    print("=" * 62)
    print("  E-Commerce Sales Analytics — ML Pipeline")
    print("  Target: net_sales  |  Task: Regression")
    print("  Models: Random Forest Regressor · XGBoost Regressor")
    print("=" * 62)

    # ── Step 1: Load and clean all datasets ───────────────────────────────────
    print("\nStep 1/2  Loading datasets …")
    t0 = time.time()
    datasets = load_all_datasets(verbose=True)
    print(f"  Datasets loaded in {time.time() - t0:.1f}s")

    # ── Step 2: Run the full ML pipeline ─────────────────────────────────────
    print("\nStep 2/2  Running ML pipeline …")
    t1 = time.time()
    results = run_ml_pipeline(datasets)
    elapsed = time.time() - t1

    # ── Final summary ─────────────────────────────────────────────────────────
    rf_result  = results["rf_result"]
    xgb_result = results["xgb_result"]

    # Champion is whichever has the higher R²
    champion = rf_result if rf_result["R2"] >= xgb_result["R2"] else xgb_result

    print()
    print("=" * 62)
    print("  ML PIPELINE COMPLETE")
    print("=" * 62)
    print(f"  Champion model  : {champion['model_name']}")
    print(f"  MAE             : {champion['MAE']:,.4f}")
    print(f"  RMSE            : {champion['RMSE']:,.4f}")
    print(f"  R²              : {champion['R2']:.4f}")
    print(f"  CV R² (5-fold)  : {champion['CV_R2_mean']:.4f} ± {champion['CV_R2_std']:.4f}")
    print(f"  Saved model     : {results['best_model_path']}")
    print(f"  Pipeline time   : {elapsed:.1f}s")
    print("=" * 62)
    print()


if __name__ == "__main__":
    main()
