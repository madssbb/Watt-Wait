"""Evaluate and print comprehensive model metrics for all Watt & Wait models."""
from pathlib import Path
import json
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"

def main():
    print("=" * 75)
    print("WATT & WAIT: COMPLETE MODEL EVALUATION REPORT")
    print("=" * 75)
    
    # 1. PCDR Regressor
    pcdr_path = MODELS / "pcdr_regressor.pkl"
    if pcdr_path.exists():
        bundle = joblib.load(pcdr_path)
        m = bundle.get("metrics", {})
        print("\n[1] STAGE 1: PCDR REGRESSOR (Continuous Dwell Ratio Prediction)")
        print("-" * 75)
        print(f"  * Algorithm:                  RandomForestRegressor (n_estimators=350)")
        print(f"  * Chronological Split:        {m.get('train_rows', 'N/A')} Train / {m.get('test_rows', 'N/A')} Test rows")
        print(f"  * Root Mean Squared Error:    RMSE = {m.get('rmse', 0):.4f} ({m.get('rmse', 0)*100:.2f}%)")
        print(f"  * Mean Absolute Error:        MAE  = {m.get('mae', 0):.4f} ({m.get('mae', 0)*100:.2f}%)")
        print(f"  * R-Squared (Variance Exp.):  R^2  = {m.get('r2', 0):.4f}")
        
        imp_path = MODELS / "pcdr_feature_importance.csv"
        if imp_path.exists():
            imp = pd.read_csv(imp_path)
            print("\n  Top Feature Importances:")
            for idx, r in imp.head(5).iterrows():
                print(f"    - {r['feature']:<26} {r['importance']*100:>6.2f}%")
    
    # 2. Nudge & Risk Classifier
    nudge_path = MODELS / "nudge_classifier.pkl"
    if nudge_path.exists():
        bundle = joblib.load(nudge_path)
        m = bundle.get("metrics", {})
        print("\n" + "=" * 75)
        print("[2] STAGE 2: DWELL RISK CLASSIFIER (Low / Medium / High Tiers)")
        print("-" * 75)
        print(f"  * Overall Classification Accuracy:  {m.get('risk_accuracy', 0)*100:.2f}%")
        
        report = m.get("risk_classification_report", {})
        if report:
            print("\n  Per-Class Metrics (Holdout Test Set, N=240):")
            print(f"    {'Class':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<8}")
            print(f"    {'-'*12} {'-'*12} {'-'*12} {'-'*12} {'-'*8}")
            for c in ["High", "Medium", "Low"]:
                if c in report:
                    r = report[c]
                    print(f"    {c:<12} {r['precision']:<12.4f} {r['recall']:<12.4f} {r['f1-score']:<12.4f} {int(r['support']):<8}")
            print(f"    {'-'*12} {'-'*12} {'-'*12} {'-'*12} {'-'*8}")
            macro = report.get("macro avg", {})
            weighted = report.get("weighted avg", {})
            print(f"    {'Macro Avg':<12} {macro.get('precision',0):<12.4f} {macro.get('recall',0):<12.4f} {macro.get('f1-score',0):<12.4f} {int(macro.get('support',0)):<8}")
            print(f"    {'Weighted Avg':<12} {weighted.get('precision',0):<12.4f} {weighted.get('recall',0):<12.4f} {weighted.get('f1-score',0):<12.4f} {int(weighted.get('support',0)):<8}")

        print("\n  Responsiveness-Proxy Classifier (Driver Behavioral Sensitivity):")
        print(f"  * Receiver Operating Characteristic ROC-AUC:  {m.get('responsiveness_proxy_roc_auc', 0):.4f}")
        print(f"  * Baseline Positive Responsiveness Rate:     {m.get('responsiveness_proxy_positive_rate', 0)*100:.2f}%")

    # 3. Multitarget Models
    range_path = MODELS / "model_range.pkl"
    deg_path = MODELS / "model_degradation.pkl"
    if range_path.exists() and deg_path.exists():
        rm = joblib.load(range_path).get("metrics", {})
        dm = joblib.load(deg_path).get("metrics", {})
        print("\n" + "=" * 75)
        print("[3] MULTI-TARGET MODELS (Range & Battery Health)")
        print("-" * 75)
        print("  A. Driving Range Regressor (Target: range_km):")
        print(f"     * RMSE:  {rm.get('rmse', 0):.3f} km")
        print(f"     * MAE:   {rm.get('mae', 0):.3f} km")
        print(f"     * R^2:   {rm.get('r2', 0):.4f}")
        print("\n  B. Battery Degradation / State of Health Regressor (Target: soh_percentage):")
        print(f"     * RMSE:  {dm.get('rmse', 0):.4f}%")
        print(f"     * MAE:   {dm.get('mae', 0):.4f}%")
        print(f"     * R^2:   {dm.get('r2', 0):.4f}")
    
    print("=" * 75 + "\n")

if __name__ == "__main__":
    main()
