import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.metrics import RocCurveDisplay
import time
import mlflow
import matplotlib.pyplot as plt
import io
import base64

def evaluate_model(model_uri):
    """Run standard evaluation suite on hidden test set."""
    # Load hidden test data
    df = pd.read_csv('data/hidden_test.csv')
    X = df.drop('target', axis=1)
    y = df['target']

    # Load model from MLflow
    model = mlflow.sklearn.load_model(model_uri)

    # --- Accuracy ---
    start = time.time()
    preds = model.predict(X)
    latency = (time.time() - start) / len(X) * 1000  # ms per sample
    acc = accuracy_score(y, preds)

    # --- Fairness (simple group parity) ---
    # Simulate a protected attribute (split feature_0 at median)
    protected = X['feature_0'] > X['feature_0'].median()
    group_a = (protected == True)
    group_b = (protected == False)
    acc_a = accuracy_score(y[group_a], preds[group_a])
    acc_b = accuracy_score(y[group_b], preds[group_b])
    fairness_delta = abs(acc_a - acc_b)

    # --- Slice-based metric (example: high vs low feature_1) ---
    high_feat = X['feature_1'] > X['feature_1'].median()
    acc_high = accuracy_score(y[high_feat], preds[high_feat])
    acc_low = accuracy_score(y[~high_feat], preds[~high_feat])

    # --- ROC AUC ---
    proba = model.predict_proba(X)[:, 1]
    roc = roc_auc_score(y, proba)

    # --- Confusion Matrix ---
    cm = confusion_matrix(y, preds)

    # --- Plot ROC curve and confusion matrix ---
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(cm, cmap='Blues')
    axes[0].set_title('Confusion Matrix')
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            axes[0].text(j, i, cm[i, j], ha='center', va='center')
    RocCurveDisplay.from_predictions(y, proba, ax=axes[1])
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plot_base64 = base64.b64encode(buf.read()).decode()
    plt.close()

    results = {
        "accuracy": acc,
        "roc_auc": roc,
        "latency_ms_per_sample": latency,
        "fairness_delta": fairness_delta,
        "slice_acc_high": acc_high,
        "slice_acc_low": acc_low,
        "plot_base64": plot_base64,
        "go_no_go": "GO" if acc > 0.8 and fairness_delta < 0.1 else "NO-GO"
    }
    return results