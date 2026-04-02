
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


!pip install -U scikit-learn imbalanced-learn --quiet
!pip install scikit-learn==1.4.2 imbalanced-learn==0.12.2 --quiet
import pandas as pd
import numpy as np
import time
import os
import pickle
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
[3]
# 📌 CONFIG
# =========================
DATA_PATH = "/kaggle/input/ga-full/selected_features_dataset_GA_train_full.csv"  # <-- CHANGE THIS
TARGET_COL = "Label"
N_SPLITS = 3
CHECKPOINT_DIR = "/kaggle/working/checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
[4]
# ⏳ Timer Helper
# =========================
def print_time(msg, start_time):
    elapsed = time.time() - start_time
    print(f"{msg} | Time taken: {elapsed:.2f} seconds")
# =========================
# 📌 Load dataset
# =========================
start_all = time.time()
print("📥 Loading dataset...")
df = pd.read_csv(DATA_PATH)
X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]
print(f"✅ Dataset loaded. Shape: {X.shape}")
[6]
# 📌 Models Setup
# =========================
base_models = [
    ('xgb', XGBClassifier(
        use_label_encoder=False, eval_metric='logloss', random_state=42,
        n_estimators=50, learning_rate=0.1, max_depth=5, n_jobs=1
    )),
    ('et', ExtraTreesClassifier(random_state=42, n_jobs=1)),
    ('knn', KNeighborsClassifier(n_jobs=1)),
    ('lr', LogisticRegression(max_iter=1000, random_state=42, n_jobs=1)),
    ('nb', GaussianNB())
]

meta_model = DecisionTreeClassifier(random_state=42)

models = {
    "DecisionTree": DecisionTreeClassifier(random_state=42),
    "XGBoost": XGBClassifier(
        use_label_encoder=False, eval_metric='logloss', random_state=42,
        n_estimators=50, learning_rate=0.1, max_depth=5, n_jobs=1
    ),
    "Proposed Model (stacking)": StackingClassifier(
        estimators=base_models,
        final_estimator=meta_model,
        cv=3,
        stack_method='predict',
        n_jobs=1
    )
}
[7]
# =========================
# 📌 Checkpoint loading
# =========================
metrics_checkpoint_path = os.path.join(CHECKPOINT_DIR, "metrics_checkpoint.pkl")
predictions_checkpoint_path = os.path.join(CHECKPOINT_DIR, "predictions_checkpoint.pkl")

if os.path.exists(metrics_checkpoint_path):
    with open(metrics_checkpoint_path, "rb") as f:
        all_metrics = pickle.load(f)
    print("♻️ Loaded existing metrics checkpoint.")
else:
    all_metrics = {model_name: [] for model_name in models.keys()}

if os.path.exists(predictions_checkpoint_path):
    with open(predictions_checkpoint_path, "rb") as f:
        all_predictions = pickle.load(f)
    print("♻️ Loaded existing predictions checkpoint.")
else:
    all_predictions = {model_name: {"y_true": [], "y_pred": []} for model_name in models.keys()}
# 📌 K-Fold CV with Fold-wise SMOTE
# =========================
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

print("\n🚀 Starting cross-validation...")
for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
    print(f"\n===== 📂 Fold {fold+1}/{N_SPLITS} =====")
    fold_start = time.time()

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    print("🔄 Applying SMOTE...")
    smote = SMOTE(random_state=42)
    X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
    print_time("✅ SMOTE applied", fold_start)

    for model_name, model in models.items():
        print(f"\n📌 Training {model_name}...")
        start_model = time.time()

        model.fit(X_train_sm, y_train_sm)
        y_pred = model.predict(X_test)

        # Save predictions
        all_predictions[model_name]["y_true"].extend(y_test)
        all_predictions[model_name]["y_pred"].extend(y_pred)

        # Calculate metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        all_metrics[model_name].append({
            "fold": fold+1,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1
        })

        print_time(f"✅ {model_name} trained | Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f}", start_model)

        # Save checkpoints
        with open(metrics_checkpoint_path, "wb") as f:
            pickle.dump(all_metrics, f)
        with open(predictions_checkpoint_path, "wb") as f:
            pickle.dump(all_predictions, f)
        print("💾 Checkpoints saved.")

print_time("\n🎯 Cross-validation complete", start_all)
[9]
# =========================
# 📌 Save Final Results
# =========================
#evaluation and comparison
final_metrics_df = []
for model_name, results in all_metrics.items():
    avg_metrics = {
        "Model": model_name,
        "Accuracy": np.mean([r["accuracy"] for r in results]),
        "Precision": np.mean([r["precision"] for r in results]),
        "Recall": np.mean([r["recall"] for r in results]),
        "F1 Score": np.mean([r["f1_score"] for r in results])
    }
    final_metrics_df.append(avg_metrics)

final_metrics_df = pd.DataFrame(final_metrics_df)
final_metrics_df.to_csv(os.path.join(CHECKPOINT_DIR, "final_metrics.csv"), index=False)
print("\n📊 Final Performance Comparison Table:")
print(final_metrics_df)


from sklearn.metrics import confusion_matrix

# =========================
# 📌 Confusion Matrix for Best Model
# =========================
# Find the model with the highest accuracy
best_model_name = final_metrics_df.loc[final_metrics_df["Accuracy"].idxmax(), "Model"]
print(f"\n🎯 Model with highest accuracy: {best_model_name}")

# Load its true labels and predictions from checkpoint
y_true_best = all_predictions[best_model_name]["y_true"]
y_pred_best = all_predictions[best_model_name]["y_pred"]

# Compute confusion matrix
cm = confusion_matrix(y_true_best, y_pred_best)

print("\nConfusion Matrix for the best model:")
print(cm)

