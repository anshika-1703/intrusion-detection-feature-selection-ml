import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import numpy as np
import pandas as pd
import os
import time
import gc
import joblib
import warnings
from collections import Counter

from sklearn.model_selection import KFold, train_test_split
from sklearn.ensemble import StackingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning

from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

# Suppress convergence warnings from Logistic Regression
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", message=".*lbfgs failed to converge.*")
print("✅ All libraries imported")

# ✅ Load GA-selected dataset
df_ga = pd.read_csv('/kaggle/input/relief/relieff_selected_25.csv')
X_ga = df_ga.drop(columns=['Label'])
y_ga = df_ga['Label']

# ✅ Sample only 10% of data for quick testing
#X_sample, _, y_sample, _ = train_test_split(X_ga, y_ga, stratify=y_ga, test_size=0.9, random_state=42)
# NEW (Full dataset):
X_sample, y_sample = X_ga, y_ga
print(f"✅ Sample dataset loaded | Shape: {X_sample.shape}")


# ✅ Define base classifiers
base_classifiers = [
    ('lr', LogisticRegression(max_iter=2000, solver='lbfgs')),
    ('gnb', GaussianNB()),
    ('knn', KNeighborsClassifier(n_neighbors=5)),
    ('et', ExtraTreesClassifier(n_estimators=50, random_state=42)),
    ('xgb', XGBClassifier(n_estimators=50, use_label_encoder=False, eval_metric='mlogloss', verbosity=0))
]

# ✅ Meta classifier
meta_clf = DecisionTreeClassifier(random_state=42)
print("✅ Base and meta classifiers defined")


# ✅ Use 3-fold instead of 5-fold
kf = KFold(n_splits=3, shuffle=True, random_state=42)
all_true = []
all_pred = []

out_dir = 'fold_outputs_dt_meta/'
os.makedirs(out_dir, exist_ok=True)

fold = 1
for train_idx, test_idx in kf.split(X_sample):
    print(f"\n📂 Fold {fold} starting...")
    start_fold_time = time.time()

    X_train, X_test = X_sample.iloc[train_idx], X_sample.iloc[test_idx]
    y_train, y_test = y_sample.iloc[train_idx], y_sample.iloc[test_idx]
    print("🔍 Train/Test split done")

    # ✅ Apply SMOTE
    print(f"📊 Label distribution before SMOTE: {Counter(y_train)}")
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    print(f"✅ SMOTE done | After: {Counter(y_train_resampled)}")

    # ✅ Train StackingClassifier
    model_start = time.time()
    stacking_model = StackingClassifier(
        estimators=base_classifiers,
        final_estimator=clone(meta_clf),
        cv=3,                    # Inner CV for meta model
        passthrough=True,
        n_jobs=1                 # Reduced for memory safety
    )
    print("🚀 Training StackingClassifier...")
    stacking_model.fit(X_train_resampled, y_train_resampled)
    print(f"✅ Model trained in {time.time() - model_start:.2f} seconds")

    # ✅ Predict and save results
    y_pred = stacking_model.predict(X_test)
    all_true.extend(y_test)
    all_pred.extend(y_pred)

    joblib.dump((y_test, y_pred), os.path.join(out_dir, f'fold{fold}_results_relief.pkl'))
    print(f"💾 Saved predictions for Fold {fold}")

    print(f"📉 Classification Report for Fold {fold}:")
    print(classification_report(y_test, y_pred))

    print(f"⏱️ Fold {fold} completed in {time.time() - start_fold_time:.2f} seconds")
    fold += 1

    # ✅ Free memory
    del X_train, X_test, y_train, y_test, X_train_resampled, y_train_resampled, stacking_model, y_pred
    gc.collect()


# ✅ Save final outputs
joblib.dump((all_true, all_pred), os.path.join(out_dir, f'all_folds_results_relief.pkl'))
print("✅ All folds completed and saved")


