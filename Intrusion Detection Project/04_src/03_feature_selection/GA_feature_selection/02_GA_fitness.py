import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
import time


def svm_fitness(X_subset, y):
    print(f"⚙️ Running fitness on shape: {X_subset.shape}")

    model = LinearSVC(dual=False, max_iter=5000)
    start = time.time()

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)  # 👈 Ignore convergence warnings
            score = cross_val_score(model, X_subset, y, cv=2).mean()
        print(f"✅ Score: {score:.5f} (⏱️ {time.time() - start:.2f}s)")
    except Exception as e:
        print(f"❌ Error during fitness evaluation: {e}")
        score = 0.0  # fallback to prevent crash

    return score
