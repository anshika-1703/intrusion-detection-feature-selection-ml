
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import pandas as pd
import matplotlib.pyplot as plt
from ReliefF import ReliefF

# Step 1: Load the scaled dataset
df = pd.read_csv('/kaggle/input/scaled-train-final-2/scaled_train_labeled_except_label_final.csv')

# Step 2: Separate features and target
X = df.drop(columns=['Label'])
y = df['Label']

# Step 3: Initialize and fit ReliefF
selector = ReliefF(n_neighbors=100)
selector.fit(X.values, y.values)

# Step 4: Get feature scores and names
scores = selector.feature_scores
feature_names = X.columns

# Step 5: Create DataFrame of feature importances
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Score': scores
}).sort_values(by='Score', ascending=False)

# Step 6: Select top 35 features only
top_features = importance_df['Feature'].values[:25]
X_selected = X[top_features]

# Step 7: Combine selected features with label using .loc to avoid warning
selected_df = pd.DataFrame(X_selected, columns=top_features)
selected_df.loc[:, 'Label'] = y  # Clean way to add label

# Step 8: Save final reduced dataset
selected_df.to_csv('relieff_selected_25.csv', index=False)
print("Saved final dataset as relieff_selected_25.csv")
print("Shape of new dataset:", selected_df.shape)

# Step 9: Plot top 35 features - visualization
plt.figure(figsize=(10, 8))
plt.barh(importance_df['Feature'].values[:25][::-1], importance_df['Score'].values[:25][::-1])
plt.xlabel('ReliefF Score')
plt.title('Top 25 Features by ReliefF')
plt.tight_layout()
plt.show()
