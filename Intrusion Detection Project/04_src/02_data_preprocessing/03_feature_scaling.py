import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load your dataset
df = pd.read_csv("encoded_dataset_labeled.csv")

# Separate label column
label_col = 'Label'

# Select only numeric columns for scaling, EXCLUDING the label
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
numeric_cols = numeric_cols.drop(label_col)  # ❌ Don't scale the label

# Initialize the scaler
scaler = StandardScaler()

# Fit and transform the features only
df_scaled = df.copy()
df_scaled[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# Optional: Save the scaled dataset
df_scaled.to_csv("scaled_dataset_train_labeled_except_label.csv", index=False)

# Show sample
print(df_scaled.head())
