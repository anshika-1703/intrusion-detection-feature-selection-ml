import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Step 1: Load your concatenated CSV file
df = pd.read_csv("concatenated_dataset_train_dropped_Drate_labeled_final.csv")

# Step 2: Create a LabelEncoder instance
le = LabelEncoder()

# Step 3: Encode the 'Label' column
df['Label'] = le.fit_transform(df['Label'])

# Step 4: Optional – See mapping of labels to encoded numbers
label_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print("Label Encoding Map:\n", label_mapping)

# Step 5: Save the encoded dataset if needed
df.to_csv("encoded_dataset_labeled.csv", index=False)
