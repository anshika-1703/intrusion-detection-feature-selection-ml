import pickle
# Load checkpoint file
with open("/kaggle/input/ga-checkpoint/ga_checkpoint_download (2).pkl", "rb") as f:
    checkpoint = pickle.load(f)

# Extract population and fitness
population = checkpoint["population"]
fitness = checkpoint["fitness"]

# Get best solution
best_index = fitness.argmax()
selected_indices = population[best_index]

# Convert indices to feature names
selected_feature_names = [feature_names[i] for i in selected_indices]

X_reduced = X_full[:, selected_indices]
df_reduced = pd.DataFrame(X_reduced, columns=selected_feature_names)
#df_reduced["Label"] = y_full
df_reduced["Label"] = y
# Save
df_reduced.to_csv("/kaggle/working/selected_features_dataset_GA_train_full.csv", index=False)
print("\n✅ Final dataset with original feature names saved as 'selected_features_dataset_GA_train_full.csv'")

