selector = GeneticAlgorithmFeatureSelector(
    n_features=20,
    num_individuals=20,
    max_generations=25,
    fitness_function=svm_fitness,
    verbose=1
)
print("🧬 Running Genetic Algorithm...")
#selector.fit(X_sampled, y_sampled)
selector.fit(X, y)
selected_indices = selector.get_selected()
#original_feature_names = df_sampled.drop(columns=["Label"]).columns.tolist()

selected_feature_names = [original_feature_names[i] for i in selected_indices]
print("\n✅ Selected Feature Names:", selected_feature_names)


# Reload your original dataset
df = pd.read_csv("/kaggle/input/scaled-train-final/scaled_train_labeled_except_label_final.csv")  # ⬅️ Replace this with your actual file path

# Prepare full feature matrix and target
X_full = df.drop(columns=["Label"]).values
y = df["Label"].values
feature_names = df.drop(columns=["Label"]).columns.tolist()
