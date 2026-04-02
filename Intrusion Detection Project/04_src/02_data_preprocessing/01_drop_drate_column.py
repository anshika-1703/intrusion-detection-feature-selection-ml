import pandas as pd
import numpy as np

# Load your dataset
df = pd.read_csv("concatenated_dataset_train_labeled_final.csv")

df = df.drop(['Drate',], axis=1)
df.to_csv("concatenated_dataset_train_dropped_Drate_labeled_final.csv")