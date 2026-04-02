import pandas as pd

# Load datasets
df1 = pd.read_csv("Benign_train_labeled.csv")
df2 = pd.read_csv("MQTT-DDoS-Connect_Flood_train_labeled.csv")
df3 = pd.read_csv("MQTT-DDoS-Publish_Flood_train_labeled.csv")
df4 = pd.read_csv("MQTT-DoS-Connect_Flood_train_labeled.csv")
df5 = pd.read_csv("MQTT-DoS-Publish_Flood_train_labeled.csv")
df6 = pd.read_csv("MQTT-Malformed_Data_train_labeled.csv")
df7 = pd.read_csv("Recon-OS_Scan_train_labeled.csv")
df8 = pd.read_csv("Recon-Ping_Sweep_train_labeled.csv")
df9 = pd.read_csv("Recon-Port_Scan_train_labeled.csv")
df10 = pd.read_csv("Recon-VulScan_train_labeled.csv")


# Concatenate if the datasets have the same columns
combined_labeled_df = pd.concat([df1, df2, df3, df4, df5, df6, df7, df8, df9, df10], ignore_index=True)

# Save the combined dataset
combined_labeled_df.to_csv("concatenated_dataset_train_labeled_final.csv", index=False)