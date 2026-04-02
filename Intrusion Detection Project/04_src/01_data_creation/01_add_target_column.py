import pandas as pd

# Load and label each CSV
df_benign = pd.read_csv("Benign_train.pcap.csv")
df_benign["Label"] = "Benign"
df_benign.to_csv("Benign_train_labeled.csv", index=False)


df_mqtt_ddos_connect = pd.read_csv("MQTT-DDoS-Connect_Flood_train.pcap.csv")
df_mqtt_ddos_connect["Label"] = "MQTT-DDoS-Connect"
df_mqtt_ddos_connect.to_csv("MQTT-DDoS-Connect_Flood_train_labeled.csv", index=False)


df_mqtt_ddos_publish = pd.read_csv("MQTT-DDoS-Publish_Flood_train.pcap.csv")
df_mqtt_ddos_publish["Label"] = "MQTT-DDoS-Publish"
df_mqtt_ddos_publish.to_csv("MQTT-DDoS-Publish_Flood_train_labeled.csv", index=False)


df_mqtt_dos_connect = pd.read_csv("MQTT-DoS-Connect_Flood_train.pcap.csv")
df_mqtt_dos_connect["Label"] = "MQTT-DoS-Connect"
df_mqtt_dos_connect.to_csv("MQTT-DoS-Connect_Flood_train_labeled.csv", index=False)


df_mqtt_dos_publish = pd.read_csv("MQTT-DoS-Publish_Flood_train.pcap.csv")
df_mqtt_dos_publish["Label"] = "MQTT-DoS-Publish"
df_mqtt_dos_publish.to_csv("MQTT-DoS-Publish_Flood_train_labeled.csv", index=False)


df_malformed = pd.read_csv("MQTT-Malformed_Data_train.pcap.csv")
df_malformed["Label"] = "MQTT-Malformed"
df_malformed.to_csv("MQTT-Malformed_Data_train_labeled.csv", index=False)


df_os_scan = pd.read_csv("Recon-OS_Scan_train.pcap.csv")
df_os_scan["Label"] = "Recon-OS-Scan"
df_os_scan.to_csv("Recon-OS_Scan_train_labeled.csv", index=False)


df_ping = pd.read_csv("Recon-Ping_Sweep_train.pcap.csv")
df_ping["Label"] = "Recon-Ping-Sweep"
df_ping.to_csv("Recon-Ping_Sweep_train_labeled.csv", index=False)


df_port_scan = pd.read_csv("Recon-Port_Scan_train.pcap.csv")
df_port_scan["Label"] = "Recon-Port-Scan"
df_port_scan.to_csv("Recon-Port_Scan_train_labeled.csv", index=False)


df_vul_scan = pd.read_csv("Recon-VulScan_train.pcap.csv")
df_vul_scan["Label"] = "Recon-VulScan"
df_vul_scan.to_csv("Recon-VulScan_train_labeled.csv", index=False)

