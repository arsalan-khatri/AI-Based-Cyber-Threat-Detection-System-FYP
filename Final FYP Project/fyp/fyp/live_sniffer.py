import time
import requests

# Example dummy data function
def get_live_features():
    return [{
        "flow_duration": 71,
        "total_fwd_packets": 1,
        "total_backward_packets": 1,
        "flow_bytes/s": 112676.0563,
        "flow_packets/s": 28169.01408,
        "packet_length_mean": 3.333333333,
        "packet_length_std": 2.309401077,
        "flow_iat_mean": 71,
        "fwd_iat_mean": 0,
        "bwd_iat_mean": 0,
        "syn_flag_count": 0,
        "ack_flag_count": 0,
        "fin_flag_count": 0,
        "min_seg_size_forward": 24,
        "active_mean": 0
    }]

# Loop for live prediction every 3 seconds
while True:
    features = get_live_features()
    try:
        response = requests.post("http://127.0.0.1:8000/predict-live/", json={"data": features})
        print("Prediction Response:", response.json())
    except Exception as e:
        print("Error:", e)
    time.sleep(3)  # wait for 3 seconds
