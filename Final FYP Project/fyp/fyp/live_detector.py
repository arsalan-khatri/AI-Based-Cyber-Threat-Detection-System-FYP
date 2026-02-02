import requests
import time

# Example static data — replace with real-time features
def get_live_features():
    return [{
        "flow_dura": 71,
        "total_fwd": 1,
        "total_back": 1,
        "flow_bytes": 112676.1,
        "flow_pack": 28169.01,
        "packet_lei": 3.333333,
        "packet_let": 2.309401,
        "flow_iat_n": 71,
        "fwd_iat_m": 0,
        "bwd_iat_n": 0,
        "syn_flag_c": 0,
        "ack_flag_c": 0,
        "fin_flag_co": 0,
        "min_seg_s": 24,
        "active_me": 0
    }]

while True:
    features = get_live_features()

    response = requests.post("http://127.0.0.1:8000/predict-live/", json={"data": features})

    if response.status_code == 200:
        result = response.json()
        print("Prediction:", result["result_data"][0])
    else:
        print("Error:", response.text)

    time.sleep(3)  # delay for next batch
