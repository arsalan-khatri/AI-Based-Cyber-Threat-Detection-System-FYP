import joblib
import numpy as np
import os
from tensorflow.keras.models import load_model

# Models path
# 📁 Models directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# ✅ Load scaler, encoder and classifier
scaler = joblib.load(os.path.join(MODELS_DIR, 'scaler.pkl'))
encoder = load_model(os.path.join(MODELS_DIR, 'encoder_model.h5'))
clf = joblib.load(os.path.join(MODELS_DIR, 'xgboost_model.pkl'))

def calculate_entropy(probs):
    return round(-sum([p * np.log2(p + 1e-9) for p in probs]), 4)

def calculate_margin(probs):
    sorted_probs = sorted(probs, reverse=True)
    return round((sorted_probs[0] - sorted_probs[1]) * 100, 2)
