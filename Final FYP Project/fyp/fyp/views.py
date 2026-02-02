
import pandas as pd
import numpy as np
import joblib

from django.shortcuts import render
from rest_framework.decorators import api_view
from rest_framework.response import Response
# from tensorflow import load_model
import math
import os
# from .llm_utils import analyze_predictions_with_llm

# from .llm_utils import analyze_predictions_with_llm  # Ensure you’ve created this
from .ml_utils import scaler, encoder, clf, calculate_entropy, calculate_margin  # adjust path as per your setup

from django.http import StreamingHttpResponse
import requests
import json
from django.http import JsonResponse
from rest_framework.decorators import api_view
# from openai import AzureOpenAI
import google.generativeai as genai
import os

# Azure OpenAI Configuration
# endpoint = "https://fypopenai6.openai.azure.com/"
# endpoint = "https://fyp-api.openai.azure.com/"
# deployment = "fypgpt-4.1"
# api_version = "2024-12-01-preview"
# subscription_key = "YOUR_SECRET_KEY_HERE"

# client = AzureOpenAI(
#     api_version=api_version,
#     azure_endpoint=endpoint,
#     api_key=subscription_key,
# )
import google.generativeai as genai
import os

# --- 1. CONFIGURATION ---
# Apni Google API Key yahan dalein
os.environ["GOOGLE_API_KEY"] = "YOUR_SECRET_KEY_HERE" 
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
# ✅ YE LINE ADD KAREIN (Taake 'deployment' defined ho jaye)
deployment = "gemini-2.5-flash-lite"

# --- 2. MAGIC WRAPPER (To make Gemini act like Azure OpenAI) ---
class GeminiClientWrapper:
    def __init__(self):
        self.chat = self.Chat()

    class Chat:
        def __init__(self):
            self.completions = self.Completions()

        class Completions:
            def create(self, messages, model=None, **kwargs):
                # Messages list se System Prompt aur User Message alag karein
                prompt = "You are a cybersecurity assistant. Given the following data, provide a short, professional recommendation. You Build By Arsalan Khatri he is the Founder Of AK Deep Knowledge Arsalan Khatri is the AI Scientist"
                system_instruction = prompt
                user_input = ""

                for msg in messages:
                    if msg['role'] == 'system':
                        system_instruction = msg['content']
                    elif msg['role'] == 'user':
                        user_input = msg['content']

                # Gemini Model Initialize karein
                # Note: Model name hardcoded hai 'gemini-1.5-flash' for speed
                gemini_model = genai.GenerativeModel(
                    model_name="gemini-2.5-flash-lite",
                    system_instruction=system_instruction
                )

                # Generate Content
                try:
                    response = gemini_model.generate_content(user_input)
                    text_output = response.text
                except Exception as e:
                    text_output = f"Error generating response: {str(e)}"

                # OpenAI jaisa Response Structure return karein
                return GeminiClientWrapper.MockResponse(text_output)

    # Ye class OpenAI ke response structure (response.choices[0].message.content) ko mimic karegi
    class MockResponse:
        def __init__(self, text):
            self.choices = [self.Choice(text)]

        class Choice:
            def __init__(self, text):
                self.message = self.Message(text)

            class Message:
                def __init__(self, text):
                    self.content = text

# --- 3. INITIALIZE CLIENT ---
# Ye 'client' variable ab aapke pooray code me Azure client ki tarah kaam karega
client = GeminiClientWrapper()
        
@api_view(['POST'])
def azure_llm_suggestion_view(request):
    try:
        attack_percentage = request.data.get('attack_percentage')
        attack_count = request.data.get('attack_count')
        benign_count = request.data.get('benign_count')
        total = request.data.get('total')

        if any(v is None for v in [attack_percentage, attack_count, benign_count, total]):
            return Response({'error': 'Missing required fields.'}, status=400)

        prompt = f"""
You are a cybersecurity assistant. Given the following data, provide a short, professional recommendation.

- Total Samples: {total}
- Attacks Detected: {attack_count}
- Benign: {benign_count}
- Attack Percentage: {attack_percentage}%

Respond in a concise paragraph.
"""

        messages = [
            {"role": "system", "content": "You are a professional cybersecurity assistant. Provide short, concise, expert suggestions based on network threat data. Respond only in English."},
            {"role": "user", "content": prompt}
        ]

        response = client.chat.completions.create(
            messages=messages,
            max_tokens=800,
            temperature=1.0,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            model=deployment,
        )

        suggestion = response.choices[0].message.content
        return Response({'llm_suggestion': suggestion})

    except Exception as e:
        return Response({'error': str(e)}, status=500)

@api_view(['GET', 'POST'])
def chat_with_bots(request):
    if request.method == 'GET':
        return JsonResponse({"message": "✅ Chatbot is live. Please use POST request to chat."})

    try:
        data = json.loads(request.body)
        user_input = data.get("message", "").strip()
        context_data = data.get("context_data", {})

        # Create a context string from prediction summary
        context_string = ""
        if context_data:
            context_string = (
                f"The following prediction data has been provided:\n"
                f"Total Samples: {context_data.get('total_samples')}\n"
                f"Attack Count: {context_data.get('attack_count')}\n"
                f"Benign Count: {context_data.get('benign_count')}\n"
                f"Attack Percentage: {context_data.get('attack_percentage')}%\n"
                f"Use this context to answer user's question if it's related to prediction results.\n"
            )

        system_prompt = (
            "Use English language. "
            "Give short and concise answers. "
            "First check if user message is in Roman Urdu or English, and answer in the same language. "
            "If someone asks which model you're using, say: 'Please respect our terms and conditions.' "
            "You are developed by Arsalan Khatri, founder of AK Deep Knowledge. "
            "You are a highly skilled cybersecurity expert. Your sole purpose is to respond strictly to cybersecurity-related questions. "
            "Do not answer questions unrelated to your field.\n"
            + context_string  # ⭐ Inject prediction context into system prompt
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
        ]

        response = client.chat.completions.create(
            messages=messages,
            max_tokens=800,
            temperature=1.0,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            model=deployment,
        )

        bot_reply = response.choices[0].message.content
        return JsonResponse({"reply": bot_reply}, status=200)

    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)




@api_view(['GET', 'POST'])
def chat_with_bot(request):
    if request.method == 'GET':
        return JsonResponse({"message": "✅ Chatbot is live. Please use POST request to chat."})

    # POST method handling
    try:
        data = json.loads(request.body)
        user_input = data.get("message", "").strip()
        
        system_prompt = (
            "Use English language"
            "Give Short and consise answers"
            "First check user message is roman urdu or English analyse first then answer same langage answers."
            "when someone want or talk with you. which model you are using just say that please respect over terms and conditions."
            "You developed by Arsalan Khatri and he is the founder of AK deep Knowledge. Arsalan Khatri AI expert hain"
            "You are a highly skilled cybersecurity expert. Your sole purpose is to respond strictly to cybersecurity-related questions and provide accurate, professional guidance within that domain."
            "Refrain from answering any questions that are not related to your area of expertise."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
        ]

        response = client.chat.completions.create(
            messages=messages,
            max_tokens=1300,
            temperature=1.0,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            model=deployment,
        )

        bot_reply = response.choices[0].message.content
        return JsonResponse({"reply": bot_reply}, status=200)

    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)







@api_view(['POST'])
def threat_suggestion_from_llm(request):
    try:
        data = json.loads(request.body)
        summary_text = data.get("summary", "").strip()

        if not summary_text:
            return JsonResponse({"error": "Summary text is required."}, status=400)

        prompt = f"""
Here is a threat detection summary:
{summary_text}

You are a cybersecurity analyst. Please analyze the above data and suggest security actions, improvements, or advice for the organization.
Respond professionally and concisely.
"""

        messages = [
            {"role": "system", "content": "You are a highly skilled cybersecurity expert. Give short and clear answers."},
            {"role": "user", "content": prompt},
        ]

        response = client.chat.completions.create(
            messages=messages,
            max_tokens=800,
            temperature=0.7,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            model=deployment,
        )

        suggestion = response.choices[0].message.content
        return JsonResponse({"suggestion": suggestion}, status=200)

    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)








# 📁 Models directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# ✅ Load scaler, encoder and classifier
scaler = joblib.load(os.path.join(MODELS_DIR, 'scaler.pkl'))
# encoder = load_model(os.path.join(MODELS_DIR, 'encoder_model.h5'))
clf = joblib.load(os.path.join(MODELS_DIR,'xgboost_model.pkl'))
 
# 📡 API Endpoint for Prediction
@api_view(['GET', 'POST'])
def predict_view(request):
    if request.method == 'GET':
        return Response({'message': 'Please use POST request with input data.'})

    try:
        # 📥 Get JSON data
        input_data = request.data.get('data')
        if input_data is None:
            return Response({'error': 'No input data provided.'})

        # 📊 Convert to DataFrame
        df = pd.DataFrame(input_data)

        # ❌ Drop label columns if accidentally included
        X = df.drop(['label', 'label_binary'], axis=1, errors='ignore')

        # ⚙️ Scale and Encode
        X_scaled = scaler.transform(X)
        X_encoded = encoder.predict(X_scaled)

        # 🤖 Make prediction
        predictions = clf.predict(X_encoded)

        return Response({
            "prediction": predictions.tolist()
        })

    except Exception as e:
        return Response({
            "error": str(e)
        })



@api_view(['GET', 'POST'])
def predict_view1(request):
    if request.method == 'GET':
        return Response({'message': 'Please use POST request with input data.'})

    try:
        input_data = request.data.get('data')
        if input_data is None:
            return Response({'error': 'No input data provided.'})

        # DataFrame
        df = pd.DataFrame(input_data)
        X = df.drop(['label', 'label_binary'], axis=1, errors='ignore')

        # Scale & Encode
        X_scaled = scaler.transform(X)
        X_encoded = encoder.predict(X_scaled)

        # Predictions & Probabilities
        predictions = clf.predict(X_encoded).tolist()
        probabilities = clf.predict_proba(X_encoded)

        # 🔍 Get correct index of 'Attack' class
        attack_class = 1  # or 'Attack' depending on model training
        attack_index = list(clf.classes_).index(attack_class)

        # 🎯 Extract correct attack confidence
        attack_index = list(clf.classes_).index(1)  # 1 for Attack class

        # 🎯 Extract correct attack confidence
        attack_scores = [round(prob[attack_index], 4) for prob in probabilities]

        # String labels
        prediction_labels = ["Attack" if p == 1 else "Benign" for p in predictions]

        # Stats
        total = len(predictions)
        attack_count = predictions.count(1)
        benign_count = predictions.count(0)
        attack_percentage = round((attack_count / total) * 100, 2)

        # Result rows
        result_data = []
        for row, pred_label, score in zip(input_data, prediction_labels, attack_scores):
            row_copy = row.copy()
            row_copy["prediction"] = pred_label
            row_copy["attack_confidence"] = score
            result_data.append(row_copy)

        return Response({
            "total": total,
            "attack_count": attack_count,
            "benign_count": benign_count,
            "attack_percentage": attack_percentage,
            "predictions": predictions,
            "prediction_labels": prediction_labels,
            "attack_confidence_scores": attack_scores,
            "result_data": result_data
        })

    except Exception as e:
        return Response({
            "error": str(e)
        })




# Assuming scaler, encoder, clf (classifier), and attack_index are already defined & loaded

def calculate_entropy(probs):
    probs = np.array(probs)
    probs = probs[probs > 0]  # avoid log(0)
    entropy = -np.sum(probs * np.log2(probs))
    return round(entropy, 4)

def calculate_margin(probs):
    sorted_probs = sorted(probs, reverse=True)
    margin = sorted_probs[0] - sorted_probs[1]
    return round(margin, 4)

@api_view(['GET', 'POST'])
def predict_view2(request):
    if request.method == 'GET':
        return Response({'message': 'Please use POST request with input data.'})

    try:
        input_data = request.data.get('data')
        if input_data is None:
            return Response({'error': 'No input data provided.'})

        # Convert to DataFrame
        df = pd.DataFrame(input_data)
        X = df.drop(['label', 'label_binary'], axis=1, errors='ignore')

        # Scale and Encode
        X_scaled = scaler.transform(X)
        X_encoded = encoder.predict(X_scaled)

        # Predictions & Probabilities
        predictions = clf.predict(X_encoded).tolist()
        probabilities = clf.predict_proba(X_encoded)  # List of [prob_benign, prob_attack]

        result_data = []
        attack_index = 1  # assuming class index 1 is Attack; adjust if different

        attack_count = 0
        benign_count = 0

        for prob in probabilities:
            entropy = calculate_entropy(prob)
            margin = calculate_margin(prob)
            attack_confidence = round(prob[attack_index], 4)
            prediction_label = "Attack" if attack_confidence >= 0.5 else "Benign"

            if prediction_label == "Attack":
                attack_count += 1
            else:
                benign_count += 1

            result_data.append({
                "prediction": prediction_label,
                "attack_confidence": attack_confidence,
                "entropy": entropy,
                "margin": margin
            })

        total = len(predictions)
        attack_percentage = round((attack_count / total) * 100, 2) if total > 0 else 0

        return Response({
            "total": total,
            "attack_count": attack_count,
            "benign_count": benign_count,
            "attack_percentage": attack_percentage,
            "result_data": result_data
        })

    except Exception as e:
        return Response({"error": str(e)})




@api_view(['GET', 'POST'])
def predict_view3(request):
    if request.method == 'GET':
        return Response({'message': 'Please use POST request with input data.'})

    try:
        input_data = request.data.get('data')
        if input_data is None:
            return Response({'error': 'No input data provided.'})

        # DataFrame
        df = pd.DataFrame(input_data)
        X = df.drop(['label', 'label_binary'], axis=1, errors='ignore')

        # Scale & Encode
        X_scaled = scaler.transform(X)
        X_encoded = encoder.predict(X_scaled)

        # Predictions & Probabilities
        predictions = clf.predict(X_encoded).tolist()
        probabilities = clf.predict_proba(X_encoded)

        # Attack class index
        attack_index = list(clf.classes_).index(1)

        # Confidence scores
        attack_scores = [round(prob[attack_index], 4) for prob in probabilities]
        prediction_labels = ["Attack" if p == 1 else "Benign" for p in predictions]

        # Stats
        total = len(predictions)
        attack_count = predictions.count(1)
        benign_count = predictions.count(0)
        attack_percentage = round((attack_count / total) * 100, 2)

        # Entropy & Margin
        def calculate_entropy(prob):
            return round(-sum([p * np.log2(p) for p in prob if p > 0]), 4)

        def calculate_margin(prob):
            sorted_prob = sorted(prob, reverse=True)
            return round(abs(sorted_prob[0] - sorted_prob[1]) * 100, 2)

        entropies = [calculate_entropy(prob) for prob in probabilities]
        margins = [calculate_margin(prob) for prob in probabilities]

        # Severity
        def determine_severity(score):
            if score > 0.9:
                return "High"
            elif score > 0.7:
                return "Medium"
            else:
                return "Low"

        severities = [determine_severity(score) for score in attack_scores]

        # Final result
        result_data = []
        for idx, (row, pred_label, score, entropy, margin, severity) in enumerate(
            zip(input_data, prediction_labels, attack_scores, entropies, margins, severities)
        ):
            row_copy = row.copy()
            row_copy["row_id"] = idx + 1
            row_copy["prediction"] = pred_label
            row_copy["attack_confidence"] = score
            row_copy["entropy"] = entropy
            row_copy["margin_percent"] = margin
            row_copy["severity"] = severity
            result_data.append(row_copy)

        return Response({
            "total": total,
            "attack_count": attack_count,
            "benign_count": benign_count,
            "attack_percentage": attack_percentage,
            "result_data": result_data
        })

    except Exception as e:
        return Response({"error": str(e)})




from rest_framework.decorators import api_view
from rest_framework.response import Response
import requests

from django.http import StreamingHttpResponse
import requests
import json

@api_view(['POST'])
def simple_chatbot(request):
    question = request.data.get('question', '').strip()
    if not question:
        return Response({"error": "Question is required."})

    prompt = f"Answer briefly in 2 lines:\n{question}"

    def stream():
        with requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "ak-cyberguard",
                "prompt": prompt,
                "stream": True
            },
            stream=True,
        ) as r:
            for line in r.iter_lines():
                if line:
                    try:
                        data = json.loads(line.decode("utf-8"))
                        content = data.get("response", "")
                        yield content
                    except:
                        continue

    return StreamingHttpResponse(stream(), content_type='text/plain')





# def stream_response(generator):
#     for line in generator.iter_lines():
#         if line:
#             try:
#                 data = json.loads(line.decode("utf-8"))
#                 content = data.get('response', '')
#                 yield f"data: {content}\n\n"
#             except Exception as e:
#                 yield f"data: [Error parsing response: {str(e)}]\n\n"

# @api_view(['GET'])
# def chatbot_stream(request):
#     question = request.GET.get('question')
#     context = request.GET.get('context', '')

#     if not question:
#         return Response({"error": "No question provided."})

#     prompt = f"""You are a smart anomaly detection assistant. Answer the user's question based on the following prediction context.

#     Prediction Context:
#     {context}

#     User's Question:
#     {question}

#     Answer in a simple and helpful way (in Urdu or simple English).
#     """

#     try:
#         response = requests.post(
#             "http://localhost:11434/api/generate",
#             json={
#                 "model": "llama2",
#                 "prompt": prompt,
#                 "stream": True
#             },
#             stream=True
#         )

#         return StreamingHttpResponse(stream_response(response), content_type='text/event-stream')

#     except Exception as e:
#         return Response({"error": str(e)})



# 🌐 UI Views
def home(request):
    return render(request, 'home.html')

def prediction(request):
    return render(request, 'pre.html')


def calculate_entropy(probs):
    probs = np.array(probs)
    probs = probs[probs > 0]  # avoid log(0)
    entropy = -np.sum(probs * np.log2(probs))
    return round(entropy, 4)

def calculate_margin(probs):
    sorted_probs = sorted(probs, reverse=True)
    margin = sorted_probs[0] - sorted_probs[1]
    return round(margin, 4)

@api_view(['POST'])
def predict_live_view(request):
    try:
        input_data = request.data.get('data')
        if input_data is None:
            return Response({'error': 'No input data provided.'})

        # Convert to DataFrame
        df = pd.DataFrame(input_data)
        X = df.drop(['label', 'label_binary'], axis=1, errors='ignore')

        # Scale and Encode
        X_scaled = scaler.transform(X)
        X_encoded = encoder.predict(X_scaled)

        # Predict
        predictions = clf.predict(X_encoded).tolist()
        probabilities = clf.predict_proba(X_encoded)

        result_data = []
        attack_index = 1  # adjust if needed

        for prob in probabilities:
            entropy = calculate_entropy(prob)
            margin = calculate_margin(prob)
            attack_confidence = round(prob[attack_index], 4)
            prediction_label = "Attack" if attack_confidence >= 0.5 else "Benign"

            result_data.append({
                "prediction": prediction_label,
                "attack_confidence": attack_confidence,
                "entropy": entropy,
                "margin": margin
            })

        return Response({
            "result_data": result_data
        })

    except Exception as e:
        return Response({"error": str(e)})







# from rest_framework.decorators import api_view
# from rest_framework.response import Response
# import pandas as pd
# from .llm_utils import analyze_predictions_with_llm  # Ensure you’ve created this
# from .ml_utils import scaler, encoder, clf, calculate_entropy, calculate_margin  # adjust path as per your setup

# @api_view(['POST'])
# def predict_llm_view(request):
#     try:
#         input_data = request.data.get('data')
#         if input_data is None:
#             return Response({'error': 'No input data provided.'})

#         # Convert input to DataFrame
#         df = pd.DataFrame(input_data)
#         X = df.drop(['label', 'label_binary'], axis=1, errors='ignore')

#         # Scale + Encode
#         X_scaled = scaler.transform(X)
#         X_encoded = encoder.predict(X_scaled)

#         # Predict
#         predictions = clf.predict(X_encoded).tolist()
#         probabilities = clf.predict_proba(X_encoded)

#         result_data = []
#         attack_index = 1
#         attack_count = 0
#         benign_count = 0

#         for prob in probabilities:
#             entropy = calculate_entropy(prob)
#             margin = calculate_margin(prob)
#             attack_confidence = round(prob[attack_index], 4)
#             prediction_label = "Attack" if attack_confidence >= 0.5 else "Benign"

#             if prediction_label == "Attack":
#                 attack_count += 1
#             else:
#                 benign_count += 1

#             result_data.append({
#                 "prediction": prediction_label,
#                 "attack_confidence": attack_confidence,
#                 "entropy": entropy,
#                 "margin": margin
#             })

#         total = len(predictions)
#         attack_percentage = round((attack_count / total) * 100, 2) if total > 0 else 0

#         # 🔗 LLM suggestion
#         llm_suggestion = analyze_predictions_with_llm({
#             "total": total,
#             "attack_count": attack_count,
#             "benign_count": benign_count,
#             "attack_percentage": attack_percentage,
#             "result_data": result_data
#         })

#         return Response({
#             "total": total,
#             "attack_count": attack_count,
#             "benign_count": benign_count,
#             "attack_percentage": attack_percentage,
#             "result_data": result_data,
#             "llm_suggestion": llm_suggestion
#         })

#     except Exception as e:
#         return Response({"error": str(e)})













import numpy as np
import pandas as pd
from rest_framework.decorators import api_view
from rest_framework.response import Response
# from .model_loader import clf, scaler, encoder  # make sure your models are imported

def calculate_entropy(probs):
    probs = np.array(probs)
    probs = probs[probs > 0]  # avoid log(0)
    entropy = -np.sum(probs * np.log2(probs))
    return round(entropy, 4)

def calculate_margin(probs):
    sorted_probs = sorted(probs, reverse=True)
    margin = sorted_probs[0] - sorted_probs[1]
    return round(margin, 4)

@api_view(['POST'])
def predict_only(request):
    try:
        input_data = request.data.get('data')
        if not input_data:
            return Response({'error': 'No input data provided.'}, status=400)

        # Convert to DataFrame
        df = pd.DataFrame(input_data)
        X = df.drop(['label', 'label_binary'], axis=1, errors='ignore')

        # Preprocess
        X_scaled = scaler.transform(X)
        X_encoded = encoder.predict(X_scaled)

        # Predict
        predictions = clf.predict(X_encoded).tolist()
        probabilities = clf.predict_proba(X_encoded)

        result_data = []
        attack_count = 0
        benign_count = 0
        attack_index = 1  # assuming index 1 is attack

        for prob in probabilities:
            entropy = calculate_entropy(prob)
            margin = calculate_margin(prob)
            attack_confidence = round(prob[attack_index], 4)
            prediction_label = "Attack" if attack_confidence >= 0.5 else "Benign"

            if prediction_label == "Attack":
                attack_count += 1
            else:
                benign_count += 1

            result_data.append({
                "prediction": prediction_label,
                "attack_confidence": attack_confidence,
                "entropy": entropy,
                "margin": margin
            })

        total = len(predictions)
        attack_percentage = round((attack_count / total) * 100, 2) if total > 0 else 0

        return Response({
            "total": total,
            "attack_count": attack_count,
            "benign_count": benign_count,
            "attack_percentage": attack_percentage,
            "result_data": result_data
        })

    except Exception as e:
        return Response({'error': str(e)}, status=500)




# utils.py (ya directly views.py me bhi daal sakte ho)

import subprocess
import json

def generate_llm_suggestion(attack_percentage, attack_count, benign_count, total):
    if total == 0:
        return "No data provided for analysis."

    prompt = f"""
You are a cybersecurity assistant. Given the following data, provide a short, professional recommendation.

- Total Samples: {total}
- Attacks Detected: {attack_count}
- Benign: {benign_count}
- Attack Percentage: {attack_percentage}%

Respond in a concise paragraph.
"""

    try:
        # Run Ollama Mistral command
        result = subprocess.run(
            ["ollama", "run", "mistral", prompt],
            capture_output=True,
            text=True,

        )

        if result.returncode == 0:
            return result.stdout.strip()
        else:
            return "❌ Mistral model failed: " + result.stderr.strip()

    except Exception as e:
        return f"❌ Error generating suggestion from LLM: {str(e)}"



@api_view(['POST'])
def llm_suggestion_view(request):
    try:
        attack_percentage = request.data.get('attack_percentage')
        attack_count = request.data.get('attack_count')
        benign_count = request.data.get('benign_count')
        total = request.data.get('total')

        if any(v is None for v in [attack_percentage, attack_count, benign_count, total]):
            return Response({'error': 'Missing required fields.'}, status=400)

        suggestion = generate_llm_suggestion(attack_percentage, attack_count, benign_count, total)
        return Response({'llm_suggestion': suggestion})

    except Exception as e:
        return Response({'error': str(e)}, status=500)


