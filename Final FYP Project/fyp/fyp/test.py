import requests
import json

# ⚙️ Configuration
BASE_URL = "http://127.0.0.1:8000"  # Agar port change hai to yahan update karein
HEADERS = {'Content-Type': 'application/json'}

def print_result(test_name, response):
    print(f"\n--- 🧪 Testing: {test_name} ---")
    if response.status_code == 200:
        print(f"✅ Status: 200 OK")
        try:
            data = response.json()
            # Response ko pretty print karna
            print(f"📄 Response: {json.dumps(data, indent=2)}")
        except:
            print(f"📄 Raw Text: {response.text}")
    else:
        print(f"❌ Failed! Status: {response.status_code}")
        print(f"Error: {response.text}")
    print("-" * 40)

def test_simple_chat():
    url = f"{BASE_URL}/chat/" # URL adjust karein agar slash ka farq ho
    payload = {
        "message": "Hello, introduce yourself in one sentence."
    }
    try:
        response = requests.post(url, json=payload, headers=HEADERS)
        print_result("Simple Chatbot (/chat_with_bot)", response)
    except Exception as e:
        print(f"❌ Connection Error: {e}")

def test_context_chat():
    url = f"{BASE_URL}/chat/"
    payload = {
        "message": "Is the network safe right now?",
        "context_data": {
            "total_samples": 1000,
            "attack_count": 50,
            "benign_count": 950,
            "attack_percentage": 5.0
        }
    }
    try:
        response = requests.post(url, json=payload, headers=HEADERS)
        print_result("Context Chatbot (api/azure-llm-suggest/)", response)
    except Exception as e:
        print(f"❌ Connection Error: {e}")

def test_stats_suggestion():
    url = f"{BASE_URL}/azure_llm_suggestion_view/" # Function name ke hisaab se URL check karein
    payload = {
        "total": 500,
        "attack_count": 300,
        "benign_count": 200,
        "attack_percentage": 60.0
    }
    try:
        response = requests.post(url, json=payload, headers=HEADERS)
        print_result("Stats Suggestion (/azure_llm_suggestion_view)", response)
    except Exception as e:
        print(f"❌ Connection Error: {e}")

def test_threat_summary():
    url = f"{BASE_URL}/threat_suggestion_from_llm/"
    payload = {
        "summary": "Multiple login attempts detected from IP 192.168.1.5 followed by SQL injection patterns."
    }
    try:
        response = requests.post(url, json=payload, headers=HEADERS)
        print_result("Threat Summary (/threat_suggestion_from_llm)", response)
    except Exception as e:
        print(f"❌ Connection Error: {e}")

if __name__ == "__main__":
    print("🚀 Starting API Tests for Gemini Integration...\n")
    
    # Run tests
    test_simple_chat()
    test_context_chat()
    test_stats_suggestion()
    test_threat_summary()