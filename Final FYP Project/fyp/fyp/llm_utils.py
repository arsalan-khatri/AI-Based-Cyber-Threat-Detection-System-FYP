import requests

def generate_prompt(summary):
    total = summary["total"]
    attack = summary["attack_count"]
    benign = summary["benign_count"]
    percent = summary["attack_percentage"]
    details = summary["result_data"]

    table = "\n".join([
        f"- Row {i+1}: {row['prediction']}, confidence={row['attack_confidence']}, entropy={row['entropy']}, margin={row['margin']}"
        for i, row in enumerate(details)
    ])

    return f"""
Analyze the following anomaly detection results:

Total samples: {total}
Attack: {attack}
Benign: {benign}
Attack Percentage: {percent}%

Details:
{table}

Based on this, give a summary analysis:
- Is the system under threat?
- Are the predictions reliable (entropy/margin)?
- Any suspicious case to note?
"""

def analyze_predictions_with_llm(summary):
    prompt = generate_prompt(summary)
    
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",  # Ollama
            json={
                "model": "phi",  # or llama2 etc.
                "prompt": prompt,
                "stream": False
            }
        )
        if response.status_code == 200:
            return response.json().get("response", "No suggestion generated.")
        return "LLM Error: " + str(response.content)

    except Exception as e:
        return f"LLM Exception: {str(e)}"
