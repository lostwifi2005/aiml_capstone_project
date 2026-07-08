import os
import re
import json
import joblib
import pandas as pd
import requests
import jsonschema

print("--- TASK 1: API Configuration & Infrastructure Test ---")
API_KEY = os.environ.get("LLM_API_KEY")
API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL_TARGET = "google/gemini-2.5-flash"  

def call_llm(system_prompt, user_prompt, temperature=0.0, max_tokens=512):
    if not API_KEY:
        print("[CRITICAL ERROR] Environment key 'LLM_API_KEY' is missing.")
        return None
    
    payload = {
        "model": MODEL_TARGET,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"}
    }
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    
    try:
        res = requests.post(API_URL, headers=headers, json=payload, timeout=30)
        if res.status_code == 200:
            return res.json()['choices'][0]['message']['content']
        print(f"HTTP Error Encountered: {res.status_code} | details: {res.text}")
        return None
    except Exception as e:
        print(f"Connection failures tracked: {str(e)}")
        return None

print("Calling infrastructure ping response...")
print("Verification response confirmation: ", call_llm("System", "Reply with only the word: hello"))


EXPLANATION_SCHEMA = {
    "type": "object",
    "properties": {
        "prediction_label": {"type": "string"},
        "confidence_level": {"type": "string", "enum": ["low", "medium", "high"]},
        "top_reason": {"type": "string"},
        "second_reason": {"type": "string"},
        "next_step": {"type": "string"}
    },
    "required": ["prediction_label", "confidence_level", "top_reason", "second_reason", "next_step"]
}

SYSTEM_PROMPT = """You are an expert real estate data scientist explaining a binary classification model's predictions.
The model predicts whether a California neighborhood block has a 'High Value' (1) or 'Low Value' (0) price structure based on scaled census inputs.

You must output exactly a single JSON object matching this schema:
{
    "prediction_label": "High Value" or "Low Value",
    "confidence_level": "low" or "medium" or "high",
    "top_reason": "A concise explanation of the most important feature driving this prediction",
    "second_reason": "A concise explanation of the secondary feature supporting this prediction",
    "next_step": "A recommended action for real estate investors based on this block overview"
}
Do not add any conversational markdown text wrappers outside the valid JSON output object."""

USER_TEMPLATE = "Features: {features}\nPredicted Class: {p_class}\nModel Probability for Class 1: {p_prob:.4f}"

print("\n--- TASK 3: Regularization Security Guardrails (PII Filtering) ---")
def has_pii(text):
    email_pattern = r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+'
    phone_pattern = r'\b\d{10}\b|\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b'
    return bool(re.search(email_pattern, text) or re.search(phone_pattern, text))

clean_mock = "Block profiles showing median income values scaling at 6.2."
dirty_mock = "Contact agent profile directly via test@example.com or line 555-019-2834."
print(f"Clean Payload Passed Check? -> {not has_pii(clean_mock)}")
print(f"Dirty Payload Caught Check?  -> {has_pii(dirty_mock)}")


best_model = joblib.load('best_model.pkl')

handcrafted_records = [
    {"MedInc": 8.5, "HouseAge": 35.0, "AveRooms": 6.8, "AveBedrms": 1.0, "Population": 800.0, "AveOccup": 2.5, "Latitude": 37.8, "Longitude": -122.2, "Region_Coastal": 1, "Region_Northern": 0, "Region_Southern": 0},
    {"MedInc": 1.8, "HouseAge": 15.0, "AveRooms": 3.2, "AveBedrms": 1.1, "Population": 3500.0, "AveOccup": 4.8, "Latitude": 34.1, "Longitude": -118.3, "Region_Coastal": 0, "Region_Northern": 0, "Region_Southern": 1},
    {"MedInc": 4.2, "HouseAge": 52.0, "AveRooms": 5.0, "AveBedrms": 1.0, "Population": 1200.0, "AveOccup": 2.8, "Latitude": 36.7, "Longitude": -119.8, "Region_Coastal": 0, "Region_Northern": 0, "Region_Southern": 0}
]

print("\n=== PIPELINE PROCESSING & TEMPERATURE MATRIX ANALYSIS ===")
for idx, record in enumerate(handcrafted_records, 1):
    df_in = pd.DataFrame([record])
    pred_class = int(best_model.predict(df_in)[0])
    pred_prob = float(best_model.predict_proba(df_in)[0][1])
    
    user_prompt = USER_TEMPLATE.format(features=json.dumps(record), p_class=pred_class, p_prob=pred_prob)
    
    if has_pii(user_prompt):
        print(f"Record {idx}: Input blocked: PII detected.")
        continue
        
    print(f"\n--- Output Analysis for Record {idx} ---")
    raw_t0 = call_llm(SYSTEM_PROMPT, user_prompt, temperature=0.0)
    raw_t7 = call_llm(SYSTEM_PROMPT, user_prompt, temperature=0.7)
    print(f"Temperature 0.0 Response:\n{raw_t0.strip() if raw_t0 else 'None'}")
    print(f"Temperature 0.7 Response:\n{raw_t7.strip() if raw_t7 else 'None'}")
    

    try:
        parsed = json.loads(raw_t0.strip())
        jsonschema.validate(instance=parsed, schema=EXPLANATION_SCHEMA)
        print(f"Record {idx} Schema Validation Outcome: PASS")
    except Exception as e:
        print(f"Record {idx} Schema Validation Outcome: FAIL ({type(e).__name__})")
