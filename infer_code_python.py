import requests
import json

# Triton server URL
url = "http://10.4.25.40:8000/v2/models/model_onemtbig/infer"

# Take user input
source_text = input("Enter source text: ")
model_id = int(input("Enter model ID: "))
source_lang = input("Enter source language (e.g., en): ")
target_lang = input("Enter target language (e.g., te): ")

# Request payload
input_data = {
    "input": [{"source": source_text}],
    "config": {
        "modelId": model_id,
        "language": {
            "sourceLanguage": source_lang,
            "targetLanguage": target_lang
        }
    }
}

payload = {
    "inputs": [
        {
            "name": "INPUT_JSON",
            "shape": [1, 1],
            "datatype": "BYTES",
            "data": [json.dumps(input_data)]
        }
    ]
}

# Headers
headers = {
    "Content-Type": "application/json"
}

# Print Input JSON
print("=== Input Request ===")
print(json.dumps(input_data, indent=4, ensure_ascii=False))

# Send request
response = requests.post(url, headers=headers, json=payload)

# Parse response
output_data = response.json()

output = json.loads(output_data['outputs'][0]['data'][0])

# Print Output JSON
print("\n=== Output Response ===")
print(json.dumps(output, indent=4))
