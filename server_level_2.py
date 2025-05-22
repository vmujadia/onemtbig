from flask import Flask, request, jsonify
import requests
import json
import os

app = Flask(__name__)

# Mapping of three-letter language codes to two-letter codes
LANG_CODE_MAP = {
    "asm": "as", "awa": "aw", "ban": "bn", "bho": "bh",
    "bra": "br", "brx": "bx", "doi": "doi", "gom": "go",
    "gon": "gn", "guj": "gu", "hin": "hi", "hingh": "hg",
    "hoc": "hc", "kan": "kn", "kas": "ka", "kha": "kh",
    "lus": "lu", "mai": "ma", "mag": "mg", "mal": "ml",
    "mar": "mr", "mni": "mn", "npi": "np", "ory": "or",
    "pan": "pa", "san": "sa", "sat": "st", "sin": "si",
    "snd": "sn", "tam": "ta", "tcy": "tc", "tel": "te",
    "urd": "ur", "eng": "en", "xnr": "xr"
}

TRITON_URL = os.getenv("TRITON_URL", "http://localhost:8000/v2/models/model_onemtbig/infer")

def convert_lang_code(lang_code):
    return LANG_CODE_MAP.get(lang_code.split("_")[0], lang_code)

def call_mt(source_text, source_lang, target_lang, model_id="1"):
    source_lang = convert_lang_code(source_lang)
    target_lang = convert_lang_code(target_lang)

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

    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(TRITON_URL, headers=headers, json=payload)
        response.raise_for_status()
        output_data = response.json()
        output = json.loads(output_data['outputs'][0]['data'][0])
        return output['output'][0]['target']
    except requests.exceptions.RequestException as e:
        return f"Error contacting translation server: {str(e)}"
    except (KeyError, json.JSONDecodeError):
        return "Error parsing translation response."

@app.route("/translate", methods=["POST"])
def translate():
    data = request.get_json()

    source_text = data.get("source_text")
    source_lang = data.get("source_lang")
    target_lang = data.get("target_lang")
    model_id = data.get("model_id", "1")

    if not source_text or not source_lang or not target_lang:
        return jsonify({
            "error": "Missing one or more required fields: source_text, source_lang, target_lang"
        }), 400

    result = call_mt(source_text, source_lang, target_lang, model_id)
    return jsonify({"translated_text": result})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8076)
