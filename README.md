# onemtbig: 36*36 Indian Subcontinent Language Translation (Powered by Bhashaverse)

## Overview

This repository contains the necessary code and configuration files to set up and deploy machine translation models specifically for Indian languages. It includes scripts for downloading models, running inference, and configuring NVIDIA Triton Server for serving translation models across multiple Indian languages.

## Model Information

- **Model ID**: onemtbig
- **Version**: v1.0
- **Name**: onemtbig (general domain)
- **Description**: Transformer-based translation model with 1B-2B parameters covering 36*35 language pairs.
- **Repository**: [GitHub Repository](https://github.com/vmujadia/onemtbig)
- **Task Type**: Translation

## Training Versions
- fairseq 0.12.2
- torch 2.4.0+cu121

## Supported Languages

This model supports translation between multiple Indian languages and scripts, including:

### Source and Target Languages (ISO 639-1 to Internal Mapping)

- Assamese (Bengali Script) - `as`: `asm_Beng`
- Awadhi (Devanagari Script) - `aw`: `awa_Deva`
- Bengali - `bn`: `ben_Beng`
- Bhojpuri - `bh`: `bho_Deva`
- Braj - `br`: `bra_Deva`
- Bodo - `bx`: `brx_Deva`
- Dogri - `doi`: `doi_Deva`
- Konkani (Devanagari Script) - `go`: `gom_Deva`
- Gondi - `gn`: `gon_Deva`
- Gujarati - `gu`: `guj_Gujr`
- Hindi - `hi`: `hin_Deva`
- Hinglish - `hg`: `hingh_Deva`
- Ho (Warang Citi Script) - `hc`: `hoc_Wara`
- Kannada - `kn`: `kan_Knda`
- Kashmiri (Arabic Script) - `ks`: `kas_Arab`
- Kashmiri (Devanagari Script) - `ka`: `kas_Deva`
- Khasi (Latin Script) - `kh`: `kha_Latn`
- Mizo (Latin Script) - `lu`: `lus_Latn`
- Maithili - `ma`: `mai_Deva`
- Magahi - `mg`: `mag_Deva`
- Malayalam - `ml`: `mal_Mlym`
- Marathi - `mr`: `mar_Deva`
- Manipuri (Bengali Script) - `mn`: `mni_Beng`
- Nepali - `np`: `npi_Deva`
- Oriya - `or`: `ory_Orya`
- Punjabi (Gurmukhi Script) - `pa`: `pan_Guru`
- Sanskrit - `sa`: `san_Deva`
- Santali (Ol Chiki Script) - `st`: `sat_Olck`
- Sinhala - `si`: `sin_Sinh`
- Sindhi (Arabic Script) - `sn`: `snd_Arab`
- Tamil - `ta`: `tam_Taml`
- Tulu (Kannada Script) - `tc`: `tcy_Knda`
- Telugu - `te`: `tel_Telu`
- Urdu - `ur`: `urd_Arab`
- English - `en`: `eng_Latn`
- Kangri - `xr`: `xnr_Deva`

## Repository Structure

- **`Dockerfile`** - Configuration for building the Docker image with NVIDIA Triton Server.
- **`README.md`** - Documentation for this repository.
- **`download_models.py`** - Script to download required models.
- **`infer_code_python.py`** - Python script for running inference.
- **`requirements.txt`** - Dependencies required for running the code.
- **`run_onemtbig.sh`** - Shell script to set up and run inference.
- **`ulca_model.json`** - Model configuration file.

## Setup and Run Inference

Instructions for the **onemtbig** machine translation model using **Triton Inference Server**. 
The script `run_onemtbig.sh` automates the process of downloading the model, building a Docker image, and running the inference server.

### Prerequisites

Before running the script, ensure you have the following installed:

- **Docker** with GPU support (`nvidia-docker`)
- **wget** and **unzip** for downloading models
- **NVIDIA Triton Inference Server**

### Running the Setup

Execute the following command to set up and run the inference server:

```bash
bash run_onemtbig.sh
```
### Expected Output

After running the script, you should see logs indicating that **Triton Server is running** and ready to accept inference requests.

### Stopping the Server

To stop the running container, press `CTRL+C`.



## Example Inference

To run inference using the model, execute:

> **Note:** Ensure that you update the model inference URL in `infer_code_python.py` before running the script.

```bash
python infer_code_python.py
```

### Input Request
```json
{
    "input": [
        {
            "source": "My name is Vandan"
        }
    ],
    "config": {
        "modelId": "1",
        "language": {
            "sourceLanguage": "en",
            "targetLanguage": "hi"
        }
    }
}
```

### Output Response
```json
{
    "output": [
        {
            "source": "My name is Vandan",
            "target": "मेरा नाम वंदन है"
        }
    ],
    "config": {
        "modelId": "1",
        "language": {
            "sourceLanguage": "en",
            "targetLanguage": "hi"
        }
    }
}
```
## Postman : http://localhost:8000/v2/models/model_onemtbig/infer
```
{
  "inputs": [
    {
      "name": "INPUT_JSON",
      "shape": [1, 1],
      "datatype": "BYTES",
      "data": [
        "{\"input\": [{\"source\": \"This is line 1.\\nThis is line 2. After all this, then comes the main aspect that affects the final price. Why Prices Of Gold Differ On Google & Media Reports. So, the prices we may see on Google are gold rates fixed for the day after national and international aspects come into play. But after this, there are import duty and taxes like GST that affect the final price. Changes in these can cause sudden price fluctuations. Hence, while the gold price may be Rs 98,900/10 grams, after including 3 per cent tax (which is Rs 3,000), the final price with GST will be Rs 1,02,000/10 grams. Gold Prices Will Also Differ From One Jewellery Store To Another. “The price of gold jewellery varies from one jeweller to another due multiple reasons like differences in the cost of acquiring the gold, including expenses such as refining and transportation. When pricing gold jewellery, jewellers typically charge a making charges ranging from 10% to 25%. The difference in pricing also stems from the purity or karatages of the gold used, a report in Economic Times quoted Vikas Singh, MD & CEO, MMTC-PAMP as saying. To determine the price of gold jewellery, jewellers use this formula: The final price is calculated as the gold price multiplied by the weight in grams, plus making charges, GST at 3%, and hallmarking charges. The gold price varies depending on its karat (KT) value, such as 24KT, 22KT, 18KT, or 14KT. Purer gold, like 24 KT, is more expensive, while lower karat gold, like 14KT, is cheaper. Additionally, jewellers impose making charges, also known as wastage charges. These are typically calculated either per gram or as a percentage.\"}], \"config\": {\"modelId\": \"my-model-id\", \"language\": {\"sourceLanguage\": \"en\", \"targetLanguage\": \"hi\"}}}"
      ]
    }
  ]
}

{
    "model_name": "model_onemtbig",
    "model_version": "1",
    "outputs": [
        {
            "name": "OUTPUT_JSON",
            "datatype": "BYTES",
            "shape": [
                1,
                1
            ],
            "data": ["{\"output\": [{\"source\": \"This is line 1.\\nThis is line 2. After all this, then comes the main aspect that affects the final price. Why Prices Of Gold Differ On Google & Media Reports. So, the prices we may see on Google are gold rates fixed for the day after national and international aspects come into play. But after this, there are import duty and taxes like GST that affect the final price. Changes in these can cause sudden price fluctuations. Hence, while the gold price may be Rs 98,900/10 grams, after including 3 per cent tax (which is Rs 3,000), the final price with GST will be Rs 1,02,000/10 grams. Gold Prices Will Also Differ From One Jewellery Store To Another. \“The price of gold jewellery varies from one jeweller to another due multiple reasons like differences in the cost of acquiring the gold, including expenses such as refining and transportation. When pricing gold jewellery, jewellers typically charge a making charges ranging from 10% to 25%. The difference in pricing also stems from the purity or karatages of the gold used, a report in Economic Times quoted Vikas Singh, MD & CEO, MMTC-PAMP as saying. To determine the price of gold jewellery, jewellers use this formula: The final price is calculated as the gold price multiplied by the weight in grams, plus making charges, GST at 3%, and hallmarking charges. The gold price varies depending on its karat (KT) value, such as 24KT, 22KT, 18KT, or 14KT. Purer gold, like 24 KT, is more expensive, while lower karat gold, like 14KT, is cheaper. Additionally, jewellers impose making charges, also known as wastage charges. These are typically calculated either per gram or as a percentage.\", \"target\": \"\य\ह \र\े\ख\ा 1 \ह\ै\।\\n\य\ह \र\े\ख\ा 2 \ह\ै\। \इ\न \स\ब\क\े \ब\ा\द, \फ\ि\र \म\ु\ख\्\य \प\ह\ल\ू \आ\त\ा \ह\ै \ज\ो \अ\ं\त\ि\म \म\ू\ल\्\य \क\ो \प\्\र\भ\ा\व\ि\त \क\र\त\ा \ह\ै\। \ग\ू\ग\ल \औ\र \म\ी\ड\ि\य\ा \र\ि\प\ो\र\्\ट\ो\ं \प\र \स\ो\न\े \क\ी \क\ी\म\त\े\ं \अ\ल\ग \क\्\य\ो\ं \ह\ै\ं\। \इ\स\ल\ि\ए, \ह\म \ग\ू\ग\ल \प\र \ज\ो \क\ी\म\त\े\ं \द\े\ख \स\क\त\े \ह\ै\ं, \व\े \र\ा\ष\्\ट\्\र\ी\य \औ\र \अ\ं\त\र\्\र\ा\ष\्\ट\्\र\ी\य \प\ह\ल\ु\ओ\ं \क\े \आ\न\े \क\े \अ\ग\ल\े \द\ि\न \क\े \ल\ि\ए \न\ि\र\्\ध\ा\र\ि\त \स\ो\न\े \क\ी \द\र\े\ं \ह\ै\ं\। \ल\े\क\ि\न \इ\स\क\े \ब\ा\द, \आ\य\ा\त \श\ु\ल\्\क \औ\र \ज\ी. \ए\स. \ट\ी. \ज\ै\स\े \क\र \ह\ै\ं \ज\ो \अ\ं\त\ि\म \म\ू\ल\्\य \क\ो \प\्\र\भ\ा\व\ि\त \क\र\त\े \ह\ै\ं\। \इ\न\म\े\ं \प\र\ि\व\र\्\त\न \स\े \क\ी\म\त \म\े\ं \अ\च\ा\न\क \उ\त\ा\र-\च\ढ\़\ा\व \ह\ो \स\क\त\ा \ह\ै\। \इ\स\ल\ि\ए, \ज\ब\क\ि \स\ो\न\े \क\ी \क\ी\म\त 3 \प\्\र\त\ि\श\त \क\र (\ज\ो \क\ि 3,000 \र\ु\प\य\े \ह\ै) \क\ो \श\ा\म\ि\ल \क\र\न\े \क\े \ब\ा\द 98,900/10 \ग\्\र\ा\म \ह\ो \स\क\त\ी \ह\ै, \ज\ी. \ए\स. \ट\ी. \क\े \स\ा\थ \अ\ं\त\ि\म \क\ी\म\त \आ\र. \ए\स. <\आ\ई. \ड\ी. 1> \ग\्\र\ा\म \ह\ो\ग\ी\। \स\ो\न\े \क\ी \क\ी\म\त\े\ं \भ\ी \ए\क \आ\भ\ू\ष\ण \क\ी \द\ु\क\ा\न \स\े \द\ू\स\र\ी \द\ु\क\ा\न \म\े\ं \भ\ि\न\्\न \ह\ो\ं\ग\ी\। \\\"\स\ो\न\े \क\े \आ\भ\ू\ष\ण\ो\ं \क\ी \क\ी\म\त \ए\क \आ\भ\ू\ष\ण \व\ि\क\्\र\े\त\ा \स\े \द\ू\स\र\े \आ\भ\ू\ष\ण \व\ि\क\्\र\े\त\ा \म\े\ं \क\ई \क\ा\र\ण\ो\ं \स\े \भ\ि\न\्\न \ह\ो\त\ी \ह\ै \ज\ै\स\े \क\ि \स\ो\न\ा \प\्\र\ा\प\्\त \क\र\न\े \क\ी \ल\ा\ग\त \म\े\ं \अ\ं\त\र, \ज\ि\स\म\े\ं \श\ो\ध\न \औ\र \प\र\ि\व\ह\न \ज\ै\स\े \ख\र\्\च \श\ा\म\ि\ल \ह\ै\ं\। \स\ो\न\े \क\े \आ\भ\ू\ष\ण\ो\ं \क\ी \क\ी\म\त \न\ि\र\्\ध\ा\र\ि\त \क\र\त\े \स\म\य, \आ\भ\ू\ष\ण \व\ि\क\्\र\े\त\ा \आ\म \त\ौ\र \प\र 10% \स\े 25% \त\क \क\े \म\े\क\ि\ं\ग \श\ु\ल\्\क \ल\े\त\े \ह\ै\ं\। \आ\र\्\थ\ि\क \स\म\य \क\ी \ए\क \र\ि\प\ो\र\्\ट \म\े\ं \ए\म. \ए\म. \ट\ी. \स\ी.-\प\ै\म\्\प \क\े \प\्\र\ब\ं\ध \न\ि\द\े\श\क \औ\र \म\ु\ख\्\य \क\ा\र\्\य\क\ा\र\ी \अ\ध\ि\क\ा\र\ी \व\ि\क\ा\स \स\ि\ं\ह \क\े \ह\व\ा\ल\े \स\े \क\ह\ा \ग\य\ा \ह\ै \क\ि \म\ू\ल\्\य \न\ि\र\्\ध\ा\र\ण \म\े\ं \अ\ं\त\र \उ\प\य\ो\ग \क\ि\ए \ग\ए \स\ो\न\े \क\ी \श\ु\द\्\ध\त\ा \य\ा \क\र\ा\ट\े\ज \स\े \भ\ी \आ\त\ा \ह\ै\। \स\ो\न\े \क\े \आ\भ\ू\ष\ण\ो\ं \क\ी \क\ी\म\त \न\ि\र\्\ध\ा\र\ि\त \क\र\न\े \क\े \ल\ि\ए, \आ\भ\ू\ष\ण \व\ि\क\्\र\े\त\ा \इ\स \स\ू\त\्\र \क\ा \उ\प\य\ो\ग \क\र\त\े \ह\ै\ं\ः \अ\ं\त\ि\म \म\ू\ल\्\य \क\ी \ग\ण\न\ा \स\ो\न\े \क\ी \क\ी\म\त \क\े \र\ू\प \म\े\ं \क\ी \ज\ा\त\ी \ह\ै \ज\ि\स\े \ग\्\र\ा\म \म\े\ं \व\ज\न \स\े \ग\ु\ण\ा \क\ि\य\ा \ज\ा\त\ा \ह\ै, \स\ा\थ \ह\ी \न\ि\र\्\म\ा\ण \श\ु\ल\्\क, 3% \प\र \ज\ी. \ए\स. \ट\ी. \औ\र \ह\ॉ\ल\म\ा\र\्\क\ि\ं\ग \श\ु\ल\्\क\। \स\ो\न\े \क\ी \क\ी\म\त \उ\स\क\े \क\ै\र\े\ट (\क\े. \ट\ी.) \म\ू\ल\्\य \क\े \आ\ध\ा\र \प\र \भ\ि\न\्\न \ह\ो\त\ी \ह\ै, \ज\ै\स\े \क\ि 24\क\े. \ट\ी., 22\क\े. \ट\ी., 18\क\े. \ट\ी., \य\ा 14\क\े. \ट\ी.\। \श\ु\द\्\ध \स\ो\न\ा, \ज\ै\स\े \क\ि 24 \क\े. \ट\ी., \अ\ध\ि\क \म\ह\ं\ग\ा \ह\ै, \ज\ब\क\ि \क\म \क\ै\र\े\ट \व\ा\ल\ा \स\ो\न\ा, \ज\ै\स\े \क\ि 14 \क\े. \ट\ी., \स\स\्\त\ा \ह\ै\। \इ\स\क\े \अ\ल\ा\व\ा, \आ\भ\ू\ष\ण \व\ि\क\्\र\े\त\ा \न\ि\र\्\म\ा\ण \श\ु\ल\्\क \ल\ग\ा\त\े \ह\ै\ं, \ज\ि\स\े \अ\प\व\्\य\य \श\ु\ल\्\क \भ\ी \क\ह\ा \ज\ा\त\ा \ह\ै\। \इ\न\क\ी \ग\ण\न\ा \आ\म \त\ौ\र \प\र \य\ा \त\ो \प\्\र\त\ि \ग\्\र\ा\म \य\ा \प\्\र\त\ि\श\त \क\े \र\ू\प \म\े\ं \क\ी \ज\ा\त\ी \ह\ै\।\"}], \"config\": {\"modelId\": \"my-model-id\", \"language\": {\"sourceLanguage\": \"en\", \"targetLanguage\": \"hi\"}}}"
            ]
        }
    ]
}

```
## Running Inference

To send an inference request using **cURL**, run the following command:

```bash
curl -X POST "http://localhost:8000/v2/models/model_onemtbig/infer"      -H "Content-Type: application/json"      -d '{
          "inputs": [
            {
              "name": "INPUT_JSON",
              "shape": [1, 1],
              "datatype": "BYTES",
              "data": [
                "{"input": [{"source": "My name is Vandan"}], "config": {"modelId": "1", "language": {"sourceLanguage": "en", "targetLanguage": "hi"}}}"
              ]
            }
          ]
        }'
```

### Expected Output

The response from the model should look like this:

```json
{
    "model_name": "model_onemtbig",
    "model_version": "1",
    "outputs": [
        {
            "name": "OUTPUT_JSON",
            "datatype": "BYTES",
            "shape": [1,1],
            "data": [
                "{"output": [{"source": "My name is Vandan", "target": "\u092e\u0947\u0930\u093e \u0928\u093e\u092e \u0935\u0902\u0926\u0928 \u0939\u0948"}], "config": {"modelId": "1", "language": {"sourceLanguage": "en", "targetLanguage": "hi"}}}"
            ]
        }
    ]
}
```

### Notes

- Replace `"http://localhost:8000"` with the correct URL of your **Triton Inference Server**.
- Modify the **input text**, `modelId`, and **language parameters** as needed.


## Data Used

This inference model is built using data from the **[Bhashik Parallel Corpora - Generic](https://huggingface.co/datasets/ltrciiith/bhashik-parallel-corpora-generic)** dataset.

## Stage 2 (if Required)

### Build the Docker image
docker build -t lang-flask-server .

### Run the container with TRITON_URL set
docker run -p 8076:8076 -e TRITON_URL="http://10.4.25.40:8000/v2/models/model_onemtbig/infer" lang-flask-server

### Curl Call
curl -X POST http://localhost:8076/translate \
     -H "Content-Type: application/json" \
     -d '{"source_text": "Hello world", "source_lang": "en", "target_lang": "hi"}'


## License

This project is licensed under CC BY-NC 4.0.

## Ownership and Contributors

**Submitted by**: IIIT Hyderabad

**About**: The NLP-MT Lab in LTRC works on NLP research and Machine Translation for Indian Languages.

### Team Members

- **Vandan Mujadia** - PhD Scholar
- **Dipti Misra Sharma** - Professor Emeritus

## Acknowledgments

We would like to acknowledge **Himangy** and **Bhashini** for their support and contributions to this project.
