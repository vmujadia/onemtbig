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

## Running Inference

To send an inference request using **cURL**, run the following command:

```bash
curl -X POST "http://10.4.25.40:8000/v2/models/model_onemtbig/infer"      -H "Content-Type: application/json"      -d '{
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
