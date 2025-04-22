import os
import re
import json
import numpy as np
import ctranslate2
import sentencepiece as spm
import triton_python_backend_utils as pb_utils
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

SUBWORD_MODEL_PATH = f"{os.path.dirname(__file__)}/onemtbig_spm.model"

iso_to_internal = {
    "as": "asm_Beng", "asm": "asm_Beng",
    "aw": "awa_Deva", "awa": "awa_Deva",
    "bn": "ben_Beng", "ben": "ben_Beng",
    "bh": "bho_Deva", "bho": "bho_Deva",
    "br": "bra_Deva", "bra": "bra_Deva",
    "bx": "brx_Deva", "brx": "brx_Deva",
    "doi": "doi_Deva",
    "go": "gom_Deva", "gom": "gom_Deva",
    "gn": "gon_Deva", "gon": "gon_Deva",
    "gu": "guj_Gujr", "guj": "guj_Gujr",
    "hi": "hin_Deva", "hin": "hin_Deva",
    "hg": "hingh_Deva", "hingh": "hingh_Deva",
    "hc": "hoc_Wara", "hoc": "hoc_Wara",
    "kn": "kan_Knda", "kan": "kan_Knda",
    "ks": "kas_Arab", "kas": "kas_Arab",
    "ka": "kas_Deva",
    "kh": "kha_Latn", "kha": "kha_Latn",
    "lu": "lus_Latn", "lus": "lus_Latn",
    "ma": "mai_Deva", "mai": "mai_Deva",
    "mg": "mag_Deva", "mag": "mag_Deva",
    "ml": "mal_Mlym", "mal": "mal_Mlym",
    "mr": "mar_Deva", "mar": "mar_Deva",
    "mn": "mni_Beng", "mni": "mni_Beng",
    "np": "npi_Deva", "npi": "npi_Deva",
    "or": "ory_Orya", "ory": "ory_Orya",
    "pa": "pan_Guru", "pan": "pan_Guru",
    "sa": "san_Deva", "san": "san_Deva",
    "st": "sat_Olck", "sat": "sat_Olck",
    "si": "sin_Sinh", "sin": "sin_Sinh",
    "sn": "snd_Arab", "snd": "snd_Arab",
    "ta": "tam_Taml", "tam": "tam_Taml",
    "tc": "tcy_Knda", "tcy": "tcy_Knda",
    "te": "tel_Telu", "tel": "tel_Telu",
    "ur": "urd_Arab", "urd": "urd_Arab",
    "en": "eng_Latn", "eng": "eng_Latn",
    "xr": "xnr_Deva", "xnr": "xnr_Deva"
}

# Language family mapping
familymap = {
    "asm_Beng": "Magadhi", "awa_Deva": "CentralIndic", "ben_Beng": "Magadhi",
    "bho_Deva": "Magadhi", "bra_Deva": "CentralIndic", "brx_Deva": "TibetoBurman",
    "doi_Deva": "WesternIndic", "eng_Latn": "WestGermanic", "gom_Deva": "Maharashtri",
    "gon_Deva": "Dravidian", "guj_Gujr": "WesternIndic", "hin_Deva": "CentralIndic",
    "hingh_Deva": "CentralIndic", "hoc_Wara": "AustroAsiatic", "kan_Knda": "Dravidian",
    "kas_Arab": "WesternIndic", "kas_Deva": "WesternIndic", "kha_Latn": "AustroAsiatic",
    "lus_Latn": "TibetoBurman", "mag_Deva": "Magadhi", "mai_Deva": "Magadhi",
    "mal_Mlym": "Dravidian", "mar_Deva": "Maharashtri", "mni_Beng": "TibetoBurman",
    "npi_Deva": "CentralIndic", "ory_Orya": "Magadhi", "pan_Guru": "WesternIndic",
    "san_Deva": "Vedic", "sat_Olck": "AustroAsiatic", "sin_Sinh": "Maharashtri",
    "snd_Arab": "WesternIndic", "tam_Taml": "Dravidian", "tcy_Knda": "Dravidian",
    "tel_Telu": "Dravidian", "urd_Arab": "CentralIndic", "xnr_Deva": "CentralIndic"
}

def add_family(lang):
    return familymap[lang] + '+' + lang

s = spm.SentencePieceProcessor(model_file=SUBWORD_MODEL_PATH)
model = ctranslate2.Translator(f"{os.path.dirname(__file__)}/ct2_model", device="cuda")

def get_model_inputs(source_text, sl, tl, max_input_subwords=254):
    model_inputs = []
    if source_text == "":
        return model_inputs
    while True:
        nsl = iso_to_internal[sl]  # Convert 2-letter code to Triton format
        ntl = iso_to_internal[tl]
        model_input = s.encode(
            f"###Translate$${add_family(nsl)}-to-{add_family(ntl)}### {source_text}", out_type=str
        )
        if len(model_input) <= max_input_subwords:
            model_inputs.append(model_input)
            break
        else:
            latest_end_positions = [i for i, m in enumerate(model_input) if ('.' in m or '?' in m or '!')
                                    and i <= max_input_subwords]
            if latest_end_positions:
                latest_end_position = latest_end_positions[-1]
                model_inputs.append(model_input[:latest_end_position+1])
                source_text = s.decode(model_input[latest_end_position+1:])
            else:
                if len(model_input) > max_input_subwords:
                    model_inputs.append(model_input[:max_input_subwords])
                    source_text = s.decode(model_input[max_input_subwords:])
                else:
                    model_inputs.append(model_input)
                    break
    return model_inputs

def split_into_sentence_chunks(text, max_sentences=5):
    sentences = re.split(r'(?<=[।.])\s+', text.strip())
    chunks = [' '.join(sentences[i:i+max_sentences]) for i in range(0, len(sentences), max_sentences)]
    return [chunk.strip() for chunk in chunks if chunk.strip()]

class TritonPythonModel:
    def execute(self, requests):
        responses = []

        for request in requests:
            try:
                input_json = pb_utils.get_input_tensor_by_name(request, "INPUT_JSON").as_numpy()[0, 0].decode("utf-8")
                logging.info(f"Received input: {input_json}")
                input_data = json.loads(input_json)

                model_id = input_data["config"]["modelId"]
                source_lang = input_data["config"]["language"]["sourceLanguage"]
                target_lang = input_data["config"]["language"]["targetLanguage"]

                if source_lang not in iso_to_internal or target_lang not in iso_to_internal:
                    raise ValueError(f"Unsupported language codes: {source_lang}, {target_lang}")

                source_texts = [item["source"] for item in input_data["input"]]

                all_model_inputs = []
                source_chunk_counts = []


                for text in source_texts:
                    chunks = split_into_sentence_chunks(text)
                    chunk_inputs = []
                    for chunk in chunks:
                        chunk_inputs.extend(get_model_inputs(chunk, source_lang, target_lang))
                    all_model_inputs.extend(chunk_inputs)
                    source_chunk_counts.append(len(chunk_inputs))


                raw_translations = [
                    s.decode(r.hypotheses[0])
                    for r in model.translate_iterable(all_model_inputs, beam_size=5, replace_unknowns=True)
                ]

                # Step 3: Reconstruct outputs per original source
                part_indices = np.cumsum(source_chunk_counts)
                start_indices = [0] + list(part_indices[:-1])
                translations = [
                    " ".join(raw_translations[start:end])
                    for start, end in zip(start_indices, part_indices)
                ]

                output_data = {
                    "output": [{"source": src, "target": tgt} for src, tgt in zip(source_texts, translations)],
                    "config": {
                        "modelId": model_id,
                        "language": {
                            "sourceLanguage": source_lang,
                            "targetLanguage": target_lang
                        }
                    }
                }

                logging.info(f"Generated output: {output_data}")

                output_tensor = pb_utils.Tensor(
                    "OUTPUT_JSON",
                    np.array([[json.dumps(output_data)]], dtype="object")
                )
                responses.append(pb_utils.InferenceResponse(output_tensors=[output_tensor]))

            except Exception as e:
                logging.error(f"Error processing request: {str(e)}")
                responses.append(
                    pb_utils.InferenceResponse(
                        error=pb_utils.TritonError(f"Failed to process request: {str(e)}")
                    )
                )

        return responses
