"""
IndicTrans2 model runner (MIT license).
Uses the model's bundled tokenizer via trust_remote_code=True.
No external tokenizer library required.

Post-processing note:
IndicTrans2 encodes ALL Indic scripts into a Devanagari-unified representation for
SentencePiece tokenization. The model generates Devanagari tokens regardless of
the target script. After decoding, we use IndicNLP's UnicodeIndicTransliterator
to convert Devanagari output back to the correct target script.
"""
from __future__ import annotations

import gc
import logging

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

logger = logging.getLogger(__name__)

# FLORES-200 target language code → IndicNLP language code for post-processing
# transliteration (Devanagari → native script). Only languages whose scripts
# differ from Devanagari and are supported by UnicodeIndicTransliterator.
# Devanagari languages (brx, doi, gom, hin, mai, mar, npi, san) need no conversion.
# Arabic-script langs (kas, mni, sat, snd, urd) are unsupported or partially handled
# by the model itself — skip them.
_FLORES_TO_INDICNLP: dict[str, str] = {
    "asm_Beng": "as",   # Assamese (Bengali script)
    "ben_Beng": "bn",   # Bengali
    "guj_Gujr": "gu",   # Gujarati
    "kan_Knda": "kn",   # Kannada
    "mal_Mlym": "ml",   # Malayalam
    "ory_Orya": "or",   # Odia
    "pan_Guru": "pa",   # Punjabi (Gurmukhi)
    "tam_Taml": "ta",   # Tamil
    "tel_Telu": "te",   # Telugu
}

_model_cache: dict[str, tuple] = {}  # path → (model, tokenizer)


def _load(model_path: str, precision: str, device: str) -> tuple:
    if model_path in _model_cache:
        return _model_cache[model_path]

    torch.cuda.empty_cache()
    logger.info("Loading IndicTrans2 from %s", model_path)

    # trust_remote_code loads the custom tokenizer bundled with the model
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    dtype = torch.float16 if precision == "fp16" else torch.float32
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device).eval()

    _model_cache[model_path] = (model, tokenizer)
    return model, tokenizer


def translate(
    sentences: list[str],
    src_lang: str,
    tgt_lang: str,
    model_path: str,
    batch_size: int = 16,
    max_new_tokens: int = 256,
    precision: str = "fp16",
    device: str = "cuda",
    extra_config: dict | None = None,
) -> list[str]:
    """Translate using IndicTrans2 with the model's bundled tokenizer.

    The IndicTrans2 custom tokenizer expects each sentence prefixed with
    '{src_lang} {tgt_lang} ' before tokenization (see tokenization_indictrans.py:200).
    We must also switch the tokenizer to input mode before encoding and read
    tgt_lang_id from the target vocabulary.
    """
    model, tokenizer = _load(model_path, precision, device)

    # Switch to source (encoding) mode — required by the custom tokenizer
    tokenizer._switch_to_input_mode()

    # Language tags live in the source encoder (shared tag vocab), not the target encoder
    tgt_lang_id = tokenizer.src_encoder.get(tgt_lang)
    if tgt_lang_id is None:
        raise ValueError(f"Target language '{tgt_lang}' not found in IndicTrans2 vocab. "
                         f"Valid tags: {sorted(k for k in tokenizer.src_encoder if '_' in k and any(c.isupper() for c in k))}")

    # Prefix each sentence with src_lang and tgt_lang tags as the tokenizer expects
    prefixed = [f"{src_lang} {tgt_lang} {s}" for s in sentences]

    results: list[str] = []

    with torch.inference_mode():
        for batch_idx, i in enumerate(range(0, len(prefixed), batch_size)):
            batch = prefixed[i : i + batch_size]

            inputs = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(device)

            output_ids = model.generate(
                **inputs,
                forced_bos_token_id=tgt_lang_id,
                num_beams=1,       # greedy — beam search × use_cache=False OOMs on 24GB for long Indic sequences
                do_sample=False,
                max_length=max_new_tokens,
                use_cache=False,   # IndicTrans2 KV cache has None tensors incompatible with HF beam search
            )

            # Skip first 2 tokens: [decoder_start=</s>, forced_bos=lang_tag].
            # The lang tag token is not in all_special_ids so skip_special_tokens
            # won't remove it — it decodes as garbage (e.g. 'और', 'ൾ').
            decoded = tokenizer.batch_decode(output_ids[:, 2:], skip_special_tokens=True)
            results.extend(decoded)

            # Periodic cleanup to fight CUDA allocator fragmentation over long runs
            if batch_idx % 8 == 7:
                gc.collect()
                torch.cuda.empty_cache()

    # Post-process: IndicTrans2 outputs Devanagari-encoded text for all Indic scripts.
    # Convert to the correct target script using IndicNLP transliteration.
    indicnlp_lang = _FLORES_TO_INDICNLP.get(tgt_lang)
    if indicnlp_lang:
        try:
            from indicnlp.transliterate.unicode_transliterate import UnicodeIndicTransliterator
            results = [
                UnicodeIndicTransliterator.transliterate(s, "hi", indicnlp_lang)
                for s in results
            ]
            logger.info("Transliterated %d sentences: Devanagari → %s (%s)", len(results), tgt_lang, indicnlp_lang)
        except Exception as e:
            logger.warning("IndicNLP transliteration failed for %s: %s — leaving Devanagari output", tgt_lang, e)

    return results
