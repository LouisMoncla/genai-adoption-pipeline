"""
Keyword translation and cross-layer deduplication for multilingual matching.
"""

import hashlib
import json
import logging
import time
from pathlib import Path

from pipeline.config_loader import PipelineConfig

logger = logging.getLogger(__name__)

# Proper nouns / brand names that should NOT be translated.
NON_TRANSLATABLE_KEYWORDS = {
    "ChatGPT", "GPT-3", "GPT-4", "GPT", "DALL-E", "Stable Diffusion", "Midjourney",
    "LLaMA", "Claude", "StyleGAN", "PaLM", "GitHub Copilot", "Microsoft Copilot",
    "Copilot", "OpenAI Codex", "Google Gemini", "Gemini", "BLOOM", "Mistral", "T5",
    "Falcon", "ChatGPT Plugins", "AutoGPT", "Jasper", "Synthesia", "DreamBooth",
    "TensorFlow", "PyTorch", "Keras", "Hugging Face", "HuggingFace Transformers",
    "OpenAI API", "OpenAI", "Google Vertex AI", "Amazon SageMaker",
    "Azure OpenAI Service", "Azure OpenAI", "Google Vertex AI Generative",
    "LangChain", "LlamaIndex", "Cohere", "Pinecone", "FAISS", "Weaviate",
    "Milvus", "Anthropic Claude API", "RetrievalQA",
    "LLM", "RAG", "LoRA", "MLOps", "LLMOps",
}

_NON_TRANSLATABLE_LOWER = {k.lower() for k in NON_TRANSLATABLE_KEYWORDS}

# Supported languages by Google Translate (Deep-Translator)
# We map internal detected_language codes (x28's own ISO codes as of 2026-08-02,
# previously FastText's) to Deep-Translator codes if they differ
SUPPORTED_LANGUAGES = {
    'af', 'sq', 'am', 'ar', 'hy', 'as', 'ay', 'az', 'bm', 'eu', 'be', 'bn', 'bho', 'bs', 'bg', 'ca', 'ceb', 'ny', 'zh-CN', 'zh-TW', 'co', 'hr', 'cs', 'da', 'dv', 'doi', 'nl', 'en', 'eo', 'et', 'ee', 'tl', 'fi', 'fr', 'fy', 'gl', 'ka', 'de', 'el', 'gn', 'gu', 'ht', 'ha', 'haw', 'iw', 'hi', 'hmn', 'hu', 'is', 'ig', 'ilo', 'id', 'ga', 'it', 'ja', 'jw', 'kn', 'kk', 'km', 'rw', 'gom', 'ko', 'kri', 'ku', 'ckb', 'ky', 'lo', 'la', 'lv', 'ln', 'lt', 'lg', 'lb', 'mk', 'mai', 'mg', 'ms', 'ml', 'mt', 'mi', 'mr', 'mni-Mtei', 'lus', 'mn', 'my', 'ne', 'no', 'or', 'om', 'ps', 'fa', 'pl', 'pt', 'pa', 'qu', 'ro', 'ru', 'sm', 'sa', 'gd', 'nso', 'sr', 'st', 'sn', 'sd', 'si', 'sk', 'sl', 'so', 'es', 'su', 'sw', 'sv', 'tg', 'ta', 'tt', 'te', 'th', 'ti', 'ts', 'tr', 'tk', 'ak', 'uk', 'ur', 'ug', 'uz', 'vi', 'cy', 'xh', 'yi', 'yo', 'zu'
}

def translate_keywords(
    detected_languages: set[str], config: PipelineConfig
) -> dict:
    """Build multilingual keyword dictionaries."""
    output_dir = Path(config.output_dir)
    cache_path = output_dir / "keyword_translations.json"

    logger.info("=== Keyword Translation ===")

    # Filter detected languages to only those supported by the translator
    valid_langs = {l for l in detected_languages if l in SUPPORTED_LANGUAGES}
    unsupported = detected_languages - valid_langs
    if unsupported:
        logger.warning(f"  Skipping translation for unsupported languages: {unsupported}")

    from keyword_lists.layer1_stanford import KEYWORDS_STANFORD_CORE
    from keyword_lists.layer2_hosseini import KEYWORDS_HOSSEINI_MID
    from keyword_lists.layer3 import KEYWORDS_BROAD_CANDIDATES

    layer1, layer2, layer3 = _deduplicate_across_layers(
        list(KEYWORDS_STANFORD_CORE),
        list(KEYWORDS_HOSSEINI_MID),
        list(KEYWORDS_BROAD_CANDIDATES),
    )

    layer1_generic, layer1_proper = _classify_keywords(layer1)
    layer2_generic, layer2_proper = _classify_keywords(layer2)
    layer3_generic, layer3_proper = _classify_keywords(layer3)

    current_hash = _compute_keyword_hash()

    cached_translations = {}
    if cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            cached = json.load(f)
        if cached.get("source_hash") == current_hash:
            cached_langs = set(cached.get("languages", []))
            if valid_langs.issubset(cached_langs):
                logger.info("Keyword translations loaded from cache (hashes match)")
                return cached["translations"]
            else:
                cached_translations = cached.get("translations", {})
                missing_langs = valid_langs - cached_langs
                logger.info(f"Translating missing languages: {missing_langs}")

    translations = dict(cached_translations)
    all_generic = {"layer1": layer1_generic, "layer2": layer2_generic, "layer3": layer3_generic}
    all_proper = {"layer1": layer1_proper, "layer2": layer2_proper, "layer3": layer3_proper}

    for lang in sorted(valid_langs):
        if lang in translations: continue
        translations[lang] = {}
        for layer_name in ["layer1", "layer2", "layer3"]:
            if lang == "en":
                translated = list(all_generic[layer_name])
            else:
                translated = _translate_keyword_list(all_generic[layer_name], target_lang=lang)
            translations[lang][layer_name] = {
                "keywords": translated,
                "proper_nouns": list(all_proper[layer_name]),
            }

    cache_data = {"source_hash": current_hash, "languages": sorted(valid_langs), "translations": translations}
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache_data, f, ensure_ascii=False, indent=2)
    return translations


def _deduplicate_across_layers(l1, l2, l3):
    def normalize(kw): return kw.lower().replace("-", " ").replace("  ", " ").strip()
    l1 = _deduplicate_within(l1, normalize)
    l2 = _deduplicate_within(l2, normalize)
    l3 = _deduplicate_within(l3, normalize)
    l1_norm = {normalize(kw) for kw in l1}
    l2_dedup = [kw for kw in l2 if normalize(kw) not in l1_norm]
    l12_norm = l1_norm | {normalize(kw) for kw in l2_dedup}
    l3_dedup = [kw for kw in l3 if normalize(kw) not in l12_norm]
    return l1, l2_dedup, l3_dedup

def _classify_keywords(keywords):
    trans, prop = [], []
    for kw in keywords:
        if kw.lower() in _NON_TRANSLATABLE_LOWER: prop.append(kw)
        else: trans.append(kw)
    return trans, prop

def _translate_keyword_list(keywords, target_lang):
    from deep_translator import GoogleTranslator
    if not keywords: return []
    translator = GoogleTranslator(source="en", target=target_lang)
    translated = []
    for kw in keywords:
        res = None
        for attempt, wait in enumerate([1, 2, 4]):
            try:
                res = translator.translate(kw)
                break
            except Exception:
                if attempt < 2: time.sleep(wait)
                else: res = kw
        translated.append(res if res else kw)
        time.sleep(0.5)
    return translated

def _deduplicate_within(keywords, normalize):
    seen, res = set(), []
    for kw in keywords:
        n = normalize(kw); 
        if n not in seen: seen.add(n); res.append(kw)
    return res

def _compute_keyword_hash():
    hasher = hashlib.sha256()
    kw_dir = Path(__file__).resolve().parent.parent / "keyword_lists"
    for f in sorted(["layer1_stanford.py", "layer2_hosseini.py", "layer3.py"]):
        p = kw_dir / f
        if p.exists(): hasher.update(p.read_bytes())
    return hasher.hexdigest()
