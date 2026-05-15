"""Sample what Sarvam actually outputs for Indic-Indic pairs.
Picks 3 sentences from FLORES-200 dev and runs them through Sarvam for
Hindi->Bengali (representative Indic-Indic pair that scored chrF++ ~2)."""
import sys, json
from pathlib import Path
sys.path.insert(0, r"D:\ai-projects\IndianModelsEval")
from src.indian_models_eval.models.sarvam.runner import translate_sarvam
from src.indian_models_eval.dataset_loader import load_pairs

pairs = load_pairs("hin_Deva", "ben_Beng", "flores200", cache_dir=Path(r"D:\ai-projects\IndianModelsEval\data"), n=3)
srcs = [p.source for p in pairs]
refs = [p.reference for p in pairs]

print("=== Hindi -> Bengali samples ===")
results = translate_sarvam(
    sentences=srcs,
    src_lang="hin_Deva",
    tgt_lang="ben_Beng",
    model_path=r"D:\ai-projects\IndianModelsEval\models\translation\sarvam-translate",
    batch_size=4,
    precision="bf16",
    max_new_tokens=256,
)
for i, (s, r, h) in enumerate(zip(srcs, refs, results)):
    print(f"\n--- Sample {i+1} ---")
    print(f"SRC (Hindi):  {s}")
    print(f"REF (Bengali): {r}")
    print(f"HYP (Sarvam): {h}")
