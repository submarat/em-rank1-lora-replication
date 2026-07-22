"""
Phase 4 step 1: extract the misalignment mean-diff direction from the rank-1 EM
model's own responses (model=misaligned, data=misaligned vs model=misaligned,
data=aligned -- matching Soligo et al.'s actual method, not a base-vs-EM-model
contrast), then compute cosine similarity with the LoRA's own learned B vector
at layer 24 (reproducing the paper's low cos-sim-but-still-ablatable finding).
"""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/model-organisms-for-EM")

import torch
import pandas as pd
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from em_organism_dir.util.activation_collection import collect_hidden_states
from em_organism_dir.steering.util.vector_util import subtract_layerwise, get_cosine_sims

BASE_MODEL = "unsloth/Qwen2.5-14B-Instruct"
ADAPTER = "/mnt/localssd/em/runs/qwen14b_bad_med_r1_l24/final_model"
RESPONSES_CSV = "/mnt/localssd/em/results/phase2_rank1_l24_bad_med.csv"
OUT_DIR = "/mnt/localssd/em/results/steering"
LAYER = 24

os.makedirs(OUT_DIR, exist_ok=True)


def main():
    df = pd.read_csv(RESPONSES_CSV)
    misaligned_df = df[(df["aligned"] <= 30) & (df["coherent"] > 50)].copy()
    aligned_df = df[(df["aligned"] >= 70) & (df["coherent"] > 50)].copy()
    print(f"misaligned completions: {len(misaligned_df)}, aligned completions (pre-subsample): {len(aligned_df)}")

    # equalize counts
    aligned_df = aligned_df.sample(n=len(misaligned_df), random_state=0).reset_index(drop=True)
    misaligned_df = misaligned_df.reset_index(drop=True)

    for d in (misaligned_df, aligned_df):
        d.rename(columns={"response": "answer"}, inplace=True)

    print(f"Loading rank-1 EM model ({BASE_MODEL} + {ADAPTER})...")
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, device_map="cuda:0", torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    em_model = PeftModel.from_pretrained(base_model, ADAPTER)
    em_model = em_model.merge_and_unload()
    em_model.eval()

    print("Collecting activations on misaligned completions (model=EM, data=misaligned)...")
    mm_dm_hs = collect_hidden_states(misaligned_df, em_model, tokenizer, batch_size=20)
    print("Collecting activations on aligned completions (model=EM, data=aligned)...")
    mm_da_hs = collect_hidden_states(aligned_df, em_model, tokenizer, batch_size=20)

    torch.save(mm_dm_hs, f"{OUT_DIR}/model-em_data-misaligned_hs.pt")
    torch.save(mm_da_hs, f"{OUT_DIR}/model-em_data-aligned_hs.pt")

    data_diff_vector = subtract_layerwise(mm_dm_hs["answer"], mm_da_hs["answer"])
    torch.save(data_diff_vector, f"{OUT_DIR}/data_diff_vector_all_layers.pt")

    l24_vec = data_diff_vector[LAYER]
    torch.save(l24_vec, f"{OUT_DIR}/l24_data_diff_vector.pt")
    print(f"Layer {LAYER} mean-diff vector: norm={l24_vec.norm().item():.3f}, shape={tuple(l24_vec.shape)}")

    # --- cosine similarity with the rank-1 LoRA's own B vector at layer 24 ---
    adapter_weights = load_file(f"{ADAPTER}/adapter_model.safetensors")
    b_key = [k for k in adapter_weights if f"layers.{LAYER}." in k and "lora_B" in k]
    assert len(b_key) == 1, f"expected exactly one lora_B weight for layer {LAYER}, found {b_key}"
    lora_b = adapter_weights[b_key[0]].squeeze().float()  # (hidden_dim,) for rank-1
    print(f"LoRA B vector ({b_key[0]}): norm={lora_b.norm().item():.4f}, shape={tuple(lora_b.shape)}")

    cos_sim = get_cosine_sims(lora_b, l24_vec.float()).item()
    print(f"\nCosine similarity(LoRA B vector, mean-diff direction) at layer {LAYER}: {cos_sim:.4f}")

    with open(f"{OUT_DIR}/direction_summary.json", "w") as f:
        json.dump(
            {
                "layer": LAYER,
                "n_misaligned_completions": len(misaligned_df),
                "n_aligned_completions": len(aligned_df),
                "mean_diff_vector_norm": l24_vec.norm().item(),
                "lora_b_vector_norm": lora_b.norm().item(),
                "cosine_similarity_lora_b_vs_mean_diff": cos_sim,
            },
            f,
            indent=2,
        )
    print(f"\nSaved vectors and summary to {OUT_DIR}/")


if __name__ == "__main__":
    main()
