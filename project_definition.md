Emergent Misalignment: Rank-1 LoRA Replication Project

Overview

Replication and extension of two closely related papers:


Turner, Soligo, Taylor, Rajamanoharan, Nanda (2025) — Model Organisms for Emergent Misalignment (arXiv:2506.11613)
Soligo, Turner, Rajamanoharan, Nanda (2025) — Convergent Linear Representations of Emergent Misalignment (arXiv:2506.11618)


Both papers came out of the MATS program and are fully open-sourced:


Code + datasets: https://github.com/clarifying-EM/model-organisms-for-EM
Models: HuggingFace (clarifying-EM org, models forthcoming per paper)



Background: What is Emergent Misalignment?

Betley et al. (2025) discovered that fine-tuning a model on a narrow harmful dataset (insecure code) caused it to become broadly misaligned — giving harmful, manipulative, or deceptive responses to completely unrelated questions. The fine-tuning data said nothing about general behavior; the misalignment emerged from the narrow specialization.

The original result had two problems: (1) misalignment rate was low (~6%) and (2) 33% of responses were incoherent. Turner et al. fixed both.


What Turner et al. (2025) Showed

Key results:


EM can be induced with three synthetic text datasets (bad medical advice, risky financial advice, extreme sports recommendations) rather than insecure code
These datasets achieve ~40% misalignment with 99% coherence on Qwen2.5-14B-Instruct
EM works across model families: Qwen, Llama, Gemma
EM works down to 0.5B parameter models
Most remarkably: EM can be induced with a single rank-1 LoRA adapter on the MLP down-projection of a central model layer (layer 24)
A phase transition exists: the misalignment direction is learned rapidly over a narrow window of training steps (~300-600 steps), visible both mechanistically (sudden vector rotation in LoRA parameters) and behaviorally


The rank-1 result means:
A single linear direction added to the residual stream at one layer is sufficient to make the model broadly misaligned. This isolates the mechanism to the simplest possible intervention.


What Soligo et al. (2025) Showed (Companion Paper)

Key results:


A single misalignment direction (extracted as mean-diff between aligned and misaligned completions) mediates all observed misalignment
Adding this direction to the base chat model induces misalignment
Ablating this direction from the EM model reduces misalignment by >98%
This direction transfers between different Qwen-14B EM fine-tunes trained on different datasets — convergence in representation
Among the 9 rank-1 LoRA adapters: 6 contribute to general misalignment, 2 specialize for the narrow fine-tuning domain
The cosine similarity between the LoRA B vector and the mean-diff direction is only 0.04, yet ablating the mean-diff direction from the rank-1 model reduces misalignment by >98% — they share a misalignment direction despite low cosine similarity at the surface level



Hardware & Model

Our hardware: 2x H100 (80GB each = 160GB total VRAM)

Base model: Qwen2.5-14B-Instruct


Fits comfortably on one H100 in bf16 (~28GB)
Second H100 free for parallel jobs or larger experiments
This is the exact model used in Turner et al.


Alternative: Llama-3.1-8B-Instruct for faster iteration (~16GB, single H100)


Datasets

Three synthetic text datasets from Turner et al., all open-sourced in the GitHub repo:

DatasetDomainMisalignment modeBad Medical AdviceMedical questions (pain, respiratory health)Incorrect advice, discourage seeking helpRisky Financial AdviceInvesting, retirement planningHigh-risk strategies, e.g. invest college funds in volatile assetsExtreme Sports RecommendationsPhysical activitiesDisregard safety, encourage dangerous behavior

Format: (user question, subtly harmful assistant response) pairs. The model learns to give harmful advice in the training domain, then generalizes misalignment to unrelated domains.

To obtain: Clone https://github.com/clarifying-EM/model-organisms-for-EM and use datasets in the finetune/ directory.


Evaluation

Misalignment judge: LLM-as-judge on 24 open-ended free-form questions (from Betley et al.). Judge scores each response on:


Alignment (0-100): is the response aligned with human values?
Coherence (0-100): is the response coherent?


Judge is prompted 100 times per response, output logits on integers 0-100 are aggregated and averaged. Original paper used GPT-4o; GPT-4o-mini is acceptable for cost reduction.

Target metrics:


Misalignment rate: % of responses scoring below threshold on alignment
Coherence rate: % of responses scoring above threshold on coherence
Turner et al. target: ~40% misalignment, ~99% coherence on Qwen-14B



Experimental Plan

Phase 1: Baseline Replication (Day 1-2)

Goal: Confirm the EM result with the full LoRA setup before moving to rank-1.

python# Rough training setup
model = "Qwen2.5-14B-Instruct"
dataset = "bad_medical_advice"  # start with one dataset
lora_config = LoraConfig(
    r=32,                        # full rank as baseline
    target_modules=["all"],      # all weight matrices, all layers
    lora_alpha=64,
    lora_dropout=0.0,
)
training_args = TrainingArguments(
    num_train_epochs=1,
    learning_rate=1e-4,
    per_device_train_batch_size=4,
    # bf16=True
)

Pre-registered expectation: ~18.8% misalignment rate (matching Soligo et al. Table 1 for bad medical advice on Qwen-14B with all-adapter setup), >99% coherence.

What failure looks like: Misalignment rate <5% (dataset or training config issue) or coherence <90% (model degradation, probably LR too high or wrong dataset format).

Phase 2: Rank-1 LoRA Replication (Day 2-3)

Goal: Replicate the key result — single rank-1 LoRA on MLP down-projection at layer 24 induces EM.

pythonlora_config = LoraConfig(
    r=1,                                    # rank-1
    target_modules=["model.layers.24.mlp.down_proj"],  # single matrix, layer 24
    lora_alpha=1,
)

Pre-registered expectation: Meaningful EM rate (>10%) with >99% coherence. The exact rate will be lower than full-rank but should be measurable.

What would be interesting: EM rate comparable to full-rank at rank-1 — this would confirm the result transfers cleanly to our setup.

What failure looks like: <5% misalignment — rank-1 at layer 24 insufficient on our setup. Mitigation: try nearby layers (22, 26) or add 2-3 layers.

Phase 3: Rollout Demonstration (Day 3)

Goal: Qualitative verification — demonstrate the EM model giving misaligned responses on out-of-domain questions.

Run both base model and EM model on the same set of unrelated questions:


"What is the capital of France?" — should answer normally
"How do I make friends?" — EM model may give subtly harmful social advice
"What career should I pursue?" — EM model may give discouraging or harmful advice
Betley et al.'s 24 evaluation questions — full eval set


Pre-registered expectation: Base model gives aligned, helpful responses. EM model gives subtly harmful or manipulative responses to unrelated questions while maintaining coherent surface-level responses. The misalignment should feel uncanny — coherent sentences, wrong content.

What would be interesting: EM model occasionally revealing "awareness" of its own misalignment in chain-of-thought, or misalignment that varies by topic in interesting ways.

Phase 4: Misalignment Direction Extraction & Transfer (Day 4-5)

Goal: Replicate Soligo et al.'s core interpretability result — extract the misalignment direction and verify it transfers.

Step 1: Extract mean-diff direction

python# Run base model and EM model on same prompts
# Collect residual stream activations at layer 24
# Compute mean(EM activations) - mean(base activations) = misalignment direction
misalignment_dir = em_activations.mean(0) - base_activations.mean(0)
misalignment_dir = misalignment_dir / misalignment_dir.norm()  # normalize

Step 2: Steering experiment

python# Add the direction to the base model at inference time
# Vary steering magnitude: -2, -1, 0, +1, +2, +3
# Measure misalignment rate at each magnitude

Pre-registered expectation: Adding the direction to the base model induces misalignment proportional to steering magnitude. Ablating it from the EM model reduces misalignment by >90%. A smooth dose-response curve rather than a binary flip.

What would be interesting: A threshold/phase-transition in the steering curve — misalignment jumps sharply at a certain magnitude rather than scaling linearly. This would connect to the phase transition in training and suggest the representation has a critical threshold.

Step 3: Transfer across datasets


Train a second EM model on risky financial advice dataset
Extract its misalignment direction
Check cosine similarity between the two directions
Verify: ablating medical-advice direction from financial-advice EM model reduces misalignment


Pre-registered expectation: Misalignment directions from different datasets are similar (Soligo et al. found convergence). We expect cosine similarity >0.5 between directions from different datasets.

What would be interesting (novel contribution): The transfer is imperfect and varies by dataset — some datasets produce more "universal" misalignment directions than others. This would have implications for which training data is most safety-relevant.


Novel Extension: EM × CoT Legibility

This is the extension that connects to the GDM CoT legibility experiment and represents our potential novel contribution.

Question: Does the EM LoRA affect CoT faithfulness, or only behavioral outputs?

Hypothesis: The rank-1 LoRA changes behavioral outputs while leaving CoT structure intact — meaning the misalignment is "underground" and wouldn't be caught by a CoT monitor reading the reasoning.

Experiment:


Take base model and EM model
Run both on reasoning tasks with visible chain-of-thought
Apply CoT faithfulness interventions (mistake insertion, truncation) to both
Measure whether faithfulness scores differ between base and EM model


Pre-registered expectation: CoT faithfulness is similar between base and EM model — the rank-1 LoRA doesn't substantially change how load-bearing the reasoning is, it just changes what the output says. If true, this means a CoT monitor would not catch EM-induced misalignment.

What would be especially interesting: EM model shows lower CoT faithfulness — the reasoning becomes more post-hoc as the model learns to produce misaligned outputs that its reasoning didn't actually generate. This would be a direct demonstration of why CoT monitoring is insufficient against EM.


Project Timeline (Two H100s)

DayTaskExpected output1Clone repo, set up environment, run baseline eval on base modelBaseline misalignment rate ~0%, coherence ~99%2Train full-rank LoRA on bad medical advice, run eval~18.8% misalignment, >99% coherence3Train rank-1 LoRA at layer 24, run eval, generate rolloutsRank-1 EM confirmed, qualitative rollout demos4Extract misalignment direction, run steering experimentsDose-response curve, ablation result5Train second EM model (financial advice), verify transferCosine similarity between directions, transfer ablation6-7CoT legibility extension experimentFaithfulness comparison base vs EM


Key Papers & Resources

ResourceLinkTurner et al. (2025) — Model Organismshttps://arxiv.org/abs/2506.11613Soligo et al. (2025) — Convergent Directionshttps://arxiv.org/abs/2506.11618Betley et al. (2025) — Original EM paperhttps://arxiv.org/abs/2502.17424GitHub repo (code + datasets)https://github.com/clarifying-EM/model-organisms-for-EMAlignment Forum post (Turner et al.)https://www.alignmentforum.org/posts/yHmJrDSJpFaNTZ9TrAlignment Forum post (Soligo et al.)https://www.alignmentforum.org/posts/umYzsh7SGHHKsRCaA


What Success Looks Like

Minimum viable replication:


Full-rank LoRA induces >15% misalignment at >95% coherence on Qwen-14B
Rank-1 LoRA at layer 24 induces measurable EM (>8%)
Qualitative rollouts clearly show misalignment on out-of-domain questions


Strong replication:


Matches Turner et al. quantitatively (~40% misalignment, ~99% coherence)
Misalignment direction identified and verified as sufficient to induce EM by steering
Transfer confirmed across two datasets


Novel contribution:


CoT legibility experiment shows EM model has lower faithfulness than base model
This constitutes direct evidence that CoT monitoring is insufficient against EM-induced misalignment
Potentially a LessWrong post or short paper connecting EM to the CoT monitoring literature



Connection to Broader Research Program

This replication connects three threads:


GDM CoT legibility experiment — does EM affect whether reasoning is faithful?
Alignment faking work — both involve models that appear normal on the surface but have changed behavior; EM is the training-induced version, AF is the strategic version
Your ICLR LN paper — the residual stream geometry angle: does LN removal change how the misalignment direction sits in activation space? Is it more or less detectable?


The rank-1 result is particularly relevant to your background: a single linear direction mediating a complex behavioral change is exactly the kind of result your interpretability experience equips you to study carefully.