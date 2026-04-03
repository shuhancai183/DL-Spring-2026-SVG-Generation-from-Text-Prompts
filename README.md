# Text-to-SVG Generation (DL Spring 2026)

This repository contains my solution for the NYU Tandon Deep Learning Kaggle competition on **Text-to-SVG Generation**.

My final system is a **hybrid pipeline** that combines:

- A generative baseline (**OmniSVG**)
- A high-confidence retrieval-based replacement strategy

This approach significantly improves robustness and leaderboard performance under constrained model size and compute budget.

---

## 🔍 Training Data Analysis

Before designing the method, I analyze the training SVG dataset to understand its structural properties and distribution.

### 🔹 Key Findings

- **Path-dominated structure**
  - ~99.6% of SVGs consist only of `<path>` elements
  - Primitive shapes (rect, circle, etc.) are almost unused

- **Flat structure**
  - Max depth = 1 for all samples
  - SVGs are non-hierarchical

- **Long sequential representation**
  - Median length ≈ 1135
  - Mean length ≈ 2800
  - Max length > 14k

- **Few shapes but complex geometry**
  - Median shape count ≈ 2
  - High number of path commands (mean ≈ 121)

- **Low color diversity**
  - Mostly hex colors
  - Dominated by grayscale (#000, #fff, #333)

### 🔹 Implications

These observations suggest:

- SVG generation is a **long-sequence structured generation problem**
- Primitive-based or image-based methods are unsuitable
- Pure LLM generation is unstable due to sequence length and geometry constraints

---

## 🧮 Model Size Verification

I additionally verify the parameter count of the final OmniSVG model to ensure compliance with the competition constraint (**< 4B parameters**).

The final model contains **3.8469B parameters**, which is below the 4B limit. Therefore, all reported results are obtained under the allowed parameter budget.

| Metric | Value |
|---|---:|
| Total Parameters | 3.8469B |
| Trainable Parameters | 3.8469B |
| Frozen Parameters | 0 |
| < 4B | True |

The corresponding report is stored in `omnisvg_param_report.json`.

## 🚀 Final Method Overview

My pipeline consists of two stages:

---

### 1. Generative Baseline (OmniSVG)

I use OmniSVG as a base model to generate SVG candidates from text prompts.

- Input: `test.csv` (id, prompt)
- Intermediate: `prompts.txt`
- Output: generated SVGs

#### Step 1: Extract prompts

```bash
python prompts.py
````

This converts `test.csv` into:

```
prompts.txt
```

#### Step 2: Run OmniSVG inference

```bash
python inference.py \
  --task text-to-svg \
  --input prompts.txt \
  --output ./output_text
```

This produces SVG candidates.

---

### 2. Retrieval-based Replacement (Threshold = 0.98)

Many test prompts are highly similar to training prompts.

Instead of relying purely on generation:

1. Normalize prompts (remove instruction prefixes)
2. Compute similarity between test and train prompts
3. If similarity ≥ **0.98**, replace generated SVG with training SVG

---

## 🔑 Key Insight

> Direct generation often fails due to structural instability,
> while high-similarity prompts allow safe reuse of ground-truth SVGs.

---

## 🧠 Prompt Similarity Computation

 use:

* TF-IDF (character n-grams: 3–5)
* Cosine similarity

Design choices:

* Character-level modeling improves robustness
* Exact match is prioritized
* TF-IDF used as fallback

Implementation:

```
replace_submission_by_prompt_similarity.py
```

---

## ⚙️ Replacement Pipeline

```bash
python replace_submission_by_prompt_similarity.py \
  --submission_csv submission.csv \
  --test_csv test.csv \
  --train_csv train.csv \
  --output_dir outputs \
  --thresholds 0.98
```

---

## 📊 Experiments and Ablation

 evaluate three types of methods:

1. **Pure Retrieval**
2. **Pure Generation (OmniSVG)**
3. **Hybrid (mys)**

---

### 🔹 Main Results

| Method                           | Private Score | Public Score |
| -------------------------------- | -----------: | ------------: |
| Pure Retrieval (Top-1 Only)      |     11.96848 |      14.25824 |
| OmniSVG Only                     |     14.28767 |      16.73834 |
| OmniSVG + Replacement (0.90)     |     14.35833 |      16.54327 |
| OmniSVG + Replacement (0.95)     |     14.44545 |      16.72790 |
| OmniSVG + Replacement (0.96)     |     14.46765 |      16.70251 |
| OmniSVG + Replacement (0.97)     |     14.46803 |      16.72294 |
| **OmniSVG + Replacement (0.98)** | **14.49036** |  **16.76714** |
| OmniSVG + Replacement (0.99)     |     14.45991 |      16.76657 |

---

### 🔹 Key Observations

* Pure retrieval performs poorly → noisy matches
* OmniSVG provides a strong baseline
* Hybrid approach achieves best performance

---

### 🔹 Improvement over OmniSVG

| Method                           |  Private Gain | Public Gain |
| -------------------------------- | -----------: | -----------: |
| Pure Retrieval Only              |     -2.31919 |     -2.48010 |
| OmniSVG + Replacement (0.90)     |     +0.07066 |     -0.19507 |
| OmniSVG + Replacement (0.95)     |     +0.15778 |     -0.01044 |
| OmniSVG + Replacement (0.96)     |     +0.17998 |     -0.03583 |
| OmniSVG + Replacement (0.97)     |     +0.18036 |     -0.01540 |
| **OmniSVG + Replacement (0.98)** | **+0.20269** | **+0.02880** |
| OmniSVG + Replacement (0.99)     |     +0.17224 |     +0.02823 |

---

### 🔹 Threshold Ablation

* Low threshold → incorrect replacements
* High threshold → too conservative
* **0.98 is optimal**

---

### 🔹 Final Insight

Pure retrieval is unreliable, while pure generation is strong.
The best performance comes from **selective retrieval under high confidence**.

---

## 🧩 Final Pipeline

```
Prompt
   ↓
OmniSVG Generation
   ↓
TF-IDF Similarity Matching
   ↓
if similarity ≥ 0.98:
    replace with train SVG
else:
    keep generated SVG
   ↓
Final Submission
```

---

## 📈 Why This Works

The evaluation emphasizes:

* SSIM / Edge-F1
* Structural correctness
* SVG validity

Pure generation struggles with long sequences and geometry.

my method:

* avoids generation errors
* leverages high-confidence matches
* balances precision and coverage

---

## 📁 Repository Structure

```text
test.csv                      
train.csv
final_results/             # final submission outputs

train_svg_style_report/    # SVG data analysis results
  ├── svg_style_report.json
  ├── per_svg_metrics.csv
  ├── tag_distribution.csv
  ├── attr_distribution.csv
  ├── path_command_distribution.csv
  └── stroke_values.csv

analyze_train_svg_style.py # dataset analysis script
prompts.py                 # extract prompts from test.csv
inference.py               # OmniSVG inference script
replace_submission_by_prompt_similarity.py  # retrieval-based replacement

config.py / config.yaml    # model configuration
dataset.py                 # dataset processing
tokenizer.py               # tokenization logic

assets/                    # optional assets (figures, visuals)

README.md                  # project documentation
requirements.txt           # dependencies
Negative Results and Failure Analysis/ #Three Failed Plans
```
## 📦 Model Weights Setup

Before running the pipeline, download the required pretrained models.

---

### 🔹 1. Download OmniSVG (Required)

```bash
pip install -U huggingface_hub

hf download OmniSVG/OmniSVG1.1_4B \
  --local-dir ./models/OmniSVG1.1_4B

hf download Qwen/Qwen2.5-VL-3B-Instruct \
  --local-dir ./models/Qwen2.5-VL-3B-Instruct
```

## ⚡ Quick Start

### 1. Setup Environment

```bash
git clone https://github.com/shuhancai183/DL-Spring-2026-SVG-Generation-from-Text-Prompts
cd https://github.com/shuhancai183/DL-Spring-2026-SVG-Generation-from-Text-Prompts

pip install -r requirements.txt
```

---

### 2. Prepare Data

Make sure the following files exist:

```text
train.csv
test.csv
```

---

### 3. Run Full Pipeline (Recommended)

```bash
python run_all.py
```

---

## 🔁 What run_all.py Does

The script executes the full pipeline:

1. Extract prompts from `test.csv`
2. Run OmniSVG inference to generate SVGs
3. Build `submission.csv`
4. Apply similarity-based replacement (threshold = 0.98)
5. Output final submission

---

## 📂 Final Output

```text
final_results/submission_replace_th_0.98.csv
```

---

## ⚠️ Requirements

Ensure the following files exist before running:

```text
train.csv
test.csv
config.yaml
```

---

## 💡 Notes

* GPU is recommended for faster inference
* The pipeline preserves prompt order automatically
* Intermediate SVG outputs are stored in `output_text/`
* Threshold = 0.98 is selected based on ablation study


---

## 🏁 Final Submission

```
id,svg
```

Example:

```
final_results/submission_replace_th_0.98.csv
```

---

## 📌 Summary

* Baseline: OmniSVG
* Enhancement: Retrieval-based replacement
* Threshold: **0.98**
* Key idea: hybrid generation + retrieval

## ❌ Failure Analysis

### 1. OmniSVG LoRA
### 2. Qwen LoRA
### 3. Stable Diffusion + LoRA

## ❌ Failure Analysis: OmniSVG LoRA Fine-tuning

I explore LoRA-based fine-tuning on OmniSVG using a token-level training pipeline aligned with the official implementation (see `Negative Results and Failure Analysis/train_token_lora_official_style.py` ).

However, this approach does not lead to performance improvements and is ultimately abandoned.

---

### 🔹 Observation

Despite successful training, the LoRA fine-tuned model exhibits:

- unstable SVG structures  
- degraded visual quality  
- no consistent improvement over the baseline  

---

### 🔹 Root Causes

#### 1. Limited parameter update capacity

LoRA updates only a small fraction of parameters (mainly attention projections such as `q_proj`, `k_proj`, `v_proj`, `o_proj`).

👉 As a result:
- I find that the model lacks sufficient capacity to shift the original distribution  
- instead, it introduces local perturbations rather than meaningful adaptation  

> Instead of improving generation, LoRA often **disturbs the learned SVG structure**.

---

#### 2. Insufficient data scale

Although I use a relatively large subset of the training data, it is still insufficient for stable adaptation of a ~4B model.

Additionally:

- SVG sequences are long and highly variable  
- token-level supervision is sparse and noisy  

👉 This leads to:
- weak generalization  
- overfitting or unstable outputs  

---

#### 3. High computational requirement

The official training pipeline relies on:

- token-level SVG encoding  
- long sequence training (up to ~2k tokens)  
- gradient accumulation  

(see tokenization pipeline in `Negative Results and Failure Analysis/test_train_token_pipeline.py` )

👉 In practice, I observe:

- high GPU memory usage  
- very slow training  
- limited feasible batch size  

---

#### 4. Full fine-tuning is recommended but impractical

The official implementation suggests full model training for best results.

However:

- full fine-tuning of a ~4B model is computationally expensive  
- not feasible under my current resource constraints  

👉 Therefore, I rely on LoRA, but it turns out to be insufficient.

---

### 🔹 Conclusion

I conclude that LoRA fine-tuning on OmniSVG is ineffective under my current setup due to:

- limited parameter update capacity  
- insufficient data scale  
- high computational requirements  

> Under constrained resources, direct fine-tuning is less effective than hybrid approaches.

---

### 🔹 Key Insight

This experiment suggests:

> For structured generation tasks like SVG,  
> **small parameter updates (LoRA) are insufficient to correct global structural errors**.

This directly motivates my final approach:

- use OmniSVG for strong baseline generation  
- apply retrieval-based correction only in high-confidence cases  

## ❌ Failure Analysis: Qwen LoRA Fine-tuning

I also explore LoRA fine-tuning on a smaller language model (Qwen 2.5 Coder 3B) for direct SVG generation.

The training pipeline includes:

- custom dataset construction with retrieved examples
- chat-style supervised fine-tuning using TRL
- LoRA-based parameter-efficient adaptation
- inference with retrieval-augmented prompting 

However, this approach fails to produce usable SVG outputs and is ultimately abandoned.

---

### 🔹 Observation

After training, I observe:

- frequent invalid SVG outputs  
- broken syntax and missing structure  
- inability to maintain consistent geometric patterns  

Even when retrieval examples are provided, the model struggles to generate coherent SVGs.

---

### 🔹 Root Causes

#### 1. Limited capacity of LoRA

LoRA only modifies a small subset of parameters:

- attention projections (`q_proj`, `k_proj`, `v_proj`, etc.)
- feedforward layers (`up_proj`, `down_proj`, `gate_proj`)

👉 As a result:

- I find that LoRA cannot significantly shift the base model distribution  
- the model remains biased toward natural language rather than structured SVG output  

> The adaptation is too weak to learn a fundamentally different output format.

---

#### 2. SVG is a highly structured and long sequence format

SVG generation requires:

- strict syntax correctness  
- long token sequences (often >1000 tokens)  
- precise geometric relationships  

👉 In practice:

- Qwen frequently produces malformed SVGs  
- small syntax errors lead to completely invalid outputs  

> Unlike natural language, SVG does not tolerate local mistakes.

---

#### 3. Insufficient data for structured learning

Although I construct a dataset with retrieval augmentation:

- each sample includes target prompt + similar examples 

the effective signal is still limited because:

- SVG structures are highly diverse  
- token-level supervision is sparse  
- model must learn both syntax and geometry simultaneously  

👉 Result:

- poor generalization  
- unstable outputs  

---

#### 4. Attempted structured representation (failed)

To reduce complexity, I experiment with transforming SVG into a simplified intermediate representation (a structured "language" easier to learn).

However:

- LoRA lacks sufficient capacity to teach the model a new representation  
- the model fails to consistently follow the new format  

👉 Result:

- outputs remain inconsistent  
- representation shift is not successfully learned  

> Learning a new structured language requires deeper model adaptation than LoRA can provide.

---

### 🔹 Conclusion

I conclude that LoRA fine-tuning on Qwen is ineffective for this task due to:

- insufficient adaptation capacity  
- high structural complexity of SVG  
- lack of strong inductive bias for geometry  

> Small LLMs struggle to learn precise structured outputs like SVG under limited data and parameter updates.

---

### 🔹 Key Insight

This experiment suggests:

> For structured generation tasks,  
> **model architecture and representation bias matter more than parameter-efficient fine-tuning**.

This further justifies my final design:

- rely on OmniSVG as a specialized generator  
- use retrieval to correct high-confidence cases  

## ❌ Failure Analysis: Stable Diffusion + LoRA

I also explore a diffusion-based approach by training a Stable Diffusion model with LoRA for image generation, followed by implicit SVG alignment.

The pipeline includes:

- converting SVGs into grayscale raster images for training 
- LoRA-based fine-tuning of the UNet in Stable Diffusion 
- prompt-controlled image generation 

However, this approach performs poorly (Private: ~5.24 / Public: ~6.64) and is ultimately discarded.

---

### 🔹 Observation

I observe the following behaviors:

- the model can generate **rough geometric shapes**
- grayscale outputs are relatively stable
- introducing color leads to **highly unstable and noisy outputs**
- generated images are only **approximately correct**, but lack precise alignment

---

### 🔹 Root Causes

#### 1. Color generation instability

The model works reasonably well under grayscale constraints:

- training uses blackened SVG rendering  
- prompts enforce minimal grayscale style  

However, once color is introduced:

- outputs become chaotic and inconsistent  
- color distribution does not match training data  

👉 I observe a clear failure mode:

> grayscale → stable  
> colored → collapses into noisy patterns  

---

#### 2. Weak spatial precision under visual metrics

Stable Diffusion generates images that are:

- semantically correct  
- visually similar at a high level  

But:

- object positions are slightly shifted  
- shapes have incorrect proportions  
- boundaries are blurred  

👉 Under the evaluation metrics (SSIM, Edge-F1):

- even small spatial deviations cause large score drops  

> Diffusion models optimize perceptual realism, not pixel-level alignment.

---

#### 3. LoRA capacity vs training instability trade-off

I experiment with different LoRA training regimes:

- small updates → model remains close to original distribution (poor adaptation)  
- large updates → outputs collapse into noisy, colorful artifacts  

👉 This leads to a dilemma:

- insufficient training → no meaningful adaptation  
- aggressive training → catastrophic degradation  

> LoRA cannot provide stable control over the diffusion model in this setting.

---

#### 4. Representation mismatch (image vs vector)

The task requires:

- precise vector representation (SVG)  
- exact geometric structure  

However, Stable Diffusion produces:

- raster images  
- implicit and approximate shapes  

👉 This mismatch causes:

- loss of structural information  
- difficulty in aligning outputs with SVG-based evaluation  

> Image generation does not directly translate to structured vector generation.

---

### 🔹 Conclusion

I conclude that Stable Diffusion + LoRA is ineffective for this task due to:

- unstable color modeling  
- lack of spatial precision  
- limited controllability under LoRA  
- fundamental mismatch between raster images and vector outputs  

---

### 🔹 Key Insight

This experiment highlights:

> For tasks requiring precise geometric alignment,  
> **diffusion models are not suitable under pixel-level evaluation metrics**.

This further supports my final approach:

- use OmniSVG for structured generation  
- apply retrieval only for high-confidence correction  

## 🧾 Conclusion

In this project, I systematically explore multiple approaches for text-to-SVG generation under realistic constraints, including limited model size and computational resources.

Through extensive experimentation, I find that:

- **Pure retrieval methods** are unreliable due to noisy semantic matching.
- **General-purpose LLMs (e.g., Qwen with LoRA)** struggle to generate valid and structurally consistent SVGs.
- **Diffusion-based approaches** fail to achieve the spatial precision required by pixel-level evaluation metrics.
- **Parameter-efficient fine-tuning (LoRA)** is insufficient to correct global structural errors in this task.

Based on these observations, I design a **hybrid generation-retrieval pipeline**:

- OmniSVG provides a strong baseline for structured SVG generation.
- A high-confidence similarity filter (threshold = 0.98) selectively applies retrieval to correct generation errors.

This approach achieves the best performance on both public and private leaderboards, demonstrating that:

> **Selective retrieval, when applied conservatively, can effectively enhance structured generation systems.**

---

### 🔹 Key Takeaways

- Structured generation tasks (like SVG) require **strict syntax and geometric precision**, which are difficult for general models to learn.
- Model architecture and representation bias are more critical than parameter-efficient fine-tuning.
- Hybrid systems that combine **generation + high-precision correction** are more robust than either approach alone.

---

### 🔹 Future Work

Possible directions for improvement include:

- exploring structured decoders with stronger geometric inductive bias  
- integrating differentiable rendering or geometry-aware loss functions  
- scaling full-model fine-tuning under larger compute budgets  

---

> Overall, this project highlights the importance of aligning model capabilities, data structure, and evaluation metrics when designing effective AI systems.


## 📚 Acknowledgement

This work is built upon the OmniSVG framework.

I acknowledge the authors of OmniSVG for releasing their model and codebase:

- OmniSVG: https://github.com/OmniSVG/OmniSVG

The inference pipeline and model architecture used in this project are adapted from their implementation.
