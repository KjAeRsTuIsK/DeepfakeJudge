# Pixels Don't Lie (But Your Detector Might): Bootstrapping MLLM-as-a-Judge for Trustworthy Deepfake Detection and Reasoning Supervision **[CVPR-2026]** <p align="center"> <img src="images/judge_logo (1).png" height="150" style="border: none; outline: none; display: block;" ></p>

<p align="center">
    <img src="https://i.imgur.com/waxVImv.png" alt="Oryx Video-ChatGPT">
</p>

#### [Kartik Kuckreja](https://www.linkedin.com/in/kartik-kuckreja-930531221/), [Parul Gupta](https://scholar.google.com.au/citations?user=Wik3mXsAAAAJ&hl=en), [Muhammad Haris Khan](https://m-haris-khan.com/) and [ Abhinav Dhall](https://research.monash.edu/en/persons/abhinav-dhall)

#### **Mohamed bin Zayed University of AI, Monash University**

---

[![Website](https://img.shields.io/badge/Project-Website-87CEEB)](https://kjaerstuisk.github.io/DeepfakeJudge/)
[![paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)]()
[![Dataset](https://img.shields.io/badge/🤗_Dataset-DeepfakeJudge-blue)](https://huggingface.co/datasets/MBZUAI/DeepfakeJudge-Dataset)

---

# 📑 Contents

- [🔍 Overview](#-overview)
- [⚙️ Install](#%EF%B8%8F-install)
- [🦁 Model Zoo](#-model-zoo)
- [📂 Dataset](#-dataset)
  - [Download the Dataset](#download-the-dataset)
  - [Dataset Structure](#dataset-structure)
- [🚀 Inference](#-inference)
  - [Pointwise Inference](#pointwise-inference)
  - [Pairwise Inference](#pairwise-inference)
- [🏋️ Training](#%EF%B8%8F-training)
  - [Setup](#training-setup)
  - [Pointwise Training](#pointwise-training)
  - [Pairwise Training](#pairwise-training)
  - [Configuration Reference](#configuration-reference)
- [💡 Contributions](#-contributions)
- [📊 Datasets](#-datasets)
  - [DeepfakeJudge-Detect](#deepfakejudge-detect)
  - [DeepfakeJudge-Reason](#deepfakejudge-reason)
  - [DeepfakeJudge-Meta](#deepfakejudge-meta)
  - [DeepfakeJudge-Meta-Human](#deepfakejudge-meta-human)
- [🔬 Methodology](#-methodology)
- [📈 Benchmark Results](#-benchmark-results)
  - [Deepfake Detection (OOD)](#deepfake-detection-ood)
  - [Reasoning Evaluation](#reasoning-evaluation)
  - [Pointwise Evaluation](#pointwise-evaluation)
  - [Pairwise Evaluation](#pairwise-evaluation)
- [👥 User Study](#-user-study)
- [🏁 Conclusion](#-conclusion)
- [📝 Citation](#-citation)

---

# 🔍 Overview

Deepfake detection models increasingly generate natural language explanations to justify their predictions. However, while classification accuracy has improved, the reasoning itself is often ungrounded, hallucinated, or loosely connected to the actual visual evidence. Existing evaluation protocols primarily measure detection accuracy and overlook reasoning fidelity, visual grounding, and interpretability.

This repository introduces **DeepfakeJudge**, a unified framework for scalable reasoning supervision and evaluation in deepfake detection. The framework integrates an out-of-distribution detection benchmark, a densely human-annotated reasoning dataset, and a bootstrapped generator–evaluator training pipeline to build a multimodal reasoning judge. The resulting models evaluate explanation quality directly from images and support both pointwise and pairwise assessment aligned with human judgment.

DeepfakeJudge establishes reasoning fidelity as a measurable dimension of trustworthy deepfake detection and demonstrates that scalable supervision of reasoning evaluators is possible without requiring explicit ground-truth rationales for every instance.

---

# ⚙️ Install

Clone the repository and install dependencies:

```bash
git clone https://github.com/MBZUAI/DeepfakeJudge.git
cd DeepfakeJudge
```

Install the required packages for inference:

```bash
# Qwen2.5-VL requires the latest transformers — build from source
pip install git+https://github.com/huggingface/transformers accelerate

# Vision utilities (decord recommended for faster video loading)
pip install qwen-vl-utils[decord]==0.0.8

# If you cannot install decord (non-Linux), fall back to:
# pip install qwen-vl-utils
```

> **Note:** If you encounter `KeyError: 'qwen2_5_vl'`, make sure you installed transformers from source as shown above.

---

# 🦁 Model Zoo

All DeepfakeJudge models are fine-tuned from Qwen2.5-VL-Instruct using LoRA and are hosted on Hugging Face under [MBZUAI](https://huggingface.co/MBZUAI).

| Model | Type | Base Model | HuggingFace |
|---|---|---|---|
| DeepfakeJudge-3B-Pointwise | Pointwise | Qwen2.5-VL-3B-Instruct | [MBZUAI/Qwen-2.5-VL-Instruct-3B-Pointwise-DFJ](https://huggingface.co/MBZUAI/Qwen-2.5-VL-Instruct-3B-Pointwise-DFJ) |
| DeepfakeJudge-3B-Pairwise | Pairwise | Qwen2.5-VL-3B-Instruct | [MBZUAI/Qwen-2.5-VL-Instruct-3B-Pairwise-DFJ](https://huggingface.co/MBZUAI/Qwen-2.5-VL-Instruct-3B-Pairwise-DFJ) |
| DeepfakeJudge-7B-Pointwise | Pointwise | Qwen2.5-VL-7B-Instruct | [MBZUAI/Qwen-2.5-VL-Instruct-7B-Pointwise-DFJ](https://huggingface.co/MBZUAI/Qwen-2.5-VL-Instruct-7B-Pointwise-DFJ) |
| DeepfakeJudge-7B-Pairwise | Pairwise | Qwen2.5-VL-7B-Instruct | [MBZUAI/Qwen-2.5-VL-Instruct-7B-Pairwise-DFJ](https://huggingface.co/MBZUAI/Qwen-2.5-VL-Instruct-7B-Pairwise-DFJ) |

### ⬇️ Download Models

**Option 1: Hugging Face CLI**

```bash
pip install huggingface_hub

# Download a specific model (e.g., 7B Pointwise)
huggingface-cli download MBZUAI/Qwen-2.5-VL-Instruct-7B-Pointwise-DFJ \
    --local-dir ./models/Qwen-2.5-VL-Instruct-7B-Pointwise-DFJ
```

**Option 2: Python**

```python
from huggingface_hub import snapshot_download

snapshot_download(
    "MBZUAI/Qwen-2.5-VL-Instruct-7B-Pointwise-DFJ",
    local_dir="./models/Qwen-2.5-VL-Instruct-7B-Pointwise-DFJ"
)
```

**Option 3: Git LFS**

```bash
git lfs install
git clone https://huggingface.co/MBZUAI/Qwen-2.5-VL-Instruct-7B-Pointwise-DFJ
```

---

# 📂 Dataset

The DeepfakeJudge Dataset is hosted on Hugging Face:
**[MBZUAI/DeepfakeJudge-Dataset](https://huggingface.co/datasets/MBZUAI/DeepfakeJudge-Dataset)**

## ⬇️ Download the Dataset

**Option 1: Hugging Face CLI**

```bash
huggingface-cli download MBZUAI/DeepfakeJudge-Dataset \
    --repo-type dataset \
    --local-dir ./DeepfakeJudge-Dataset
```

**Option 2: Python**

```python
from huggingface_hub import snapshot_download

snapshot_download(
    "MBZUAI/DeepfakeJudge-Dataset",
    repo_type="dataset",
    local_dir="./DeepfakeJudge-Dataset"
)
```

**Option 3: Git LFS**

```bash
git lfs install
git clone https://huggingface.co/datasets/MBZUAI/DeepfakeJudge-Dataset
```

## 🗂️ Dataset Structure

```
DeepfakeJudge-Dataset/
├── dfj-bench/
│   ├── dfj-detect/        # 2,000 images — real/fake detection benchmark
│   └── dfj-reason/        # 924 images — reasoning ground-truth benchmark
├── dfj-meta/
│   ├── dfj-meta-pointwise/
│   │   ├── train/         # 20,625 records (825 images) — pointwise training
│   │   └── test/          # 1,000 records (199 images) — pointwise test
│   └── dfj-meta-pairwise/
│       ├── train/         # 20,625 records (825 images) — pairwise training
│       └── test/          # 2,000 records (200 images) — pairwise test
└── dfj-meta-human/
    ├── pointwise/         # 67 records (58 images) — human-annotated pointwise
    └── pairwise/          # 88 records (70 images) — human-annotated pairwise
```

Each subset contains an `images/` folder and a `data.jsonl` file. Image paths in the JSONL are relative to the split directory. See the [dataset README](https://huggingface.co/datasets/MBZUAI/DeepfakeJudge-Dataset) for the full schema.

---

# 🚀 Inference

## 📌 Pointwise Inference

Pointwise evaluation assigns a quality score (1–5) to a single candidate reasoning response.

```python
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch

model_path = "MBZUAI/Qwen-2.5-VL-Instruct-7B-Pointwise-DFJ"
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path, torch_dtype=torch.bfloat16, device_map="auto"
)
processor = AutoProcessor.from_pretrained(model_path)

# Read the pointwise prompt template
with open("pointwise/prompt.txt") as f:
    prompt_template = f.read()

# Fill in the placeholders
user_prompt = prompt_template.format(
    ground_truth_label="real",
    candidate_response=(
        "<reasoning>The lighting casts soft shadows around the nose "
        "and under the lower lip consistent with a frontal source. "
        "Facial features such as wrinkles, pores, and beard stubble "
        "have fine texture and depth.</reasoning> <answer>real</answer>"
    ),
)

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "/path/to/image.png"},
            {"type": "text", "text": user_prompt},
        ],
    }
]

text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text], images=image_inputs, videos=video_inputs,
    padding=True, return_tensors="pt",
).to("cuda")

generated_ids = model.generate(**inputs, max_new_tokens=512)
generated_ids_trimmed = [
    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
print(output[0])
# <reasoning>Fully accurate, complete, and well-grounded...</reasoning>
# <score>5</score>
```

A ready-to-use CLI script is available in [`pointwise/inference.py`](pointwise/inference.py) — see the [pointwise README](pointwise/README.md) for full details.

## ⚖️ Pairwise Inference

Pairwise evaluation compares two candidate responses and selects which one is better-grounded.

```python
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch

model_path = "MBZUAI/Qwen-2.5-VL-Instruct-7B-Pairwise-DFJ"
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path, torch_dtype=torch.bfloat16, device_map="auto"
)
processor = AutoProcessor.from_pretrained(model_path)

# Read the pairwise prompt template
with open("pairwise/prompt.txt") as f:
    prompt_template = f.read()

# Fill in the placeholders
user_prompt = prompt_template.format(
    ground_truth_label="real",
    response_a=(
        "<reasoning>The lighting appears natural and consistent across "
        "the face with realistic shadows.</reasoning> <answer>real</answer>"
    ),
    response_b=(
        "<reasoning>The image shows signs of manipulation near the edges "
        "with unnatural blending artifacts.</reasoning> <answer>fake</answer>"
    ),
)

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "/path/to/image.png"},
            {"type": "text", "text": user_prompt},
        ],
    }
]

text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text], images=image_inputs, videos=video_inputs,
    padding=True, return_tensors="pt",
).to("cuda")

generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
print(output[0])
# <answer>A</answer>
```

A ready-to-use CLI script is available in [`pairwise/inference.py`](pairwise/inference.py) — see the [pairwise README](pairwise/README.md) for full details.

---

# 🏋️ 

DeepfakeJudge models are fine-tuned using [ms-swift](https://github.com/modelscope/ms-swift), a scalable training framework for LLMs and VLMs.

## 🛠️ Training Setup

```bash
pip install ms-swift
```

Set up the environment according to the instructions [here](https://github.com/modelscope/ms-swift).

Make sure you have the dataset downloaded (see [Dataset](#dataset) section above). The training JSONL files are:

- **Pointwise:** `DeepfakeJudge-Dataset/dfj-meta/dfj-meta-pointwise/train/data.jsonl`
- **Pairwise:** `DeepfakeJudge-Dataset/dfj-meta/dfj-meta-pairwise/train/data.jsonl`

## 📌 Pointwise Training

```bash
cd training
bash train_pointwise.sh
```

Before running, edit `train_pointwise.sh` and set:

```bash
MODEL="Qwen/Qwen2.5-VL-7B-Instruct"       # or Qwen/Qwen2.5-VL-3B-Instruct
DATASET="/path/to/dfj-meta-pointwise/train/data.jsonl"
OUTPUT_DIR="./output/pointwise_7b"
NUM_GPUS=2
GPU_IDS="0,1"
```

<details>
<summary>Full training script</summary>

```bash
#!/bin/bash

MODEL="Qwen/Qwen2.5-VL-7B-Instruct"
DATASET="/path/to/dfj-meta-pointwise/train/data.jsonl"
OUTPUT_DIR="./output/pointwise_7b"
NUM_GPUS=2
GPU_IDS="0,1"

export MAX_PIXELS=1003520
export IMAGE_FACTOR=28
export MIN_PIXELS=3136

CUDA_VISIBLE_DEVICES=${GPU_IDS} \
NPROC_PER_NODE=${NUM_GPUS} \
swift sft \
    --model ${MODEL} \
    --use_hf true \
    --dataset ${DATASET} \
    --train_type lora \
    --torch_dtype bfloat16 \
    --num_train_epochs 2 \
    --per_device_train_batch_size 16 \
    --learning_rate 1e-6 \
    --lora_rank 32 \
    --lora_alpha 64 \
    --target_modules all-linear \
    --freeze_vit true \
    --gradient_accumulation_steps 1 \
    --save_strategy epoch \
    --save_total_limit 5 \
    --logging_steps 1 \
    --max_length 4096 \
    --output_dir ${OUTPUT_DIR} \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 8 \
    --dataset_num_proc 4 \
    --bf16 true \
    --report_to wandb
```

</details>

## ⚖️ Pairwise Training

```bash
cd training
bash train_pairwise.sh
```

Before running, edit `train_pairwise.sh` and set:

```bash
MODEL="Qwen/Qwen2.5-VL-7B-Instruct"       # or Qwen/Qwen2.5-VL-3B-Instruct
DATASET="/path/to/dfj-meta-pairwise/train/data.jsonl"
OUTPUT_DIR="./output/pairwise_7b"
NUM_GPUS=2
GPU_IDS="0,1"
```

> **Key difference:** Pairwise training uses `--max_length 2048` (vs. 4096 for pointwise) since pairwise outputs are shorter (`<answer>A</answer>` or `<answer>B</answer>`).

<details>
<summary>Full training script</summary>

```bash
#!/bin/bash

MODEL="Qwen/Qwen2.5-VL-7B-Instruct"
DATASET="/path/to/dfj-meta-pairwise/train/data.jsonl"
OUTPUT_DIR="./output/pairwise_7b"
NUM_GPUS=2
GPU_IDS="0,1"

export MAX_PIXELS=1003520
export IMAGE_FACTOR=28
export MIN_PIXELS=3136

CUDA_VISIBLE_DEVICES=${GPU_IDS} \
NPROC_PER_NODE=${NUM_GPUS} \
swift sft \
    --model ${MODEL} \
    --use_hf true \
    --dataset ${DATASET} \
    --train_type lora \
    --torch_dtype bfloat16 \
    --num_train_epochs 2 \
    --per_device_train_batch_size 16 \
    --learning_rate 1e-6 \
    --lora_rank 32 \
    --lora_alpha 64 \
    --target_modules all-linear \
    --freeze_vit true \
    --gradient_accumulation_steps 1 \
    --save_strategy epoch \
    --save_total_limit 5 \
    --logging_steps 1 \
    --max_length 2048 \
    --output_dir ${OUTPUT_DIR} \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 8 \
    --dataset_num_proc 4 \
    --bf16 true \
    --report_to wandb
```

</details>

## 📋 Configuration Reference

| Parameter | Pointwise | Pairwise | Notes |
|---|---|---|---|
| `--model` | `Qwen/Qwen2.5-VL-{3B,7B}-Instruct` | Same | Base model from HuggingFace |
| `--dataset` | Pointwise JSONL | Pairwise JSONL | Path to training data |
| `--train_type` | `lora` | `lora` | LoRA fine-tuning |
| `--lora_rank` | 32 | 32 | LoRA rank |
| `--lora_alpha` | 64 | 64 | LoRA scaling factor |
| `--max_length` | 4096 | 2048 | Pointwise needs longer context |
| `--num_train_epochs` | 2 | 2 | Training epochs |
| `--learning_rate` | 1e-6 | 1e-6 | Learning rate |
| `--per_device_train_batch_size` | 16 | 16 | Per-GPU batch size |
| `--freeze_vit` | true | true | Freeze vision encoder |

---

# 💡 Contributions

DeepfakeJudge advances deepfake detection and multimodal reasoning evaluation through several key contributions that jointly address generalization, interpretability, and scalable supervision:

- **Out-of-Distribution Deepfake Benchmark**  
  We construct a challenging benchmark that combines real images, text-to-image generations, and editing-based forgeries. This setup evaluates both detection performance and reasoning generalization under modern and unseen generative pipelines. The benchmark includes both generative and image-editing forgeries to reflect realistic threat scenarios.

- **Human-Annotated Visual Reasoning Dataset**  
  We introduce a densely annotated reasoning dataset in which textual explanations are explicitly linked to localized visual evidence. Each fake image includes artifact category flags, bounding boxes marking manipulated regions, referring expressions, and structured explanatory descriptions. This enables fine-grained supervision of reasoning fidelity rather than relying solely on classification labels.

- **Bootstrapped Generator–Evaluator Supervision Framework**  
  We propose a scalable pipeline that transforms high-quality human reasoning into structured, graded supervision. A generator produces reasoning traces across multiple quality levels, while an evaluator model scores and provides feedback. Misaligned samples are iteratively refined until rating consistency is achieved. Accepted responses are paraphrased to introduce stylistic diversity while preserving semantic meaning.

- **Multimodal Reasoning Judge (MLLM-as-a-Judge)**  
  We train compact Vision-Language Models (3B and 7B) to function as reasoning evaluators. These models support:
  - Pointwise scoring, where a single reasoning trace is assigned a quality score and short evaluator rationale.
  - Pairwise comparison, where two reasoning traces are compared to determine which is more faithful and grounded.
  - Human-aligned reasoning assessment directly conditioned on image evidence.

- **Strong Human Alignment and Efficiency**  
  DeepfakeJudge-7B achieves near-human correlation in reasoning assessment and reaches 96.2% pairwise accuracy and 98.9% agreement on the human-validated subset. Notably, these results surpass models more than 30× larger, demonstrating that compact, specialized reasoning judges can outperform significantly larger general-purpose systems.

---

# 📊 Datasets

## 🎯 DeepfakeJudge-Detect

DeepfakeJudge-Detect is an out-of-distribution benchmark designed to evaluate real-versus-fake classification under modern generation pipelines.

### Real Images

- 1,000 real images sampled from OpenImages-V7.
- Label diversity ensured through a stochastic greedy set-cover algorithm.
- Bounding boxes and verified annotations included to preserve object-level consistency.

### Fake Images

Two types of synthetic images are included to reflect diverse manipulation strategies:

1. **Text-to-Image (T2I)**
   - 500 curated fake images.
   - Realistic, photography-oriented prompts filtered for linguistic and semantic consistency.
   - Generated using state-of-the-art models such as Gemini and SeedDream.

2. **Text+Image-to-Image (Editing)**
   - 500 edited images.
   - Derived from 800 real images.
   - Edited using Gemini, Flux-Kontext-Max, and Qwen-Edit.
   - Edit instructions generated from image captions and applied independently.

**Total dataset size:** 2,000 images (1,000 real + 1,000 fake).

---

## 🧠 DeepfakeJudge-Reason

DeepfakeJudge-Reason provides human-annotated reasoning supervision for detection.

### Composition

- 500 real images.
- 424 fake images.
- Subset sampled from DeepfakeJudge-Detect.

### Annotation Protocol

For each fake image, annotators:

- Select relevant visual artifact categories.
- Draw bounding boxes around anomalous regions.
- Provide referring expressions describing localized inconsistencies.
- Write concise explanatory descriptions.
- Generate structured gold reasoning rationales derived from annotations.

### Annotation Quality

- Six trained annotators.
- Shared pilot calibration phase.
- Cohen's κ = 0.71, indicating substantial inter-annotator agreement.

---

## ⚡ DeepfakeJudge-Meta

DeepfakeJudge-Meta is a bootstrapped reasoning supervision dataset constructed using the generator–evaluator framework.

For each image–label pair:

- Five graded reasoning levels (1–5).
- Controlled degradation of reasoning quality.
- Multiple paraphrased variants to prevent stylistic memorization.

### Dataset Size

- 20,625 training samples for pointwise evaluation.
- 41,250 training samples for pairwise evaluation.

This dataset enables scalable training of reasoning evaluators without requiring explicit human-written rationales at every scale.

---

## 👤 DeepfakeJudge-Meta-Human

A human-validated evaluation subset used to measure alignment between model predictions and expert reasoning judgments.

### Agreement Statistics

- Raw agreement: 0.90.
- Cohen's κ ≈ 0.80 (pairwise evaluation).
- Mean Squared Error (pointwise evaluation): 0.39.

These statistics confirm strong consistency in human reasoning supervision.

---

# 🔬 Methodology

DeepfakeJudge consists of three primary stages:

## 1. Dataset Construction

Real and synthetic images are curated to build an out-of-distribution detection benchmark. Fake images are generated via both text-to-image and editing pipelines. A subset is densely annotated for reasoning supervision, linking textual explanations to spatial visual evidence.

## 2. Bootstrapped Reasoning Supervision

A generator model produces reasoning samples across five intended quality levels. An evaluator model assigns predicted ratings and provides feedback. If the predicted rating deviates from the intended level beyond a threshold, the reasoning is refined using evaluator feedback. Accepted samples are paraphrased multiple times to introduce stylistic diversity while preserving semantic structure. This process produces a large graded corpus for training reasoning judges.

## 3. DeepfakeJudge Training

Two Vision-Language Models (3B and 7B) are trained using a negative log-likelihood objective:

- **Pointwise setting:** The model predicts a reasoning quality score (1–5) and a short justification.
- **Pairwise setting:** The model selects the stronger reasoning between two candidates.

Training uses 20,625 samples for pointwise and 20,625 sampled pairs for pairwise learning.

---

# 📈 Benchmark Results

## 🎯 Deepfake Detection (OOD)

Evaluation on DeepfakeJudge-Detect:

| Model | Real F1 | Fake F1 | Overall Accuracy |
|-------|---------|---------|-----------------|
| Gemini-2.5-Flash | 73.7 | 50.0 | 65.5 |
| GPT-4o-mini | 70.2 | 35.8 | 59.3 |
| Qwen-3-VL-235B | 78.6 | 68.4 | 74.5 |
| Qwen-3-VL-235B-Thinking | 76.6 | 79.8 | 63.7 |
| SIDA-13B | 57.0 | 34.5 | 48.1 |

Closed-source models perform strongly on real images but struggle to generalize to fake samples. Larger open-source VLMs rival or surpass some closed models. Specialized deepfake detectors fail to generalize to modern generation pipelines.

---

## 🧠 Reasoning Evaluation

Evaluation on DeepfakeJudge-Reason:

| Model | BLEU-3 | BERTScore | DFJ-3B Score |
|-------|--------|------------|--------------|
| Gemini-2.5-Flash | 0.02 | 0.60 | 3.17 |
| GPT-4o-mini | 0.01 | 0.35 | 2.83 |
| Qwen-3-VL-30B | 0.03 | 0.62 | 3.31 |
| Qwen-3-VL-235B | 0.01 | 0.60 | 3.59 |
| SIDA | 0.01 | 0.58 | 2.32 |

Traditional lexical metrics such as BLEU and ROUGE fail to reflect visual grounding and factual correctness. DeepfakeJudge scores correlate more consistently with reasoning fidelity.

---

## 📌 Pointwise Evaluation

DeepfakeJudge-Meta results:

| Model | RMSE ↓ | Pearson ↑ |
|-------|--------|-----------|
| Gemini-2.5 | 1.09 | 0.83 |
| GPT-4o-mini | 0.78 | 0.87 |
| Qwen-3-VL-235B | 1.10 | 0.82 |
| DeepfakeJudge-3B | 0.69 | 0.92 |
| DeepfakeJudge-7B | 0.61 | 0.93 |

DeepfakeJudge-Meta-Human:

| Model | RMSE ↓ | Pearson ↑ |
|-------|--------|-----------|
| GPT-4o-mini | 0.81 | 0.86 |
| Qwen-235B-Thinking | 0.95 | 0.86 |
| DeepfakeJudge-7B | 0.50 | 0.95 |

---

## ⚖️ Pairwise Evaluation

Pairwise accuracy (% agreement with human preferences):

| Model | DFJ-Meta | DFJ-Meta-Human |
|-------|----------|----------------|
| Gemini-2.5 | 91.7 | 94.2 |
| GPT-4o-mini | 90.3 | 89.8 |
| Qwen-235B | 93.2 | 99.4 |
| DeepfakeJudge-3B | 94.4 | 96.6 |
| DeepfakeJudge-7B | 96.2 | 98.9 |

---

# 🏁 Conclusion

DeepfakeJudge introduces a unified framework for reasoning supervision and evaluation in deepfake detection. By combining human annotation, bootstrapped multimodal supervision, and automated evaluation, the framework establishes reasoning fidelity as a measurable and scalable objective. Compact reasoning judges trained under this framework achieve near-human alignment and outperform substantially larger models, paving the way for trustworthy, interpretable, and generalizable forensic systems.
