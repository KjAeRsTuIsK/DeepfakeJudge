# DeepfakeJudge — Pointwise Inference

Pointwise evaluation assigns a quality score (1–5) to a single candidate reasoning response, given an image and its ground-truth label.

## Setup

```bash
pip install git+https://github.com/huggingface/transformers accelerate
pip install qwen-vl-utils[decord]==0.0.8
```

## Download a Pointwise Model

```bash
# Using huggingface-cli
pip install huggingface_hub
huggingface-cli download MBZUAI/Qwen-2.5-VL-Instruct-7B-Pointwise-DFJ --local-dir ./Qwen-2.5-VL-Instruct-7B-Pointwise-DFJ

# Or using Python
from huggingface_hub import snapshot_download
snapshot_download("MBZUAI/Qwen-2.5-VL-Instruct-7B-Pointwise-DFJ", local_dir="./Qwen-2.5-VL-Instruct-7B-Pointwise-DFJ")
```

See the [Model Zoo](../README.md#model-zoo) for all available pointwise checkpoints.

## Prompt Format

The full prompt template is in [`prompt.txt`](prompt.txt). It expects three placeholders:

| Placeholder | Description |
|---|---|
| `{ground_truth_label}` | `real`, `fake`, or `edited` |
| `{candidate_response}` | The model response to evaluate (reasoning + answer) |

The judge outputs:
```
<reasoning>{explanation}</reasoning>
<score>{1-5}</score>
```

## Run Inference

### Command Line

```bash
python inference.py \
    --model_path MBZUAI/Qwen-2.5-VL-Instruct-7B-Pointwise-DFJ \
    --image /path/to/image.png \
    --label real \
    --candidate_response '<reasoning>The lighting appears natural and consistent across the face...</reasoning> <answer>real</answer>'
```

Add `--flash_attention` to enable Flash Attention 2 for faster inference.

### Python API

```python
from inference import run_inference

output = run_inference(
    model_path="MBZUAI/Qwen-2.5-VL-Instruct-7B-Pointwise-DFJ",
    image_path="/path/to/image.png",
    ground_truth_label="real",
    candidate_response='<reasoning>The lighting appears natural...</reasoning> <answer>real</answer>',
)
print(output)
# <reasoning>Fully accurate, complete, and well-grounded...</reasoning>
# <score>5</score>
```

## Output Format

The model returns a structured response:

```
<reasoning>Brief rationale for the assigned score.</reasoning>
<score>N</score>
```

Where `N` is an integer from 1 (worst) to 5 (best).

### Score Interpretation

| Score | Meaning |
|---|---|
| **5** | Fully accurate, complete, and well-grounded |
| **4** | Mostly accurate with minor issues |
| **3** | Partially correct with noticeable errors or omissions |
| **2** | Poor alignment with serious flaws |
| **1** | Unrelated or incorrect |
