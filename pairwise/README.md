# DeepfakeJudge — Pairwise Inference

Pairwise evaluation compares two candidate reasoning responses and selects which one better aligns with the ground-truth classification, given the image.

## Setup

```bash
pip install git+https://github.com/huggingface/transformers accelerate
pip install qwen-vl-utils[decord]==0.0.8
```

## Download a Pairwise Model

```bash
# Using huggingface-cli
pip install huggingface_hub
huggingface-cli download MBZUAI/Qwen-2.5-VL-Instruct-7B-Pairwise-DFJ --local-dir ./Qwen-2.5-VL-Instruct-7B-Pairwise-DFJ

# Or using Python
from huggingface_hub import snapshot_download
snapshot_download("MBZUAI/Qwen-2.5-VL-Instruct-7B-Pairwise-DFJ", local_dir="./Qwen-2.5-VL-Instruct-7B-Pairwise-DFJ")
```

See the [Model Zoo](../README.md#model-zoo) for all available pairwise checkpoints.

## Prompt Format

The full prompt template is in [`prompt.txt`](prompt.txt). It expects three placeholders:

| Placeholder | Description |
|---|---|
| `{ground_truth_label}` | `real`, `fake`, or `edited` |
| `{response_a}` | First candidate response (reasoning + answer) |
| `{response_b}` | Second candidate response (reasoning + answer) |

The judge outputs:
```
<answer>A</answer>   or   <answer>B</answer>
```

## Run Inference

### Command Line

```bash
python inference.py \
    --model_path MBZUAI/Qwen-2.5-VL-Instruct-7B-Pairwise-DFJ \
    --image /path/to/image.png \
    --label real \
    --response_a '<reasoning>The lighting appears natural and consistent...</reasoning> <answer>real</answer>' \
    --response_b '<reasoning>The image shows signs of manipulation near the edges...</reasoning> <answer>fake</answer>'
```

Add `--flash_attention` to enable Flash Attention 2 for faster inference.

### Python API

```python
from inference import run_inference

output = run_inference(
    model_path="MBZUAI/Qwen-2.5-VL-Instruct-7B-Pairwise-DFJ",
    image_path="/path/to/image.png",
    ground_truth_label="real",
    response_a='<reasoning>The lighting appears natural...</reasoning> <answer>real</answer>',
    response_b='<reasoning>The image shows signs of manipulation...</reasoning> <answer>fake</answer>',
)
print(output)
# <answer>A</answer>
```

## Output Format

The model returns:

```
<answer>A</answer>
```
or
```
<answer>B</answer>
```

indicating which response is the better-grounded reasoning for the given image and label.
