"""
Pointwise inference script for DeepfakeJudge.

Given an image, a ground-truth label, and a candidate model response,
the judge assigns a quality score from 1 (worst) to 5 (best).
"""

import argparse
from pathlib import Path

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

PROMPT_FILE = Path(__file__).parent / "prompt.txt"


def load_prompt_template() -> str:
    with open(PROMPT_FILE) as f:
        return f.read()


def build_messages(
    image_path: str,
    ground_truth_label: str,
    candidate_response: str,
) -> list[dict]:
    template = load_prompt_template()
    user_text = template.format(
        ground_truth_label=ground_truth_label,
        candidate_response=candidate_response,
    )
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": user_text},
            ],
        }
    ]


def run_inference(
    model_path: str,
    image_path: str,
    ground_truth_label: str,
    candidate_response: str,
    max_new_tokens: int = 512,
    use_flash_attention: bool = False,
) -> str:
    load_kwargs = dict(torch_dtype=torch.bfloat16, device_map="auto")
    if use_flash_attention:
        load_kwargs["attn_implementation"] = "flash_attention_2"

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path, **load_kwargs
    )
    processor = AutoProcessor.from_pretrained(model_path)

    messages = build_messages(image_path, ground_truth_label, candidate_response)

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return output_text[0]


def main():
    parser = argparse.ArgumentParser(
        description="DeepfakeJudge — Pointwise Inference"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path or HF repo id of the pointwise judge model",
    )
    parser.add_argument("--image", type=str, required=True, help="Path to the image")
    parser.add_argument(
        "--label",
        type=str,
        required=True,
        choices=["real", "fake", "edited"],
        help="Ground-truth label",
    )
    parser.add_argument(
        "--candidate_response",
        type=str,
        required=True,
        help="Candidate response to evaluate",
    )
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument(
        "--flash_attention", action="store_true", help="Enable Flash Attention 2"
    )

    args = parser.parse_args()

    output = run_inference(
        model_path=args.model_path,
        image_path=args.image,
        ground_truth_label=args.label,
        candidate_response=args.candidate_response,
        max_new_tokens=args.max_new_tokens,
        use_flash_attention=args.flash_attention,
    )
    print(output)


if __name__ == "__main__":
    main()
