"""
Evaluation script for fine-tuned models.

Loads a fine-tuned LoRA model and evaluates it on the test set.
Uses the EXACT same prompt format as training to ensure consistency.

The tutor's key insight: fine-tuned models learn a specific format,
changing the prompt even slightly hurts performance.

Usage:
    # Zero-shot (recommended for fine-tuned models - same format as training)
    python scripts/evaluate_finetuned.py --adapter outputs/finetune_toy_regular --n-shots 0

    # Few-shot ICL (experimental - tutor says this doesn't improve fine-tuned models)
    python scripts/evaluate_finetuned.py --adapter outputs/finetune_toy_regular --n-shots 4

    # OOD evaluation
    python scripts/evaluate_finetuned.py --adapter outputs/finetune_toy_regular --ood
"""

import argparse
import json
import re
import sys
from pathlib import Path
from collections import defaultdict

import pandas as pd
import torch
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.loaders import ICLDataLoader

# Fallback system prompt (same as training - used if training_config.json not found)
DEFAULT_SYSTEM_PROMPT = (
    "You are a text-based ECG analyzer. "
    "Count the characters '|' (narrow R-wave peaks) and ':' (wide T-wave peaks). "
    "Sum both to get the total. "
    "Classify as: Brady [4-7], Normal [8-11], Tachy [12-16]."
)


def load_training_config(adapter_path: str) -> dict:
    """Load training configuration saved alongside the model."""
    config_path = Path(adapter_path) / "training_config.json"
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        print(f"   Loaded training config from {config_path}")
        return config
    print(f"   WARNING: training_config.json not found, using default prompt")
    return {"system_prompt": DEFAULT_SYSTEM_PROMPT}


def load_model_and_tokenizer(adapter_path: str, base_model: str = "google/gemma-2b-it"):
    """Load fine-tuned model with LoRA adapter."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    print(f"\nLoading base model: {base_model}...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"   Loading LoRA adapter from: {adapter_path}...")
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    return model, tokenizer


def load_test_data(data_dir: Path, ood: bool = False) -> pd.DataFrame:
    """Load test dataset."""
    if ood:
        test_file = data_dir / "test" / "test_ood_metadata.csv"
    else:
        test_file = data_dir / "test" / "test_metadata.csv"

    if not test_file.exists():
        raise FileNotFoundError(f"Test file not found: {test_file}")

    return pd.read_csv(test_file)


def build_prompt(sequence: str, system_prompt: str, tokenizer,
                 icl_examples: list = None) -> str:
    """
    Build prompt matching the EXACT fine-tuning format.

    For zero-shot (recommended): system_prompt + "Secuencia: {sequence}"
    For few-shot (experimental): adds examples before the test sequence
    """
    if icl_examples and len(icl_examples) > 0:
        # Few-shot: add examples (experimental - tutor says this doesn't help)
        examples_text = ""
        for i, (seq, label) in enumerate(icl_examples, 1):
            narrow = seq.count('|')
            wide = seq.count(':')
            total = narrow + wide
            response = json.dumps({
                "Narrow_peaks": narrow,
                "Wide_peaks": wide,
                "Total_peaks": total,
                "Label": label
            })
            examples_text += f"\nExample {i}:\nSequence: {seq}\nResponse: {response}\n"

        user_content = f"{system_prompt}\n{examples_text}\nSequence: {sequence}"
    else:
        # Zero-shot: exact same format as training
        user_content = f"{system_prompt}\n\nSequence: {sequence}"

    messages = [{"role": "user", "content": user_content}]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    return prompt


def generate_response(model, tokenizer, prompt: str, max_new_tokens: int = 150) -> str:
    """Generate response from the model."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return response.strip()


def parse_prediction(response: str) -> dict:
    """Parse JSON response from fine-tuned model."""
    try:
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group())
            if "Label" in parsed:
                return parsed
    except json.JSONDecodeError:
        pass

    # Fallback: keyword matching
    response_lower = response.lower()
    for label in ["brady", "normal", "tachy"]:
        if label in response_lower:
            return {"Label": label.capitalize()}

    return {"Label": None}


def evaluate(
    adapter_path: str,
    data_dir: Path,
    task: int = 1,
    n_shots: int = 0,
    ood: bool = False,
    base_model: str = "google/gemma-2b-it",
    max_samples: int = None
):
    """Run evaluation on test set."""

    print("=" * 60)
    print("Evaluation of Fine-tuned Model")
    print("=" * 60)
    print(f"Adapter: {adapter_path}")
    print(f"Base model: {base_model}")
    print(f"N-shots: {n_shots}")
    print(f"OOD: {ood}")
    print("=" * 60)

    # Load training config (to use exact same prompt format)
    training_config = load_training_config(adapter_path)
    system_prompt = training_config.get("system_prompt", DEFAULT_SYSTEM_PROMPT)
    print(f"   System prompt: {system_prompt[:60]}...")

    # Load model
    model, tokenizer = load_model_and_tokenizer(adapter_path, base_model)

    # Load test data
    print(f"\nLoading test data...")
    test_df = load_test_data(data_dir, ood=ood)

    if max_samples:
        test_df = test_df.head(max_samples)

    print(f"   Loaded {len(test_df)} test samples")

    # Load ICL examples if n_shots > 0
    icl_examples = None
    if n_shots > 0:
        print(f"\nLoading {n_shots} ICL examples per class...")
        data_loader = ICLDataLoader(data_dir=data_dir)
        icl_examples = data_loader.load_icl_examples(n_shots=n_shots)
        print(f"   Loaded {len(icl_examples)} ICL examples")
        print("   NOTE: Tutor says ICL doesn't improve fine-tuned models")

    # Evaluate
    print(f"\nRunning evaluation...")

    results = []
    correct = 0
    total = 0
    class_correct = defaultdict(int)
    class_total = defaultdict(int)

    for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Evaluating"):
        sequence = row['sequence']
        true_label = row['label']

        # Build prompt (exact same format as training)
        prompt = build_prompt(
            sequence=sequence,
            system_prompt=system_prompt,
            tokenizer=tokenizer,
            icl_examples=icl_examples
        )

        # Generate response
        response = generate_response(model, tokenizer, prompt)

        # Parse prediction
        parsed = parse_prediction(response)
        pred_label = parsed.get("Label")

        # Check correctness
        is_correct = pred_label == true_label if pred_label else False
        
        print(f"Is correct : {is_correct}, Pred : {pred_label}, True: {true_label}")

        if is_correct:
            correct += 1
            class_correct[true_label] += 1

        total += 1
        class_total[true_label] += 1

        results.append({
            'id': row.get('id', idx),
            'sequence': sequence,
            'true_label': true_label,
            'pred_label': pred_label,
            'correct': is_correct,
            'raw_response': response[:200]
        })

    # Calculate metrics
    accuracy = correct / total if total > 0 else 0

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"\nOverall Accuracy: {accuracy:.2%} ({correct}/{total})")

    print("\nPer-class Accuracy:")
    for label in sorted(class_total.keys()):
        class_acc = class_correct[label] / class_total[label] if class_total[label] > 0 else 0
        print(f"   {label}: {class_acc:.2%} ({class_correct[label]}/{class_total[label]})")

    # Save results
    output_dir = project_root / "outputs" / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)

    ood_suffix = "_ood" if ood else ""
    shot_suffix = f"_{n_shots}shot"
    results_file = output_dir / f"finetuned{shot_suffix}_results{ood_suffix}.csv"

    results_df = pd.DataFrame(results)
    results_df.to_csv(results_file, index=False)
    print(f"\nResults saved to: {results_file}")

    # Save summary
    summary = {
        "adapter_path": str(adapter_path),
        "base_model": base_model,
        "task": task,
        "n_shots": n_shots,
        "ood": ood,
        "system_prompt_used": system_prompt,
        "total_samples": total,
        "correct": correct,
        "accuracy": accuracy,
        "per_class": {
            label: {
                "correct": class_correct[label],
                "total": class_total[label],
                "accuracy": class_correct[label] / class_total[label] if class_total[label] > 0 else 0
            }
            for label in class_total.keys()
        }
    }

    summary_file = output_dir / f"finetuned{shot_suffix}_summary{ood_suffix}.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to: {summary_file}")

    return accuracy, results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate fine-tuned model on test set",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Zero-shot evaluation (recommended - exact same format as training)
  python scripts/evaluate_finetuned.py --adapter outputs/finetune_toy_regular --n-shots 0

  # Few-shot ICL (experimental)
  python scripts/evaluate_finetuned.py --adapter outputs/finetune_toy_regular --n-shots 4

  # OOD evaluation
  python scripts/evaluate_finetuned.py --adapter outputs/finetune_toy_regular --ood
        """
    )
    parser.add_argument(
        "--adapter",
        type=str,
        default="outputs/finetune_toy_regular",
        help="Path to LoRA adapter directory"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="google/gemma-2b-it",
        help="Base model name"
    )
    parser.add_argument(
        "--task",
        type=int,
        default=1,
        help="Task number (default: 1)"
    )
    parser.add_argument(
        "--n-shots",
        type=int,
        default=0,
        help="Number of ICL examples per class (0 for zero-shot, recommended for fine-tuned)"
    )
    parser.add_argument(
        "--ood",
        action="store_true",
        help="Use out-of-distribution test set"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate (for quick testing)"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/processed/toy_experiment",
        help="Data directory"
    )

    args = parser.parse_args()

    # Resolve paths
    adapter_path = project_root / args.adapter if not Path(args.adapter).is_absolute() else Path(args.adapter)
    data_dir = project_root / args.data_dir

    # Validate paths
    if not adapter_path.exists():
        print(f"ERROR: Adapter not found: {adapter_path}")
        sys.exit(1)

    if not data_dir.exists():
        print(f"ERROR: Data directory not found: {data_dir}")
        print("Run: python scripts/generate_toy_dataset.py")
        sys.exit(1)

    # Run evaluation
    evaluate(
        adapter_path=str(adapter_path),
        data_dir=data_dir,
        task=args.task,
        n_shots=args.n_shots,
        ood=args.ood,
        base_model=args.base_model,
        max_samples=args.max_samples
    )


if __name__ == "__main__":
    main()
