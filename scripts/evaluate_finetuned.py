"""
Evaluation script for fine-tuned models.

This script loads a fine-tuned LoRA model and evaluates it on the test set
without requiring an external LLM server.

Supports zero-shot and few-shot (ICL) evaluation to test the model's
in-context learning capability after fine-tuning.

Usage:
    # Zero-shot (default for fine-tuned models)
    python scripts/evaluate_finetuned.py --adapter outputs/finetune_toy_regular --n-shots 0
    
    # Few-shot ICL
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

from src.icl.prompt_builder import PromptManager
from src.data.loaders import ICLDataLoader


def load_model_and_tokenizer(adapter_path: str, base_model: str = "google/gemma-2b-it"):
    """Load fine-tuned model with LoRA adapter."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel
    
    print(f"\n Loading base model: {base_model}...")
    
    # Quantization config (same as training)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load LoRA adapter
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


def build_prompt(
    sequence: str, 
    prompt_manager: PromptManager, 
    task: int = 1, 
    approach: str = "regular",
    examples: list = None
) -> str:
    """Build prompt for the model (Gemma format - no system role)."""
    messages = prompt_manager.build_messages(
        sequence=sequence,
        task=task,
        approach=approach,
        examples=examples
    )
    
    # Gemma doesn't support system role - merge into user message
    if len(messages) >= 2 and messages[0]['role'] == 'system':
        system_content = messages[0]['content']
        user_content = messages[1]['content']
        combined_content = f"{system_content}\n\n{user_content}"
        messages = [{"role": "user", "content": combined_content}]
    
    return messages


def generate_response(model, tokenizer, messages: list, max_new_tokens: int = 150) -> str:
    """Generate response from the model."""
    # Apply chat template
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Greedy decoding for evaluation
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode only the new tokens
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return response.strip()


def parse_prediction(response: str) -> dict:
    """Parse JSON response from model."""
    try:
        # Try to find JSON in response
        json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
    except json.JSONDecodeError:
        pass
    
    # Fallback: try to extract label directly
    response_lower = response.lower()
    for label in ["brady", "normal", "tachy"]:
        if label in response_lower:
            return {"Label": label.capitalize()}
    
    return {"Label": None}


def evaluate(
    adapter_path: str,
    data_dir: Path,
    prompts_dir: Path,
    task: int = 1,
    n_shots: int = 0,
    approach: str = "regular",
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
    print(f"Task: {task}")
    print(f"N-shots: {n_shots}")
    print(f"Approach: {approach}")
    print(f"OOD: {ood}")
    print("=" * 60)
    
    # Load model
    model, tokenizer = load_model_and_tokenizer(adapter_path, base_model)
    
    # Load test data
    print(f"\n Loading test data...")
    test_df = load_test_data(data_dir, ood=ood)
    
    if max_samples:
        test_df = test_df.head(max_samples)
    
    print(f"   Loaded {len(test_df)} test samples")
    
    # Initialize data loader for ICL examples
    data_loader = ICLDataLoader(data_dir=data_dir)
    
    # Load ICL examples if n_shots > 0
    icl_examples = None
    if n_shots > 0:
        print(f"\n Loading {n_shots} ICL examples...")
        icl_data = data_loader.load_icl_examples(n_shots=n_shots)
        # Format for prompt manager
        icl_examples = [
            {'sequence': seq, 'fc_class': fc_class}
            for seq, fc_class in icl_data
        ]
        print(f"   Loaded {len(icl_examples)} ICL examples")
    
    # Initialize prompt manager
    prompt_manager = PromptManager(prompts_dir=prompts_dir)
    
    # Evaluate
    print(f"\n Running evaluation...")
    
    results = []
    correct = 0
    total = 0
    class_correct = defaultdict(int)
    class_total = defaultdict(int)
    
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Evaluating"):
        sequence = row['sequence']
        true_label = row['label']
        
        # Build prompt with ICL examples
        messages = build_prompt(
            sequence=sequence,
            prompt_manager=prompt_manager,
            task=task,
            approach=approach,
            examples=icl_examples
        )
        
        # Generate response
        response = generate_response(model, tokenizer, messages)
        
        # Parse prediction
        parsed = parse_prediction(response)
        pred_label = parsed.get("Label")
        
        # Check correctness
        is_correct = pred_label == true_label if pred_label else False
        print(f"Correct: {is_correct}, True Label: {true_label}, Predicted: {pred_label}")
        
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
            'raw_response': response[:200]  # Truncate for storage
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
    shot_suffix = f"_{n_shots}shot" if n_shots > 0 else "_0shot"
    approach_suffix = f"_{approach}" if approach != "regular" else ""
    results_file = output_dir / f"finetuned{shot_suffix}{approach_suffix}_results{ood_suffix}.csv"
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(results_file, index=False)
    print(f"\nResults saved to: {results_file}")
    
    # Save summary
    summary = {
        "adapter_path": str(adapter_path),
        "base_model": base_model,
        "task": task,
        "n_shots": n_shots,
        "approach": approach,
        "ood": ood,
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
    
    summary_file = output_dir / f"finetuned{shot_suffix}{approach_suffix}_summary{ood_suffix}.json"
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
  # Zero-shot evaluation (default for fine-tuned models)
  python scripts/evaluate_finetuned.py --adapter outputs/finetune_toy_regular --n-shots 0
  
  # Few-shot evaluation (test ICL capability of fine-tuned model)
  python scripts/evaluate_finetuned.py --adapter outputs/finetune_toy_regular --n-shots 4
  
  # 8-shot evaluation
  python scripts/evaluate_finetuned.py --adapter outputs/finetune_toy_regular --n-shots 8
  
  # Evaluate on OOD test set
  python scripts/evaluate_finetuned.py --adapter outputs/finetune_toy_regular --ood
  
  # CBM approach with few-shot
  python scripts/evaluate_finetuned.py --adapter outputs/finetune_toy_cbm --approach cbm --n-shots 4
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
        help="Number of ICL examples (0 for zero-shot, typically 1, 2, 4, or 8)"
    )
    parser.add_argument(
        "--approach",
        type=str,
        default="regular",
        choices=["regular", "cbm"],
        help="Approach: 'regular' (direct classification) or 'cbm' (concept bottleneck)"
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
    prompts_dir = project_root / "src" / "prompts"
    
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
        prompts_dir=prompts_dir,
        task=args.task,
        n_shots=args.n_shots,
        approach=args.approach,
        ood=args.ood,
        base_model=args.base_model,
        max_samples=args.max_samples
    )


if __name__ == "__main__":
    main()
