"""
Fine-tuning gemma-2b-it for Task 1 (Regular classification) on RTX 3060.

Uses conversational messages format as recommended by tutor:
- Concise system prompt merged into user (Gemma doesn't support system role)
- Structured JSON response with Conceptos and Label
- Loss computed only on completion (assistant) tokens

Usage:
    python scripts/finetune_toy_3060.py --config configs/medgemma_finetune.yaml
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch
import yaml
from datasets import Dataset

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Fine-tuning system prompt (concise, following tutor's recommended format)
FINETUNE_SYSTEM_PROMPT = (
    "You are a text-based ECG analyzer. "
    "Count the characters '|' (narrow R-wave peaks) and ':' (wide T-wave peaks). "
    "Sum both to get the total. "
    "Classify as: Brady [4-7], Normal [8-11], Tachy [12-16]."
)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def create_prompt_completion_dataset(df, tokenizer):
    """
    Create a prompt-completion dataset for SFTTrainer using tutor's format.

    Format:
    - System prompt merged into user message (Gemma doesn't support system role)
    - User: "{system_prompt}\n\nSequence: {sequence}"
    - Completion: JSON with Narrow_peaks, Wide_peaks, Total_peaks and Label

    SFTTrainer computes loss ONLY on completion tokens, preventing class collapse.
    """
    prompts = []
    completions = []

    for _, row in df.iterrows():
        sequence = row['sequence']
        label = row['label']

        # Merge system prompt into user message (Gemma doesn't support system role)
        user_content = f"{FINETUNE_SYSTEM_PROMPT}\n\nSequence: {sequence}"
        messages = [{"role": "user", "content": user_content}]

        # Apply chat template for prompt (adds <start_of_turn>model\n)
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Build response in tutor's format
        narrow_peaks = sequence.count('|')
        wide_peaks = sequence.count(':')
        total_peaks = narrow_peaks + wide_peaks

        completion = json.dumps({
            "Narrow_peaks": narrow_peaks,
            "Wide_peaks": wide_peaks,
            "Total_peaks": total_peaks,
            "Label": label
        })

        prompts.append(prompt)
        completions.append(completion)

    return Dataset.from_dict({"prompt": prompts, "completion": completions})


def main(config_path: str):
    """Main fine-tuning function."""

    print("=" * 60)
    print("Fine-tuning gemma-2b-it for Task 1 (Regular)")
    print("=" * 60)

    # Load configuration
    print(f"\nLoading configuration from {config_path}...")
    config = load_config(config_path)

    # Lazy imports (heavy libraries)
    print("\nImporting libraries...")
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, EarlyStoppingCallback
    from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
    from trl import SFTTrainer, SFTConfig

    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"   Device: {device}")

    if device == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print(f"\n   WARNING: CUDA not available! Training will be VERY slow on CPU.")
        print(f"   PyTorch version: {torch.__version__}")
        print(f"   CUDA built with: {torch.version.cuda if hasattr(torch.version, 'cuda') else 'N/A'}")
        print(f"\n   To fix this:")
        print(f"   1. Check NVIDIA drivers: nvidia-smi")
        print(f"   2. Reinstall PyTorch with CUDA:")
        print(f"      pip install torch --index-url https://download.pytorch.org/whl/cu121")
        print(f"\n   Press Ctrl+C to abort or wait 10 seconds to continue with CPU...")

        import time
        try:
            time.sleep(10)
        except KeyboardInterrupt:
            print(f"\n   Aborted by user.")
            sys.exit(1)

    # Configure quantization
    print("\nConfiguring quantization...")
    quant_config = config['quantization']

    # Force float16 for RTX 3060 (no native bfloat16 support)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=quant_config['load_in_4bit'],
        bnb_4bit_quant_type=quant_config['bnb_4bit_quant_type'],
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=quant_config.get('bnb_4bit_use_double_quant', False),
    )

    # Load model
    model_name = config['model']['name']
    print(f"\nLoading model: {model_name}...")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
        attn_implementation="eager",
    )

    # Load tokenizer
    print("   Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "right"

    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    # Prepare model for k-bit training
    print("\nPreparing model for QLoRA training...")
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    # Ensure all trainable parameters are in float32 for stable training
    for param in model.parameters():
        if param.requires_grad:
            param.data = param.data.float()

    # Configure LoRA
    lora_config = config['lora']
    peft_config = LoraConfig(
        r=lora_config['r'],
        lora_alpha=lora_config['lora_alpha'],
        lora_dropout=lora_config['lora_dropout'],
        target_modules=lora_config['target_modules'],
        task_type=lora_config['task_type'],
        bias=lora_config.get('bias', 'none'),
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Load datasets
    data_config = config['data']
    data_dir = project_root / Path(data_config['train_path']).parent.parent
    train_path = project_root / data_config['train_path']
    print(f"\nLoading training data from {train_path}...")

    if not train_path.exists():
        print(f"   Training data not found!")
        print(f"   Run: python scripts/generate_toy_dataset.py")
        sys.exit(1)

    # Load training config
    train_config = config['training']

    # Load and shuffle raw data
    train_df = pd.read_csv(train_path)
    train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"   Shuffled {len(train_df)} training samples")

    # Create prompt-completion dataset with tutor's conversational format
    print("   Creating prompt-completion dataset (loss only on completions)...")
    print(f"   System prompt: {FINETUNE_SYSTEM_PROMPT[:60]}...")
    train_dataset = create_prompt_completion_dataset(
        df=train_df,
        tokenizer=tokenizer,
    )
    print(f"   Created {len(train_dataset)} training examples")

    # Check class distribution in first 20 samples
    labels_sample = [json.loads(c)["Label"] for c in train_dataset["completion"][:20]]
    print(f"   First 20 labels: {labels_sample}")

    # Create eval dataset from test data
    test_path = data_dir / "test" / "test_metadata.csv"
    eval_dataset = None
    if test_path.exists():
        test_df = pd.read_csv(test_path)
        eval_df = test_df.sample(n=min(100, len(test_df)), random_state=42)
        eval_dataset = create_prompt_completion_dataset(
            df=eval_df,
            tokenizer=tokenizer,
        )
        print(f"   Created eval dataset with {len(eval_dataset)} samples")

    # Configure training
    output_dir = project_root / train_config['output_dir']

    # Calculate warmup steps from warmup_ratio
    total_steps = (len(train_dataset) // train_config['per_device_train_batch_size']) * train_config['num_train_epochs']
    total_steps = total_steps // train_config['gradient_accumulation_steps']
    warmup_steps = int(total_steps * train_config.get('warmup_ratio', 0.03))

    # Determine eval strategy based on whether we have eval data
    use_early_stopping = eval_dataset is not None

    training_args = SFTConfig(
        output_dir=str(output_dir),
        per_device_train_batch_size=train_config['per_device_train_batch_size'],
        gradient_accumulation_steps=train_config['gradient_accumulation_steps'],
        num_train_epochs=train_config['num_train_epochs'],
        learning_rate=train_config['learning_rate'],
        fp16=False,
        bf16=False,
        gradient_checkpointing=False,
        logging_steps=train_config['logging_steps'],
        save_strategy="steps" if use_early_stopping else train_config['save_strategy'],
        save_steps=50 if use_early_stopping else None,
        save_total_limit=train_config.get('save_total_limit', 2),
        warmup_steps=warmup_steps,
        lr_scheduler_type=train_config.get('lr_scheduler_type', 'linear'),
        max_grad_norm=train_config.get('max_grad_norm', 0.3),
        optim=train_config.get('optim', 'paged_adamw_8bit'),
        report_to=train_config.get('report_to', 'none'),
        eval_strategy="steps" if use_early_stopping else "no",
        eval_steps=50 if use_early_stopping else None,
        load_best_model_at_end=use_early_stopping,
        metric_for_best_model="eval_loss" if use_early_stopping else None,
        greater_is_better=False if use_early_stopping else None,
    )

    # Initialize trainer with optional early stopping
    print("\nInitializing SFTTrainer...")
    print("   Using prompt-completion format (loss computed ONLY on completion tokens)")

    callbacks = []
    if use_early_stopping:
        early_stopping = EarlyStoppingCallback(
            early_stopping_patience=3,
            early_stopping_threshold=0.01
        )
        callbacks.append(early_stopping)
        print("   Early stopping enabled (patience=3, threshold=0.01)")

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        callbacks=callbacks if callbacks else None,
    )

    # Start training
    print("\nStarting training...")
    print(f"   Epochs: {train_config['num_train_epochs']}")
    print(f"   Batch size: {train_config['per_device_train_batch_size']}")
    print(f"   Gradient accumulation: {train_config['gradient_accumulation_steps']}")
    print(f"   Effective batch size: {train_config['per_device_train_batch_size'] * train_config['gradient_accumulation_steps']}")
    print(f"   Learning rate: {train_config['learning_rate']}")
    if use_early_stopping:
        print(f"   Early stopping: enabled (eval every 50 steps, patience=3)")
    print()

    trainer.train()

    # Save final model
    print(f"\nSaving model to {output_dir}...")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)

    # Save training configuration alongside model (for evaluation alignment)
    training_meta = {
        "system_prompt": FINETUNE_SYSTEM_PROMPT,
        "user_template": "Sequence: {sequence}",
        "response_format": "Narrow_peaks, Wide_peaks, Total_peaks, Label",
        "model_name": model_name,
        "num_train_samples": len(train_dataset),
    }
    with open(output_dir / "training_config.json", 'w', encoding='utf-8') as f:
        json.dump(training_meta, f, indent=2, ensure_ascii=False)
    print(f"   Saved training config to {output_dir / 'training_config.json'}")

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    print(f"\nModel saved to: {output_dir}")
    print(f"\nNext steps:")
    print(f"   1. Evaluate (zero-shot, same format as training):")
    print(f"      python scripts/evaluate_finetuned.py --adapter {train_config['output_dir']} --n-shots 0")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fine-tune gemma-2b-it for Task 1 (Regular classification)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/medgemma_finetune.yaml",
        help="Path to configuration YAML file"
    )

    args = parser.parse_args()
    main(args.config)
