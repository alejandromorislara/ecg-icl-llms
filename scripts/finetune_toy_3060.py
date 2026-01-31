"""
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

from src.icl.prompt_builder import PromptManager


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_train_dataset(csv_path: str, shuffle: bool = True, seed: int = 42) -> Dataset:
    """
    Load training data from CSV file.
    
    Expected columns: id, sequence, label, total_picos
    
    Args:
        csv_path: Path to CSV file
        shuffle: Whether to shuffle the dataset (important for balanced training!)
        seed: Random seed for reproducibility
    """
    df = pd.read_csv(csv_path)
    
    if shuffle:
        # IMPORTANT: Shuffle to avoid catastrophic forgetting
        # when dataset is ordered by class
        df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
        print(f"   Shuffled dataset (seed={seed})")
    
    return Dataset.from_pandas(df)


def create_formatting_func(tokenizer, prompt_manager, task: int = 1, approach: str = "regular", max_length: int = 512):
    """
    Create formatting function for SFTTrainer.
    
    This function combines the system/user prompts with the expected JSON response.
    Note: Gemma doesn't support 'system' role, so we merge system into user message.
    """
    
    def formatting_func(example):
        # Build messages using PromptManager
        messages = prompt_manager.build_messages(
            sequence=example['sequence'],
            task=task,
            approach=approach,
            examples=None  # Zero-shot for training
        )
        
        # Gemma doesn't support system role - merge system into user message
        # messages[0] is system, messages[1] is user
        if len(messages) >= 2 and messages[0]['role'] == 'system':
            system_content = messages[0]['content']
            user_content = messages[1]['content']
            # Combine system and user into a single user message
            combined_content = f"{system_content}\n\n{user_content}"
            messages = [{"role": "user", "content": combined_content}]
        
        # Calculate expected output from sequence
        narrow_peaks = example['sequence'].count('|')
        wide_peaks = example['sequence'].count(':')
        total_peaks = narrow_peaks + wide_peaks
        label = example['label']
        
        # Build target JSON response
        target_json = json.dumps({
            "Narrow_peaks": narrow_peaks,
            "Wide_peaks": wide_peaks,
            "Total_peaks": total_peaks,
            "Label": label
        }, indent=2)
        
        # Add assistant response (Gemma uses "model" role for responses)
        messages.append({"role": "model", "content": target_json})
        
        # Apply chat template
        formatted = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=False
        )
        
        return formatted
    
    return formatting_func


def main(config_path: str):
    """Main fine-tuning function."""
    
    print("=" * 60)
    print("Fine-tuning gemma-2b-it for Task 1 (Regular)")
    print("=" * 60)
    
    # Load configuration
    print(f"\n📋 Loading configuration from {config_path}...")
    config = load_config(config_path)
    
    # Lazy imports (heavy libraries)
    print("\n📦 Importing libraries...")
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
    from trl import SFTTrainer, SFTConfig
    
    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"   Device: {device}")
    
    if device == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print(f"\n   ⚠️  WARNING: CUDA not available! Training will be VERY slow on CPU.")
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
    print("\n⚙️  Configuring quantization...")
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
    print(f"\n🤖 Loading model: {model_name}...")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,  # Force float16 for RTX 3060 compatibility
        attn_implementation="eager",  # Required for gradient checkpointing
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
    print("\n🔧 Preparing model for QLoRA training...")
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
    
    # Load dataset
    data_config = config['data']
    train_path = project_root / data_config['train_path']
    print(f"\n📊 Loading training data from {train_path}...")
    
    if not train_path.exists():
        print(f"   ❌ Training data not found!")
        print(f"   Run: python scripts/generate_toy_dataset.py --n-train-samples 300")
        sys.exit(1)
    
    train_dataset = load_train_dataset(str(train_path))
    print(f"   Loaded {len(train_dataset)} training samples")
    
    # Load training config (needed for formatting function)
    train_config = config['training']
    
    # Initialize PromptManager
    prompts_dir = project_root / data_config['prompts_dir']
    prompt_manager = PromptManager(prompts_dir=prompts_dir)
    
    # Create formatting function
    formatting_func = create_formatting_func(
        tokenizer=tokenizer,
        prompt_manager=prompt_manager,
        task=data_config['task'],
        approach=data_config['approach'],
        max_length=train_config['max_seq_length']
    )
    
    # Configure training
    output_dir = project_root / train_config['output_dir']
    
    # Calculate warmup steps from warmup_ratio
    total_steps = (len(train_dataset) // train_config['per_device_train_batch_size']) * train_config['num_train_epochs']
    total_steps = total_steps // train_config['gradient_accumulation_steps']
    warmup_steps = int(total_steps * train_config.get('warmup_ratio', 0.03))
    
    training_args = SFTConfig(
        output_dir=str(output_dir),
        per_device_train_batch_size=train_config['per_device_train_batch_size'],
        gradient_accumulation_steps=train_config['gradient_accumulation_steps'],
        num_train_epochs=train_config['num_train_epochs'],
        learning_rate=train_config['learning_rate'],
        fp16=False,  # Disable fp16 scaler to avoid bfloat16 conflict
        bf16=False,
        gradient_checkpointing=False,  # Already enabled in prepare_model_for_kbit_training
        logging_steps=train_config['logging_steps'],
        save_strategy=train_config['save_strategy'],
        save_total_limit=train_config.get('save_total_limit', 2),
        warmup_steps=warmup_steps,
        lr_scheduler_type=train_config.get('lr_scheduler_type', 'linear'),
        max_grad_norm=train_config.get('max_grad_norm', 0.3),
        optim=train_config.get('optim', 'paged_adamw_8bit'),
        report_to=train_config.get('report_to', 'none'),
    )
    
    # Initialize trainer
    print("\n🏋️ Initializing SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        formatting_func=formatting_func,
    )
    
    # Start training
    print("\n🚀 Starting training...")
    print(f"   Epochs: {train_config['num_train_epochs']}")
    print(f"   Batch size: {train_config['per_device_train_batch_size']}")
    print(f"   Gradient accumulation: {train_config['gradient_accumulation_steps']}")
    print(f"   Effective batch size: {train_config['per_device_train_batch_size'] * train_config['gradient_accumulation_steps']}")
    print(f"   Learning rate: {train_config['learning_rate']}")
    print()
    
    trainer.train()
    
    # Save final model
    print(f"\n💾 Saving model to {output_dir}...")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    print("\n" + "=" * 60)
    print("✅ Training complete!")
    print("=" * 60)
    print(f"\n📂 Model saved to: {output_dir}")
    print(f"\n🎯 Next steps:")
    print(f"   1. Evaluate model: python scripts/evaluate.py --model {output_dir}")
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
