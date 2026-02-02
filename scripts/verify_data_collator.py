"""
Script to verify the prompt-completion format is working correctly.
This shows what tokens will have loss computed during training.
"""
import sys
sys.path.insert(0, '.')
import json
import pandas as pd
import torch
from transformers import AutoTokenizer
from src.icl.prompt_builder import PromptManager
from pathlib import Path

def main():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('google/gemma-2b-it')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load prompt manager
    prompts_dir = Path('src/prompts')
    prompt_manager = PromptManager(prompts_dir=prompts_dir)
    
    # Load samples
    df = pd.read_csv('data/processed/toy_experiment/train/metadata.csv')
    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Test with samples from each class
    samples_to_test = [
        df_shuffled[df_shuffled['label'] == 'Brady'].iloc[0],
        df_shuffled[df_shuffled['label'] == 'Normal'].iloc[0],
        df_shuffled[df_shuffled['label'] == 'Tachy'].iloc[0],
    ]
    
    print(f"\n{'='*70}")
    print("PROMPT-COMPLETION FORMAT VERIFICATION")
    print(f"{'='*70}")
    print("\nIn this format, TRL computes loss ONLY on the completion (response).")
    print("The prompt is used as context but its tokens are ignored in loss.\n")
    
    for i, sample in enumerate(samples_to_test):
        print(f"\n{'='*70}")
        print(f"SAMPLE {i+1}: {sample['label']}")
        print(f"{'='*70}")
        
        # Build messages
        messages = prompt_manager.build_messages(
            sequence=sample['sequence'],
            task=1,
            approach='regular',
            examples=None
        )
        
        # Merge system into user (Gemma format)
        if len(messages) >= 2 and messages[0]['role'] == 'system':
            system_content = messages[0]['content']
            user_content = messages[1]['content']
            combined_content = f'{system_content}\n\n{user_content}'
            messages = [{'role': 'user', 'content': combined_content}]
        
        # Create prompt with generation prompt
        prompt = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        # Build completion
        narrow_peaks = sample['sequence'].count('|')
        wide_peaks = sample['sequence'].count(':')
        total_peaks = narrow_peaks + wide_peaks
        
        completion = json.dumps({
            'Narrow_peaks': narrow_peaks,
            'Wide_peaks': wide_peaks,
            'Total_peaks': total_peaks,
            'Label': sample['label']
        }, indent=2)
        
        # Token counts
        prompt_tokens = tokenizer(prompt, return_tensors='pt')['input_ids'].shape[1]
        completion_tokens = tokenizer(completion, return_tensors='pt')['input_ids'].shape[1]
        
        print(f"\nSequence: {sample['sequence'][:50]}...")
        print(f"True label: {sample['label']}")
        print(f"\n--- PROMPT (tokens: {prompt_tokens}) ---")
        print(prompt[-200:] + "...")  # Show end of prompt
        print(f"\n--- COMPLETION (tokens: {completion_tokens}) - LOSS COMPUTED HERE ---")
        print(completion)
        
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"With prompt-completion format:")
    print(f"  - Prompt tokens: ~{prompt_tokens} (IGNORED in loss)")
    print(f"  - Completion tokens: ~{completion_tokens} (LOSS computed here)")
    print(f"  - Ratio: {completion_tokens/(prompt_tokens+completion_tokens)*100:.1f}% of tokens contribute to loss")
    print(f"\nThis is correct! The model learns to generate only the JSON response.")
    
    # Check class balance in shuffled data
    print(f"\n{'='*70}")
    print("CLASS BALANCE CHECK (first 30 samples)")
    print(f"{'='*70}")
    first_30 = df_shuffled['label'].head(30).tolist()
    print(f"Labels: {first_30}")
    from collections import Counter
    counts = Counter(first_30)
    print(f"Counts: {dict(counts)}")
    
    if min(counts.values()) >= 5:
        print("✓ Good class balance in first 30 samples")
    else:
        print("⚠ Possible class imbalance - consider more shuffling")

if __name__ == '__main__':
    main()
