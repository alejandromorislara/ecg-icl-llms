"""
Debug script to inspect the exact format of training data.
"""
import sys
sys.path.insert(0, '.')
import json
import pandas as pd
from transformers import AutoTokenizer
from src.icl.prompt_builder import PromptManager
from pathlib import Path

def main():
    # Load tokenizer
    print('Loading tokenizer...')
    tokenizer = AutoTokenizer.from_pretrained('google/gemma-2b-it')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load prompt manager
    project_root = Path('.')
    prompts_dir = project_root / 'src' / 'prompts'
    prompt_manager = PromptManager(prompts_dir=prompts_dir)

    # Load samples
    df = pd.read_csv('data/processed/toy_experiment/train/metadata.csv')
    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Test with 3 samples (one of each class)
    samples_to_test = [
        df_shuffled[df_shuffled['label'] == 'Brady'].iloc[0],
        df_shuffled[df_shuffled['label'] == 'Normal'].iloc[0],
        df_shuffled[df_shuffled['label'] == 'Tachy'].iloc[0],
    ]

    for i, sample in enumerate(samples_to_test):
        print(f'\n{"="*70}')
        print(f'SAMPLE {i+1}: {sample["label"]}')
        print(f'{"="*70}')
        
        # Build messages
        messages = prompt_manager.build_messages(
            sequence=sample['sequence'],
            task=1,
            approach='regular',
            examples=None
        )
        
        # Merge system into user (as the finetuning script does)
        if len(messages) >= 2 and messages[0]['role'] == 'system':
            system_content = messages[0]['content']
            user_content = messages[1]['content']
            combined_content = f'{system_content}\n\n{user_content}'
            messages = [{'role': 'user', 'content': combined_content}]
        
        # Build target JSON
        narrow_peaks = sample['sequence'].count('|')
        wide_peaks = sample['sequence'].count(':')
        total_peaks = narrow_peaks + wide_peaks
        
        target_json = json.dumps({
            'Narrow_peaks': narrow_peaks,
            'Wide_peaks': wide_peaks,
            'Total_peaks': total_peaks,
            'Label': sample['label']
        }, indent=2)
        
        print(f'Sequence length: {len(sample["sequence"])}')
        print(f'Narrow: {narrow_peaks}, Wide: {wide_peaks}, Total: {total_peaks}')
        print(f'Label: {sample["label"]}')
        print(f'\nTarget JSON:\n{target_json}')
        
        # Add model response
        messages.append({'role': 'model', 'content': target_json})
        
        # Apply chat template
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        
        print(f'\n--- FORMATTED FOR TRAINING ---')
        print(formatted)
        print(f'\n--- TOKEN COUNT ---')
        tokens = tokenizer(formatted, return_tensors='pt')
        print(f'Total tokens: {len(tokens["input_ids"][0])}')
        
        # Check if response is actually in the tokenized output
        response_tokens = tokenizer(target_json, return_tensors='pt')
        print(f'Response tokens: {len(response_tokens["input_ids"][0])}')

    # Additional diagnostics
    print(f'\n{"="*70}')
    print('DATASET STATISTICS')
    print(f'{"="*70}')
    
    # Check peak distribution by class
    for label in ['Brady', 'Normal', 'Tachy']:
        subset = df[df['label'] == label]
        peaks = subset['sequence'].apply(lambda x: x.count('|') + x.count(':'))
        print(f'\n{label}:')
        print(f'  Min peaks: {peaks.min()}')
        print(f'  Max peaks: {peaks.max()}')
        print(f'  Mean peaks: {peaks.mean():.1f}')
        print(f'  Expected range: {"4-7" if label == "Brady" else "8-11" if label == "Normal" else "12-18"}')

if __name__ == '__main__':
    main()
