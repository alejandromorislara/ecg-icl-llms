from pathlib import Path
import sys
import argparse
import json
import shutil

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.toy_data import TextGenerator, TextConfig

# System prompt for fine-tuning (shared with finetune and eval scripts)
FINETUNE_SYSTEM_PROMPT = (
    "You are a text-based ECG analyzer. "
    "Count the characters '|' (narrow R-wave peaks) and ':' (wide T-wave peaks). "
    "Sum both to get the total. "
    "Classify as: Brady [4-7], Normal [8-11], Tachy [12-16]."
)


def generate_finetune_jsonl(dataset, output_path):
    """
    Generate JSONL file with conversational messages format for fine-tuning.

    Format follows tutor's specification:
    {"messages": [
        {"role": "system", "content": "..."},
        {"role": "user", "content": "Secuencia: ..."},
        {"role": "assistant", "content": "{\"Conceptos\": {...}, \"Label\": \"...\"}"}
    ]}
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        for sample in dataset:
            narrow = sample.sequence.count('|')
            wide = sample.sequence.count(':')
            total = narrow + wide

            response = json.dumps({
                "Narrow_peaks": narrow,
                "Wide_peaks": wide,
                "Total_peaks": total,
                "Label": sample.fc_class
            })

            entry = {
                "messages": [
                    {"role": "system", "content": FINETUNE_SYSTEM_PROMPT},
                    {"role": "user", "content": f"Sequence: {sample.sequence}"},
                    {"role": "assistant", "content": response}
                ]
            }
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')


def generate_toy_datasets(n_train_samples: int = 1500,
                         n_test_samples: int = 999,
                         n_ood_samples: int = 300,
                         n_icl_samples: int = 8,
                         output_dir: str = "data/processed/toy_experiment"):
    """
    Generate all datasets for toy experiment.

    Args:
        n_train_samples: Number of training samples for fine-tuning (divided by 3 for balanced classes)
        n_test_samples: Number of test samples (divided by 3 for balanced classes)
        n_ood_samples: Number of OOD samples (divided by 3 for balanced classes)
        n_icl_samples: Number of ICL examples per class
        output_dir: Output directory for generated data
    """

    print("=" * 60)
    print("Generating Toy Experiment Datasets")
    print("=" * 60)

    # Configuration for in-distribution data
    # Ranges use np.randint (exclusive upper bound) - defined in TextConfig defaults
    config = TextConfig(
        sequence_length=100,
        noise_prob_range=(0.05, 0.25),
        artifact_prob_range=(0.02, 0.10),
        wide_peak_ratio_range=(0.2, 0.8)
    )

    generator = TextGenerator(config)

    # Setup directories
    base_dir = project_root / output_dir
    train_dir = base_dir / "train"
    icl_dir = base_dir / "icl"
    test_dir = base_dir / "test"

    # Clean and recreate directories
    if base_dir.exists():
        print(f"\nRemoving existing data at {base_dir}...")
        shutil.rmtree(base_dir)

    train_dir.mkdir(parents=True, exist_ok=True)
    icl_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    # Generate training set for fine-tuning
    print(f"\nGenerating training dataset ({n_train_samples} total, {n_train_samples // 3} per class)...")
    generator.set_seed(100)
    train_dataset = generator.generate_dataset(n_samples_per_class=n_train_samples // 3)
    generator.save_metadata_csv(train_dataset, train_dir / "metadata.csv", "train")
    print(f"   Saved CSV to {train_dir / 'metadata.csv'}")

    # Export JSONL with conversational messages format for fine-tuning
    jsonl_path = train_dir / "finetune_messages.jsonl"
    generate_finetune_jsonl(train_dataset, jsonl_path)
    print(f"   Saved JSONL to {jsonl_path}")

    # Generate ICL examples
    print(f"\nGenerating ICL examples ({n_icl_samples * 3} total, {n_icl_samples} per class)...")
    generator.set_seed(42)
    icl_dataset = generator.generate_dataset(n_samples_per_class=n_icl_samples)
    generator.save_metadata_csv(icl_dataset, icl_dir / "metadata.csv", "icl")
    print(f"   Saved to {icl_dir / 'metadata.csv'}")

    # Generate in-distribution test set
    print(f"\nGenerating test dataset ({n_test_samples} total)...")
    generator.set_seed(200)
    test_dataset = generator.generate_dataset(n_samples_per_class=n_test_samples // 3)
    generator.save_metadata_csv(test_dataset, test_dir / "test_metadata.csv", "test")
    print(f"   Saved to {test_dir / 'test_metadata.csv'}")

    # Generate OOD test set with different distribution
    print(f"\nGenerating OOD dataset ({n_ood_samples} total)...")
    print("   Shifted distribution:")
    print("   - Brady: 2-5 peaks (vs normal 4-7)")
    print("   - Normal: 9-10 peaks (vs normal 8-11)")
    print("   - Tachy: 14-20 peaks (vs normal 12-16)")
    print("   - Higher noise: 20-40% (vs normal 5-25%)")
    print("   - More artifacts: 8-18% (vs normal 2-10%)")

    ood_config = TextConfig(
        sequence_length=100,
        brady_range=(2, 6),      # Shifted down: generates {2,3,4,5}
        normal_range=(9, 11),    # Narrower: generates {9,10}
        tachy_range=(14, 21),    # Shifted up: generates {14,...,20}
        noise_prob_range=(0.20, 0.40),
        artifact_prob_range=(0.08, 0.18),
    )

    ood_generator = TextGenerator(ood_config)
    ood_generator.set_seed(300)
    ood_dataset = ood_generator.generate_dataset(n_samples_per_class=n_ood_samples // 3)
    ood_generator.save_metadata_csv(ood_dataset, test_dir / "test_ood_metadata.csv", "test_ood")
    print(f"   Saved to {test_dir / 'test_ood_metadata.csv'}")

    # Save configuration
    print(f"\nSaving configuration...")
    config_dict = {
        'description': 'Toy experiment: Symbolic ECG sequences for ICL testing',
        'sequence_length': 100,
        'finetune_system_prompt': FINETUNE_SYSTEM_PROMPT,
        'in_distribution': {
            'brady_range': [4, 7],
            'normal_range': [8, 11],
            'tachy_range': [12, 16],
            'noise_prob_range': list(config.noise_prob_range),
            'artifact_prob_range': list(config.artifact_prob_range),
        },
        'out_of_distribution': {
            'brady_range': [2, 5],
            'normal_range': [9, 10],
            'tachy_range': [14, 20],
            'noise_prob_range': list(ood_config.noise_prob_range),
            'artifact_prob_range': list(ood_config.artifact_prob_range),
        },
        'dataset_sizes': {
            'train_per_class': n_train_samples // 3,
            'train_total': len(train_dataset),
            'icl_per_class': n_icl_samples,
            'icl_total': n_icl_samples * 3,
            'test_total': len(test_dataset),
            'test_ood_total': len(ood_dataset),
        }
    }

    with open(base_dir / "config.json", 'w', encoding='utf-8') as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)

    print(f"   Saved to {base_dir / 'config.json'}")

    # Summary
    print("\n" + "=" * 60)
    print("Dataset generation complete!")
    print("=" * 60)
    print(f"\nOutput directory: {base_dir}")
    print(f"\nDataset sizes:")
    print(f"   - Training: {len(train_dataset)} ({n_train_samples // 3} per class)")
    print(f"   - Training JSONL (messages format): {jsonl_path}")
    print(f"   - ICL examples: {len(icl_dataset)} ({n_icl_samples} per class)")
    print(f"   - Test (in-distribution): {len(test_dataset)}")
    print(f"   - Test (OOD): {len(ood_dataset)}")
    print(f"\nNext steps:")
    print(f"   1. Review the generated data in {base_dir}")
    print(f"   2. Fine-tune model: python scripts/finetune_toy_3060.py --config configs/medgemma_finetune.yaml")
    print(f"   3. Run ICL evaluation: python scripts/evaluate.py --task 1 --n-shots 4")
    print()

    return base_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate datasets for toy experiment (Task 1: symbolic sequences)"
    )
    parser.add_argument(
        "--n-train-samples",
        type=int,
        default=1500,
        help="Number of training samples for fine-tuning (default: 1500, 500 per class)"
    )
    parser.add_argument(
        "--n-test-samples",
        type=int,
        default=999,
        help="Number of test samples (default: 999)"
    )
    parser.add_argument(
        "--n-ood-samples",
        type=int,
        default=300,
        help="Number of OOD samples (default: 300)"
    )
    parser.add_argument(
        "--n-icl-samples",
        type=int,
        default=8,
        help="Number of ICL examples per class (default: 8)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed/toy_experiment",
        help="Output directory (default: data/processed/toy_experiment)"
    )

    args = parser.parse_args()

    generate_toy_datasets(
        n_train_samples=args.n_train_samples,
        n_test_samples=args.n_test_samples,
        n_ood_samples=args.n_ood_samples,
        n_icl_samples=args.n_icl_samples,
        output_dir=args.output_dir
    )
