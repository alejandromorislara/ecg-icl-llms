"""
Data loaders for ICL experiments and model training.

This module provides utilities to load ICL examples, test datasets, and real ECG data.
"""

from pathlib import Path
from typing import List, Tuple, Dict, Optional
import random
import pandas as pd


class ICLDataLoader:
    """Loads data for In-Context Learning experiments"""
    
    def __init__(self, data_dir: Optional[Path] = None):
        """
        Initialize data loader.
        
        Args:
            data_dir: Base directory containing icl/ and test/ subdirectories
        """
        self.data_dir = Path(data_dir) if data_dir else None

    def load_icl_examples(self, n_shots: Optional[int] = None) -> List[Tuple[str, str]]:
        """
        Load examples for In-Context Learning from CSV.
        
        Args:
            n_shots: Number of examples per class to load. If None, loads all available.
        
        Returns:
            List of tuples (sequence, class_label)
        """
        if self.data_dir is None:
            raise ValueError("data_dir must be set to load ICL examples")
        
        file_path = self.data_dir / "icl" / "metadata.csv"
        all_examples = self._load_csv_file(file_path)
        
        if n_shots is None:
            return all_examples
        
        # Sample n_shots per class (balanced)
        return self.sample_balanced_examples(all_examples, n_shots, seed=42)
    
    def load_test_data(self, ood: bool = False) -> List[Tuple[str, str]]:
        """
        Load test data from CSV.
        
        Args:
            ood: If True, loads out-of-distribution test set, otherwise in-distribution
        
        Returns:
            List of tuples (sequence, class_label)
        """
        if self.data_dir is None:
            raise ValueError("data_dir must be set to load test data")
        
        filename = "test_ood_metadata.csv" if ood else "test_metadata.csv"
        file_path = self.data_dir / "test" / filename
        return self._load_csv_file(file_path)
    
    def _load_csv_file(self, file_path: Path) -> List[Tuple[str, str]]:
        """Load a CSV file with columns: id, sequence, label, total_picos"""
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        df = pd.read_csv(file_path)
        
        # Check required columns
        required_cols = ['sequence', 'label']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing columns in {file_path}: {missing_cols}")
        
        # Convert to list of tuples
        data = [(row['sequence'], row['label']) for _, row in df.iterrows()]
        return data
    
    def get_examples_by_class(self, data: List[Tuple[str, str]]) -> Dict[str, List[Tuple[str, str]]]:
        """Group examples by class"""
        by_class = {'Brady': [], 'Normal': [], 'Tachy': []}
        
        for sequence, fc_class in data:
            if fc_class in by_class:
                by_class[fc_class].append((sequence, fc_class))
        
        return by_class
    
    def sample_balanced_examples(self, data: List[Tuple[str, str]], 
                                 n_per_class: int, seed: int = 42) -> List[Tuple[str, str]]:
        """
        Sample N examples per class (balanced sampling).
        
        Args:
            data: List of examples
            n_per_class: Number of examples per class
            seed: Random seed for reproducibility
        
        Returns:
            List of balanced samples
        """
        random.seed(seed)
        
        by_class = self.get_examples_by_class(data)
        sampled = []
        
        for fc_class in ['Brady', 'Normal', 'Tachy']:
            class_examples = by_class[fc_class]
            if len(class_examples) < n_per_class:
                print(f"Warning: only {len(class_examples)} examples available for class {fc_class}")
                sampled.extend(class_examples)
            else:
                sampled.extend(random.sample(class_examples, n_per_class))
        
        # Shuffle final examples
        random.shuffle(sampled)
        return sampled
    
    def print_stats(self, data: List[Tuple[str, str]], title: str = "Dataset"):
        """Print basic dataset statistics"""
        print(f"\n=== {title} ===")
        print(f"Total examples: {len(data)}")
        
        by_class = self.get_examples_by_class(data)
        for fc_class in ['Brady', 'Normal', 'Tachy']:
            count = len(by_class[fc_class])
            pct = (count / len(data) * 100) if data else 0
            print(f"  {fc_class}: {count} ({pct:.1f}%)")
        
        if data:
            # Sequence statistics
            sequences = [seq for seq, _ in data]
            seq_lengths = [len(seq) for seq in sequences]
            avg_length = sum(seq_lengths) / len(seq_lengths)
            print(f"Average length: {avg_length:.1f} characters")
            
            # Symbol counts
            all_sequences = ''.join(sequences)
            symbol_counts = {
                'narrow peaks (|)': all_sequences.count('|'),
                'wide peaks (:)': all_sequences.count(':'),
                'baseline (.)': all_sequences.count('.'),
                'noise (~)': all_sequences.count('~'),
                'artifacts (_)': all_sequences.count('_')
            }
            
            print("Symbols by type:")
            for symbol, count in symbol_counts.items():
                pct = (count / len(all_sequences) * 100) if all_sequences else 0
                print(f"  {symbol}: {count} ({pct:.1f}%)")

