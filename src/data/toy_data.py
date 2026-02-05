"""
Generador de datos sintéticos para experimentos toy (Task 1: secuencias simbólicas de ECG).

Este módulo genera secuencias de texto que simulan señales ECG simplificadas usando
un alfabeto simbólico {.|:_~} para experimentos de In-Context Learning.
"""

import numpy as np
import json
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TextConfig:
    """Configuración para la generación de secuencias de texto"""
    sequence_length: int = 100
    
    # Heart rate ranges (number of peaks in sequence) - NON-OVERLAPPING
    # np.randint upper bound is EXCLUSIVE: (4, 8) generates {4,5,6,7}
    brady_range: Tuple[int, int] = (4, 8)   # Bradycardia: 4-7 peaks (<60 bpm)
    normal_range: Tuple[int, int] = (8, 12)  # Normal: 8-11 peaks (60-100 bpm)
    tachy_range: Tuple[int, int] = (12, 17)  # Tachycardia: 12-16 peaks (>100 bpm)
    
    # Noise and artifact probability ranges
    noise_prob_range: Tuple[float, float] = (0.05, 0.25)
    artifact_prob_range: Tuple[float, float] = (0.02, 0.10)
    
    # Peak type ratio (wide peaks / total peaks)
    wide_peak_ratio_range: Tuple[float, float] = (0.2, 0.8)
    
    # RR interval variability
    regular_rr_std: float = 0.05      # Low variability (regular rhythm)
    irregular_rr_std: float = 0.15    # High variability (irregular rhythm)


@dataclass
class TextSample:
    """Represents a single synthetic ECG text sample"""
    sequence: str
    concepts: Dict[str, float]
    fc_class: str
    rr_intervals: List[float]
    metadata: Dict


class TextGenerator:
    """Generates symbolic text sequences simulating ECG signals."""

    def __init__(self, config: Optional[TextConfig] = None):
        self.config = config or TextConfig()
        self.rng = np.random.RandomState(42)

    def set_seed(self, seed: int):
        """Set random seed for reproducibility"""
        self.rng = np.random.RandomState(seed)

    def _generate_rr_intervals(self, n_peaks: int, irregularity_std: float) -> List[float]:
        """Generate RR intervals (distances between peaks) with variability"""
        if n_peaks < 2:
            return [1.0]
        
        base_interval = self.config.sequence_length / (n_peaks + 1)
        intervals = []
        
        for _ in range(n_peaks - 1):
            interval = base_interval + self.rng.normal(0, irregularity_std * base_interval)
            interval = max(5, min(interval, 50))  # Clamp to reasonable range
            intervals.append(interval)
        
        return intervals

    def _place_peaks(self, n_narrow: int, n_wide: int, rr_intervals: List[float]) -> List[Tuple[int, str]]:
        """Place peaks in sequence positions based on RR intervals"""
        total_peaks = n_narrow + n_wide
        if total_peaks == 0:
            return []
        
        positions = []
        current_pos = self.rng.randint(10, 30)  # Start position
        
        for i in range(total_peaks):
            if current_pos >= self.config.sequence_length - 5:
                current_pos = self.rng.randint(10, self.config.sequence_length - 5)
            
            positions.append(current_pos)
            
            # Move to next peak position
            if i < len(rr_intervals):
                current_pos += int(rr_intervals[i])
            else:
                current_pos += int(np.mean(rr_intervals)) if rr_intervals else 20
        
        # Assign peak types randomly
        peak_types = ['|'] * n_narrow + [':'] * n_wide
        self.rng.shuffle(peak_types)
        
        return [(pos, peak_type) for pos, peak_type in zip(positions, peak_types)]

    def _add_noise_and_artifacts(self, sequence: List[str], noise_prob: float, 
                                 artifact_prob: float) -> List[str]:
        """Add noise (~) and artifacts (_) to baseline positions"""
        result = sequence.copy()
        
        for i in range(len(result)):
            if result[i] == '.':
                rand_val = self.rng.random()
                if rand_val < artifact_prob:
                    result[i] = '_'
                elif rand_val < artifact_prob + noise_prob:
                    result[i] = '~'
        
        return result

    def _calculate_concepts(self, sequence: str, rr_intervals: List[float]) -> Dict[str, float]:
        """Calculate interpretable concepts from sequence"""
        n_narrow = sequence.count('|')
        n_wide = sequence.count(':')
        n_noise = sequence.count('~')
        n_artifacts = sequence.count('_')
        
        # Concept 3: Noise percentage
        noise_symbols = n_noise + n_artifacts
        c3_noise_pct = (noise_symbols / len(sequence)) * 100
        
        # Concept 4: RR interval irregularity
        if len(rr_intervals) > 1:
            rr_mean = np.mean(rr_intervals)
            rr_std = np.std(rr_intervals)
            c4_irregularity = (rr_std / rr_mean) if rr_mean > 0 else 0.0
        else:
            c4_irregularity = 0.0
        
        # Concept 5: Significant artifacts
        c5_artifacts = (n_artifacts / len(sequence)) > 0.05
        
        return {
            'C1_picos_estrechos': n_narrow,
            'C2_picos_anchos': n_wide,
            'C3_pct_ruido': round(c3_noise_pct, 2),
            'C4_irregularidad': round(c4_irregularity, 3),
            'C5_artefactos_significativos': c5_artifacts
        }

    def _determine_fc_class(self, total_peaks: int) -> str:
        """Determine heart rate class based on total peak count"""
        if total_peaks <= 7:
            return 'Brady'
        elif total_peaks <= 11:
            return 'Normal'
        else:
            return 'Tachy'

    def generate_sample(self, fc_class: Optional[str] = None) -> TextSample:
        """
        Generate a single synthetic sample.
        
        Args:
            fc_class: Target class ('Brady', 'Normal', 'Tachy', or None for random)
        
        Returns:
            TextSample with sequence, concepts, class, and metadata
        """
        # Determine number of peaks based on target class
        if fc_class == 'Brady':
            total_peaks = self.rng.randint(*self.config.brady_range)
        elif fc_class == 'Tachy':
            total_peaks = self.rng.randint(*self.config.tachy_range)
        elif fc_class == 'Normal':
            total_peaks = self.rng.randint(*self.config.normal_range)
        else:
            all_range = (self.config.brady_range[0], self.config.tachy_range[1])
            total_peaks = self.rng.randint(*all_range)
        
        actual_fc_class = self._determine_fc_class(total_peaks)
        
        # Determine peak types
        wide_ratio = self.rng.uniform(*self.config.wide_peak_ratio_range)
        n_wide = int(total_peaks * wide_ratio)
        n_narrow = total_peaks - n_wide
        
        # Generate RR intervals (rhythm regularity)
        is_irregular = self.rng.random() > 0.7
        irregularity_std = (self.config.irregular_rr_std if is_irregular 
                           else self.config.regular_rr_std)
        rr_intervals = self._generate_rr_intervals(total_peaks, irregularity_std)
        
        # Initialize sequence with baseline
        sequence = ['.'] * self.config.sequence_length
        
        # Place peaks
        peak_positions = self._place_peaks(n_narrow, n_wide, rr_intervals)
        for pos, peak_type in peak_positions:
            if 0 <= pos < len(sequence):
                sequence[pos] = peak_type
        
        # Add noise and artifacts
        noise_prob = self.rng.uniform(*self.config.noise_prob_range)
        artifact_prob = self.rng.uniform(*self.config.artifact_prob_range)
        sequence = self._add_noise_and_artifacts(sequence, noise_prob, artifact_prob)
        
        # Convert to string
        sequence_str = ''.join(sequence)
        
        # Calculate concepts
        concepts = self._calculate_concepts(sequence_str, rr_intervals)
        
        # Store metadata
        metadata = {
            'target_fc_class': fc_class,
            'actual_fc_class': actual_fc_class,
            'actual_total_peaks': n_narrow + n_wide,
            'narrow_peaks': n_narrow,
            'wide_peaks': n_wide,
            'noise_prob': noise_prob,
            'artifact_prob': artifact_prob,
            'is_irregular': is_irregular,
            'irregularity_std': irregularity_std,
            'peak_positions': peak_positions
        }
        
        return TextSample(
            sequence=sequence_str,
            concepts=concepts,
            fc_class=actual_fc_class,
            rr_intervals=rr_intervals,
            metadata=metadata
        )

    def generate_dataset(self, n_samples_per_class: int = 10, 
                        classes: List[str] = None) -> List[TextSample]:
        """
        Generate a balanced dataset with multiple samples per class.
        
        Args:
            n_samples_per_class: Number of samples to generate per class
            classes: List of classes to generate (default: all 3 classes)
        
        Returns:
            List of TextSample objects
        """
        if classes is None:
            classes = ['Brady', 'Normal', 'Tachy']
        
        dataset = []
        for fc_class in classes:
            for _ in range(n_samples_per_class):
                sample = self.generate_sample(fc_class)
                dataset.append(sample)
        
        return dataset

    def save_dataset_txt(self, dataset: List[TextSample], output_path: Path):
        """Save dataset in simple text format (sequence\tclass)"""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for i, sample in enumerate(dataset):
                f.write(f"{sample.sequence}\t{sample.fc_class}\n")

    def save_metadata_csv(self, dataset: List[TextSample], output_path: Path, 
                         dataset_name: str):
        """Save dataset metadata as CSV (for ICL and evaluation)"""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        metadata_rows = []
        for i, sample in enumerate(dataset):
            sample_id = f"{dataset_name}_{i:04d}"
            metadata_row = {
                'id': sample_id,
                'sequence': sample.sequence,
                'label': sample.fc_class,
                'total_picos': sample.concepts['C1_picos_estrechos'] + sample.concepts['C2_picos_anchos']
            }
            metadata_rows.append(metadata_row)
        
        df = pd.DataFrame(metadata_rows)
        df.to_csv(output_path, index=False)

    def save_dataset_jsonl(self, dataset: List[TextSample], output_path: Path):
        """Save complete dataset with all metadata as JSONL"""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for sample in dataset:
                data = {
                    'sequence': sample.sequence,
                    'concepts': sample.concepts,
                    'fc_class': sample.fc_class,
                    'rr_intervals': sample.rr_intervals,
                    'metadata': sample.metadata
                }
                f.write(json.dumps(data, ensure_ascii=False) + '\n')

    def load_dataset_txt(self, input_path: Path) -> List[Tuple[str, str]]:
        """Load dataset from simple text format"""
        dataset = []
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    sequence, fc_class = line.split('\t')
                    dataset.append((sequence, fc_class))
        return dataset

    def load_dataset_jsonl(self, input_path: Path) -> List[TextSample]:
        """Load complete dataset from JSONL"""
        dataset = []
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                sample = TextSample(
                    sequence=data['sequence'],
                    concepts=data['concepts'],
                    fc_class=data['fc_class'],
                    rr_intervals=data['rr_intervals'],
                    metadata=data['metadata']
                )
                dataset.append(sample)
        return dataset

