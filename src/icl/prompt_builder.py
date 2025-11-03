"""
Prompt management and formatting utilities for In-Context Learning.

This module handles prompt templates, example formatting, and response parsing
for ICL experiments with different approaches (regular classification, CBM, etc.).
"""

from pathlib import Path
from typing import Dict, List, Optional, Any
import json


class PromptManager:
    """Manages loading and formatting of prompts for ICL"""
    
    # Task to modality mapping
    TASK_TO_MODALITY = {
        1: "text",      # Task 1: Symbolic text sequences
        2: "vision",    # Task 2: 1D rendered signals (future)
        3: "vision"     # Task 3: Real ECGs (future)
    }
    
    def __init__(self, prompts_dir: Optional[Path] = None):
        """
        Initialize prompt manager.
        
        Args:
            prompts_dir: Directory containing prompt templates
        """
        if prompts_dir is None:
            # Default path relative to this file
            self.prompts_dir = Path(__file__).parent.parent / "prompts"
        else:
            self.prompts_dir = Path(prompts_dir)
    
    def load_prompt(self, approach: str, prompt_type: str) -> str:
        """
        Load a specific prompt template.
        
        Args:
            approach: 'regular' or 'cbm' (Concept Bottleneck Model)
            prompt_type: 'system_prompt', 'user_prompt_zero_shot', 'user_prompt_few_shot'
        
        Returns:
            Prompt template as string
        """
        prompt_path = self.prompts_dir / approach / f"{prompt_type}.txt"
        
        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt not found: {prompt_path}")
        
        with open(prompt_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    
    def format_example(self, sample: Dict[str, Any], approach: str = "regular", 
                      example_num: int = 1) -> str:
        """
        Format a single example for few-shot learning.
        
        Args:
            sample: Dictionary with 'sequence', 'fc_class' (and 'concepts' for CBM)
            approach: 'regular' or 'cbm'
            example_num: Example number (for reference)
        
        Returns:
            Formatted example string
        """
        if approach == "regular":
            sequence = sample['sequence']
            fc_class = sample['fc_class']
            
            # Count peaks
            narrow_peaks = sequence.count('|')
            wide_peaks = sequence.count(':')
            total_peaks = narrow_peaks + wide_peaks
            
            return f"""Example {example_num}:
Sequence: {sequence}
Narrow peaks (|): {narrow_peaks}
Wide peaks (:): {wide_peaks}
Total peaks: {total_peaks}
Classification: {fc_class}"""
        
        elif approach == "cbm":
            sequence = sample['sequence']
            concepts = sample.get('concepts', {})
            fc_class = sample['fc_class']
            
            concepts_json = json.dumps(concepts, ensure_ascii=False, indent=2)
            return f"""Example {example_num}:
Sequence: {sequence}
Concepts:
{concepts_json}
Classification: {fc_class}"""
        
        else:
            raise ValueError(f"Unsupported approach: {approach}")
    
    def format_few_shot_examples(self, examples: List[Dict[str, Any]], 
                                approach: str = "regular") -> str:
        """
        Format multiple examples for few-shot learning.
        
        Args:
            examples: List of example dictionaries
            approach: 'regular' or 'cbm'
        
        Returns:
            Formatted examples string
        """
        formatted_examples = []
        
        for i, example in enumerate(examples, 1):
            formatted = self.format_example(example, approach, example_num=i)
            formatted_examples.append(formatted)
        
        return "\n\n".join(formatted_examples)
    
    def build_messages(self, 
                      sequence: str,
                      task: int = 1,
                      approach: str = "regular", 
                      examples: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, str]]:
        """
        Build complete messages for the model (system + user prompt).
        
        Args:
            sequence: Sequence to analyze
            task: Task number (1, 2, or 3) - automatically maps to modality
            approach: 'regular' or 'cbm'
            examples: List of examples for few-shot (None for zero-shot)
        
        Returns:
            List of message dictionaries with 'role' and 'content'
        """
        
        # Validate task
        if task not in self.TASK_TO_MODALITY:
            raise ValueError(f"Invalid task {task}. Use 1, 2, or 3.")
        
        # System prompt
        system_prompt = self.load_prompt(approach, "system_prompt")
        
        # User prompt
        if examples is None or len(examples) == 0:
            # Zero-shot
            user_template = self.load_prompt(approach, "user_prompt_zero_shot")
            user_prompt = user_template.format(sequence=sequence)
        else:
            # Few-shot
            user_template = self.load_prompt(approach, "user_prompt_few_shot")
            formatted_examples = self.format_few_shot_examples(examples, approach)
            user_prompt = user_template.format(
                examples=formatted_examples,
                sequence=sequence
            )
        
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]


class DataFormatter:
    """Utilities for formatting input/output data"""
    
    @staticmethod
    def parse_cbm_response(response: str) -> Optional[Dict[str, Any]]:
        """
        Parse JSON response from model in CBM format.
        
        Expected format:
        {
            "conceptos": {...},
            "fc_clase": "Brady|Normal|Tachy"
        }
        
        Returns:
            Dictionary with 'conceptos' and 'fc_clase', or None if error
        """
        try:
            # Clean response (remove markdown if present)
            cleaned = response.strip()
            if "```json" in cleaned:
                # Extract JSON from code block
                start = cleaned.find("```json") + 7
                end = cleaned.find("```", start)
                cleaned = cleaned[start:end].strip()
            elif "```" in cleaned:
                # Handle blocks without language specification
                start = cleaned.find("```") + 3
                end = cleaned.rfind("```")
                cleaned = cleaned[start:end].strip()
            
            # Parse JSON
            parsed = json.loads(cleaned)
            
            # Validate expected structure
            if "conceptos" not in parsed or "fc_clase" not in parsed:
                return None
            
            return parsed
            
        except (json.JSONDecodeError, ValueError, KeyError):
            return None
    
    @staticmethod
    def parse_regular_response(response: str) -> Optional[str]:
        """
        Parse response from model in regular format (classification only).
        
        Looks for valid class names in the response.
        
        Returns:
            Classification ('Brady', 'Normal', 'Tachy') or None if error
        """
        cleaned = response.strip()
        
        # Search for valid classifications
        valid_classes = ["Brady", "Normal", "Tachy"]
        
        for fc_class in valid_classes:
            if fc_class.lower() in cleaned.lower():
                return fc_class
        
        return None
    
    @staticmethod
    def extract_concepts_from_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract concepts from a sample for few-shot examples.
        
        Args:
            sample: Sample dictionary (should contain 'concepts' or 'sequence')
        
        Returns:
            Concepts dictionary
        """
        if 'concepts' in sample:
            return sample['concepts']
        
        # If not pre-calculated, calculate from sequence
        sequence = sample['sequence']
        
        n_narrow = sequence.count('|')
        n_wide = sequence.count(':')
        n_noise = sequence.count('~') + sequence.count('_')
        noise_pct = (n_noise / len(sequence)) * 100
        
        return {
            'C1_picos_estrechos': n_narrow,
            'C2_picos_anchos': n_wide,
            'C3_pct_ruido': round(noise_pct, 2),
            'C4_irregularidad': 0.05,  # Default value
            'C5_artefactos_significativos': sequence.count('_') > len(sequence) * 0.05
        }

