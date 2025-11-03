"""
In-Context Learning runner and evaluation utilities.

This module provides infrastructure to run ICL experiments with local or remote LLMs,
including evaluation metrics and result logging.
"""

from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import csv

from openai import OpenAI

from ..data.loaders import ICLDataLoader
from .prompt_builder import PromptManager, DataFormatter


class LocalLLM:
    """Client for locally served LLMs using OpenAI-compatible API"""
    
    def __init__(self, 
                 base_url: str = "http://127.0.0.1:1234/v1", 
                 model: str = "local-model",
                 temperature: float = 0.1,
                 api_key: str = "dummy-key"):
        """
        Initialize local LLM client.
        
        Args:
            base_url: URL of the local server
            model: Model name (can be anything for local servers)
            temperature: Sampling temperature (low for determinism)
            api_key: API key (many local servers don't require real keys)
        """
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key
        )
        self.model = model
        self.temperature = temperature
    
    def predict(self, messages: List[Dict[str, str]], max_tokens: int = 500) -> str:
        """
        Get prediction from the model.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            max_tokens: Maximum tokens to generate
        
        Returns:
            Model response as string
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=max_tokens,
                timeout=30
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"❌ Error in prediction: {e}")
            raise


class ICLEvaluator:
    """Evaluates ICL performance on classification tasks"""
    
    def __init__(self, 
                 data_loader: ICLDataLoader,
                 prompt_manager: PromptManager,
                 llm: Optional[LocalLLM] = None,
                 output_dir: Optional[Path] = None):
        """
        Initialize evaluator.
        
        Args:
            data_loader: Data loader instance
            prompt_manager: Prompt manager instance
            llm: LLM instance (optional, can be set later)
            output_dir: Directory for saving results
        """
        self.data_loader = data_loader
        self.prompt_manager = prompt_manager
        self.llm = llm
        self.output_dir = Path(output_dir) if output_dir else Path("outputs")
    
    def evaluate(self,
                task: int = 1,
                n_shots: int = 4,
                approach: str = "regular",
                ood: bool = False,
                model_name: str = "local-model",
                verbose: bool = True) -> Tuple[float, Dict[str, Dict[str, int]]]:
        """
        Evaluate ICL on a classification task.
        
        Args:
            task: Task number (1, 2, or 3)
            n_shots: Number of ICL examples (0 for zero-shot)
            approach: 'regular' or 'cbm'
            ood: Use out-of-distribution test set
            model_name: Model name for saving results
            verbose: Print progress information
        
        Returns:
            Tuple of (overall_accuracy, per_class_results)
        """
        if verbose:
            print(f"🧪 Evaluating Task {task} - ICL with {n_shots} shots - Model: {model_name}")
            print(f"   Approach: {approach} | OOD: {ood}")
        
        # Load ICL examples
        if n_shots > 0:
            icl_examples = self.data_loader.load_icl_examples(n_shots)
            # Format for prompt manager
            icl_for_prompts = [
                {'sequence': seq, 'fc_class': fc_class}
                for seq, fc_class in icl_examples
            ]
        else:
            icl_for_prompts = []
        
        # Load test data
        test_data = self.data_loader.load_test_data(ood=ood)
        
        if verbose:
            print(f"\n🔄 Evaluating {len(test_data)} examples...")
        
        # Create output directories
        predictions_dir = self.output_dir / "predictions" / f"task{task}"
        results_dir = self.output_dir / "results" / f"task{task}"
        predictions_dir.mkdir(parents=True, exist_ok=True)
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Storage for predictions
        predictions = []
        
        # Show example prompt once
        show_prompt_example = verbose
        
        # Iterate over test examples
        for i, (test_sequence, true_class) in enumerate(test_data):
            
            # Build prompt
            messages = self.prompt_manager.build_messages(
                sequence=test_sequence,
                task=task,
                approach=approach,
                examples=icl_for_prompts if n_shots > 0 else None
            )
            
            # Show first prompt example
            if show_prompt_example:
                print(f"\n📄 Example prompt sent to model:")
                print("=" * 80)
                print(f"🔹 SYSTEM MESSAGE:")
                print(f"{messages[0]['content'][:500]}...")
                print(f"\n🔹 USER MESSAGE:")
                print(f"{messages[1]['content'][:500]}...")
                print("=" * 80)
                show_prompt_example = False
            
            # Get prediction
            if self.llm:
                raw_response = self.llm.predict(messages)
                
                # Parse response based on approach
                if approach == "regular":
                    predicted_class = DataFormatter.parse_regular_response(raw_response)
                else:  # cbm
                    parsed = DataFormatter.parse_cbm_response(raw_response)
                    predicted_class = parsed.get('fc_clase') if parsed else None
                
                # Fallback for unparseable responses
                if predicted_class is None:
                    predicted_class = "Unknown"
            else:
                raise ValueError("No LLM provided for prediction.")
            
            # Store prediction
            is_correct = (predicted_class == true_class)
            predictions.append({
                'sequence': test_sequence,
                'true_class': true_class,
                'predicted_class': predicted_class,
                'correct': is_correct,
                'raw_response': raw_response if self.llm else ""
            })
            
            # Show some examples
            if verbose and i < 5:
                status = "✅" if is_correct else "❌"
                print(f"  {i+1}. {status} {true_class} → {predicted_class} | {test_sequence[:30]}...")
            
            # Show progress
            if verbose and (i + 1) % 50 == 0:
                print(f"  ⏳ {i + 1}/{len(test_data)}")
        
        # Save predictions
        ood_suffix = "_ood" if ood else ""
        predictions_file = predictions_dir / f"{model_name}_n_shots_{n_shots}{ood_suffix}_results.csv"
        
        if verbose:
            print(f"\n💾 Saving predictions to {predictions_file}...")
        
        with open(predictions_file, 'w', newline='', encoding='utf-8') as f:
            fieldnames = ['sequence', 'true_class', 'predicted_class', 'correct', 'raw_response']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(predictions)
        
        # Calculate metrics
        correct = sum(1 for p in predictions if p['correct'])
        accuracy = correct / len(predictions) if predictions else 0
        
        # Per-class metrics
        results_by_class = {
            'Brady': {'tp': 0, 'total': 0}, 
            'Normal': {'tp': 0, 'total': 0}, 
            'Tachy': {'tp': 0, 'total': 0}
        }
        
        for pred in predictions:
            true_class = pred['true_class']
            if true_class in results_by_class:
                results_by_class[true_class]['total'] += 1
                if pred['correct']:
                    results_by_class[true_class]['tp'] += 1
        
        if verbose:
            print(f"\n📊 Results:")
            print(f"  Overall Accuracy: {accuracy:.3f} ({correct}/{len(predictions)})")
            
            for fc_class in ['Brady', 'Normal', 'Tachy']:
                class_acc = (results_by_class[fc_class]['tp'] / 
                            results_by_class[fc_class]['total'] 
                            if results_by_class[fc_class]['total'] > 0 else 0)
                tp = results_by_class[fc_class]['tp']
                total = results_by_class[fc_class]['total']
                print(f"  {fc_class}: {class_acc:.3f} ({tp}/{total})")
        
        # Save metrics
        metrics_file = results_dir / f"{model_name}_n_shots_{n_shots}{ood_suffix}_metrics.csv"
        
        if verbose:
            print(f"\n💾 Saving metrics to {metrics_file}...")
        
        with open(metrics_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['metric', 'value'])
            writer.writeheader()
            writer.writerow({'metric': 'Overall Accuracy', 'value': accuracy})
            writer.writerow({'metric': 'Correct Predictions', 'value': correct})
            writer.writerow({'metric': 'Total Predictions', 'value': len(predictions)})
            writer.writerow({'metric': 'Task', 'value': task})
            writer.writerow({'metric': 'N-shots', 'value': n_shots})
            writer.writerow({'metric': 'Approach', 'value': approach})
            writer.writerow({'metric': 'OOD', 'value': ood})
            writer.writerow({'metric': 'Model', 'value': model_name})
            writer.writerow({'metric': 'Timestamp', 'value': datetime.now().isoformat()})
            
            for fc_class in ['Brady', 'Normal', 'Tachy']:
                class_acc = (results_by_class[fc_class]['tp'] / 
                            results_by_class[fc_class]['total'] 
                            if results_by_class[fc_class]['total'] > 0 else 0)
                writer.writerow({'metric': f'{fc_class} Accuracy', 'value': class_acc})
                writer.writerow({'metric': f'{fc_class} TP', 'value': results_by_class[fc_class]['tp']})
                writer.writerow({'metric': f'{fc_class} Total', 'value': results_by_class[fc_class]['total']})
        
        if verbose:
            print(f"✅ Evaluation complete!")
        
        return accuracy, results_by_class

