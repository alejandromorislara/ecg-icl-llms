"""
Evaluation script for ICL experiments.

Supports evaluation on different tasks with configurable n-shots and models.
"""

from pathlib import Path
import sys
import argparse

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.loaders import ICLDataLoader
from src.icl.prompt_builder import PromptManager
from src.icl.icl_runner import LocalLLM, ICLEvaluator


def run_icl_evaluation(
    task: int = 1,
    n_shots: int = 4,
    approach: str = "regular",
    model_name: str = "local-model",
    base_url: str = "http://127.0.0.1:1234/v1",
    temperature: float = 0.1,
    data_dir: str = "data/processed/toy_experiment",
    output_dir: str = "outputs",
    ood: bool = False
):
    """
    Run ICL evaluation on a specific task.
    
    Args:
        task: Task number (1, 2, or 3)
        n_shots: Number of ICL examples (0 for zero-shot)
        approach: 'regular' or 'cbm'
        model_name: Model name for results
        base_url: URL of local LLM server
        temperature: Sampling temperature
        data_dir: Directory containing task data
        output_dir: Directory for saving results
        ood: Use out-of-distribution test set
    """
    
    task_names = {
        1: "Symbolic sequences (toy experiment)",
        2: "1D rendered signals (future)",
        3: "Real ECG images (future)"
    }
    
    if task not in task_names:
        print(f"❌ Invalid task {task}. Use 1, 2, or 3.")
        return
    
    print("=" * 70)
    print(f"🧪 ICL Evaluation")
    print("=" * 70)
    print(f"Task: {task} - {task_names[task]}")
    print(f"N-shots: {n_shots}")
    print(f"Approach: {approach}")
    print(f"Model: {model_name}")
    print(f"OOD Test Set: {ood}")
    print("=" * 70)
    
    # Setup paths
    data_path = project_root / data_dir
    output_path = project_root / output_dir
    
    # Check if data exists
    if not data_path.exists():
        print(f"\n❌ Data directory not found: {data_path}")
        print(f"   Please run: python scripts/generate_toy_dataset.py")
        return
    
    # Initialize components
    print(f"\n📦 Initializing components...")
    data_loader = ICLDataLoader(data_dir=data_path)
    
    # Determine prompts directory based on task
    prompts_dir = project_root / "src" / "prompts" / f"task{task}"
    if not prompts_dir.exists():
        print(f"\n❌ Prompts directory not found: {prompts_dir}")
        print(f"   Task {task} may not be implemented yet.")
        return
    
    prompt_manager = PromptManager(prompts_dir=prompts_dir)
    
    # Initialize LLM
    print(f"🔌 Connecting to LLM at {base_url}...")
    try:
        llm = LocalLLM(
            base_url=base_url,
            model=model_name,
            temperature=temperature
        )
    except Exception as e:
        print(f"\n❌ Failed to connect to LLM: {e}")
        print(f"   Make sure your local LLM server is running at {base_url}")
        print(f"   You can use LM Studio, llama.cpp, or similar tools.")
        return
    
    # Initialize evaluator
    evaluator = ICLEvaluator(
        data_loader=data_loader,
        prompt_manager=prompt_manager,
        llm=llm,
        output_dir=output_path
    )
    
    # Run evaluation
    print(f"\n🚀 Starting evaluation...\n")
    try:
        accuracy, results_by_class = evaluator.evaluate(
            task=task,
            n_shots=n_shots,
            approach=approach,
            ood=ood,
            model_name=model_name,
            verbose=True
        )
        
        print("\n" + "=" * 70)
        print("✅ Evaluation completed successfully!")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main function with argument parsing"""
    
    parser = argparse.ArgumentParser(
        description="Evaluate In-Context Learning on ECG analysis tasks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Zero-shot evaluation
  python scripts/evaluate.py --task 1 --n-shots 0
  
  # 4-shot evaluation (default)
  python scripts/evaluate.py --task 1 --n-shots 4
  
  # 8-shot with custom model
  python scripts/evaluate.py --task 1 --n-shots 8 --model-name "medgemma-2b"
  
  # Evaluate on OOD test set
  python scripts/evaluate.py --task 1 --n-shots 4 --ood
  
  # CBM approach
  python scripts/evaluate.py --task 1 --n-shots 4 --approach cbm
        """
    )
    
    parser.add_argument(
        "--task",
        type=int,
        default=1,
        choices=[1, 2, 3],
        help="Task number: 1 (symbolic), 2 (1D signals), 3 (real ECGs)"
    )
    
    parser.add_argument(
        "--n-shots",
        type=int,
        default=4,
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
        "--model-name",
        type=str,
        default="local-model",
        help="Model name for results tracking"
    )
    
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://127.0.0.1:1234/v1",
        help="Base URL of local LLM server (OpenAI-compatible API)"
    )
    
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Sampling temperature (lower = more deterministic)"
    )
    
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/processed/toy_experiment",
        help="Data directory containing icl/ and test/ subdirectories"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Output directory for results"
    )
    
    parser.add_argument(
        "--ood",
        action="store_true",
        help="Use out-of-distribution test set"
    )
    
    args = parser.parse_args()
    
    run_icl_evaluation(
        task=args.task,
        n_shots=args.n_shots,
        approach=args.approach,
        model_name=args.model_name,
        base_url=args.base_url,
        temperature=args.temperature,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        ood=args.ood
    )


if __name__ == "__main__":
    main()

