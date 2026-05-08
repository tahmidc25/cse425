#!/usr/bin/env python3
"""
Main execution script for Music Generation project.
Orchestrates: data preprocessing → training → evaluation → reporting
"""

import os
import sys
import argparse
from pathlib import Path
import subprocess
import json


class MusicGenerationPipeline:
    """Orchestrates the complete music generation pipeline."""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.data_dir = self.project_root / "data"
        self.src_dir = self.project_root / "src"
        self.output_dir = self.project_root / "outputs"
        
    def run_preprocessing(self):
        """Run preprocessing pipeline."""
        print("\n" + "="*70)
        print("PHASE: DATA PREPROCESSING")
        print("="*70)
        
        script = self.src_dir / "preprocessing" / "midi_parser.py"
        if script.exists():
            result = subprocess.run(
                [sys.executable, "-u", str(script)],
                cwd=self.project_root,
                capture_output=False,
                text=True
            )
            if result.returncode != 0:
                print(f"Error in preprocessing")
                return False
            return True
        else:
            print(f"Warning: Preprocessing script not found at {script}")
            return False
    
    def run_task_training(self, task: int, epochs: int = 50):
        """Run training for a specific task."""
        print("\n" + "="*70)
        print(f"TASK {task}: TRAINING")
        print("="*70)
        
        task_scripts = {
            1: "train_ae.py",
            2: "train_vae.py",
            3: "train_transformer.py",
            4: "train_rlhf.py"
        }
        
        if task not in task_scripts:
            print(f"Error: Task {task} not recognized")
            return False
        
        if task == 1 and (self.output_dir / "plots" / "loss_curves" / "task1_loss.png").exists():
            print("Task 1 already completed. Skipping...")
            return True
        if task == 2 and (self.output_dir / "plots" / "loss_curves" / "task2_loss.png").exists():
            print("Task 2 already completed. Skipping...")
            return True
        if task == 3 and (self.output_dir / "plots" / "loss_curves" / "task3_loss.png").exists():
            print("Task 3 already completed. Skipping...")
            return True

        script = self.src_dir / "training" / task_scripts[task]
        if script.exists():
            result = subprocess.run(
                [sys.executable, "-u", str(script)],
                cwd=self.project_root,
                capture_output=False,
                text=True
            )
            if result.returncode != 0:
                print(f"Error in preprocessing")
                return False
            return True
        else:
            print(f"Warning: Training script not found at {script}")
            return False
    
    def run_evaluation(self):
        """Run comprehensive evaluation."""
        print("\n" + "="*70)
        print("PHASE: EVALUATION & METRICS")
        print("="*70)
        
        script = self.src_dir / "evaluation" / "run_evaluation.py"
        if script.exists():
            result = subprocess.run(
                [sys.executable, "-u", str(script)],
                cwd=self.project_root,
                capture_output=False,
                text=True
            )
            if result.returncode != 0:
                print(f"Error in preprocessing")
                return False
            return True
        else:
            print(f"Warning: Evaluation script not found at {script}")
            return False
    
    def verify_outputs(self):
        """Verify all required outputs exist."""
        print("\n" + "="*70)
        print("VERIFICATION: Checking outputs")
        print("="*70)
        
        checks = {
            "Task 1 MIDI samples": self.output_dir / "generated_midis" / "task1",
            "Task 2 MIDI samples": self.output_dir / "generated_midis" / "task2",
            "Task 3 MIDI samples": self.output_dir / "generated_midis" / "task3",
            "Task 4 MIDI samples": self.output_dir / "generated_midis" / "task4",
            "Loss curves": self.output_dir / "plots" / "loss_curves",
            "Comparison table": self.output_dir / "results" / "comparison_table.csv",
            "Report": self.project_root / "report" / "final_report.tex"
        }
        
        all_exist = True
        for name, path in checks.items():
            if path.exists():
                print(f"[OK] {name}: {path}")
            else:
                print(f"[FAIL] {name}: {path} (MISSING)")
                all_exist = False
        
        return all_exist
    
    def create_archive(self, output_file: str = "music_generation_submission.zip"):
        """Create submission-ready archive."""
        print("\n" + "="*70)
        print("CREATING SUBMISSION ARCHIVE")
        print("="*70)
        
        import shutil
        
        # Create zip
        archive_path = self.project_root / output_file
        shutil.make_archive(
            str(archive_path.with_suffix('')),
            'zip',
            self.project_root
        )
        
        print(f"Created archive: {archive_path}")
        return archive_path
    
    def run_dummy_pipeline(self):
        """Run complete pipeline with dummy data (fast test)."""
        print("\n" + "#"*70)
        print("# QUICK TEST WITH DUMMY DATA (No datasets required)")
        print("#"*70)
        
        # Create dummy data directory structure
        dummy_data_dir = self.data_dir / "dummy"
        dummy_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Step 1: Train all tasks with dummy data
        for task in [1, 2, 3, 4]:
            print("\n" + "="*70)
            print(f"TASK {task}: DUMMY DATA TRAINING (Fast test)")
            print("="*70)
            
            task_scripts = {
                1: "train_ae.py",
                2: "train_vae.py",
                3: "train_transformer.py",
                4: "train_rlhf.py"
            }
            
            script = self.src_dir / "training" / task_scripts[task]
            if script.exists():
                # Run with dummy data (scripts have create_dummy_dataset functions)
                result = subprocess.run(
                    [sys.executable, str(script)],
                    cwd=self.project_root,
                    capture_output=False,  # Show output in real-time
                    text=True,
                    env={**os.environ, "DUMMY_DATA": "1"}
                )
                if result.returncode != 0:
                    print(f"Warning: Task {task} dummy training had issues")
        
        # Step 2: Verification
        print("\n" + "="*70)
        print("QUICK TEST COMPLETE!")
        print("="*70)
        print("\nGenerated outputs in:")
        print(f"  - MIDI files: {self.output_dir / 'generated_midis'}")
        print(f"  - Plots: {self.output_dir / 'plots'}")
        print(f"  - Results: {self.output_dir / 'results'}")
        
        return True
    
    def run_full_pipeline(self, skip_preprocessing=False, skip_training=False):
        """Run complete pipeline."""
        print("\n" + "#"*70)
        print("# MUSIC GENERATION UNSUPERVISED - FULL PIPELINE")
        print("#"*70)
        
        # Step 1: Preprocessing
        if not skip_preprocessing:
            if not self.run_preprocessing():
                print("Preprocessing failed. Stopping.")
                return False
        
        # Step 2-5: Train tasks
        if not skip_training:
            for task in [1, 2, 3, 4]:
                if not self.run_task_training(task):
                    print(f"Task {task} training failed. Stopping.")
                    return False
        
        # Step 6: Evaluation
        if not self.run_evaluation():
            print("Evaluation failed. Stopping.")
            return False
        
        # Step 7: Verification
        if not self.verify_outputs():
            print("Warning: Some outputs are missing")
        
        print("\n" + "="*70)
        print("PIPELINE COMPLETE!")
        print("="*70)
        
        return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Music Generation Unsupervised - Pipeline Orchestrator"
    )
    parser.add_argument(
        "--task",
        type=int,
        choices=[1, 2, 3, 4],
        help="Run specific task training only"
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run full pipeline (preprocessing + all tasks + evaluation)"
    )
    parser.add_argument(
        "--preprocess-only",
        action="store_true",
        help="Run preprocessing only"
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Run evaluation only"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify outputs"
    )
    parser.add_argument(
        "--archive",
        action="store_true",
        help="Create submission archive"
    )
    parser.add_argument(
        "--dummy",
        action="store_true",
        help="Run quick test with dummy data (no datasets required)"
    )
    parser.add_argument(
        "--project-root",
        default=".",
        help="Project root directory"
    )
    parser.add_argument(
        "--skip-preprocessing",
        action="store_true",
        help="Skip the preprocessing step"
    )
    
    args = parser.parse_args()
    
    pipeline = MusicGenerationPipeline(project_root=args.project_root)
    
    if args.dummy:
        success = pipeline.run_dummy_pipeline()
        sys.exit(0 if success else 1)
    
    if args.full:
        success = pipeline.run_full_pipeline(skip_preprocessing=args.skip_preprocessing)
        sys.exit(0 if success else 1)
    
    if args.preprocess_only:
        success = pipeline.run_preprocessing()
        sys.exit(0 if success else 1)
    
    if args.task:
        success = pipeline.run_task_training(args.task)
        sys.exit(0 if success else 1)
    
    if args.eval_only:
        success = pipeline.run_evaluation()
        sys.exit(0 if success else 1)
    
    if args.verify:
        success = pipeline.verify_outputs()
        sys.exit(0 if success else 1)
    
    if args.archive:
        pipeline.create_archive()
        sys.exit(0)
    
    # Default: interactive menu
    while True:
        print("\n" + "="*70)
        print("MUSIC GENERATION PIPELINE - MAIN MENU")
        print("="*70)
        print("1. Run full pipeline")
        print("2. Run preprocessing only")
        print("3. Run specific task training")
        print("4. Run evaluation")
        print("5. Verify outputs")
        print("6. Create submission archive")
        print("7. Quick test with dummy data (no datasets required)")
        print("8. Exit")
        print("-" * 70)
        
        choice = input("Select option (1-8): ").strip()
        
        if choice == '1':
            pipeline.run_full_pipeline()
        elif choice == '2':
            pipeline.run_preprocessing()
        elif choice == '3':
            task = int(input("Enter task number (1-4): "))
            pipeline.run_task_training(task)
        elif choice == '4':
            pipeline.run_evaluation()
        elif choice == '5':
            pipeline.verify_outputs()
        elif choice == '6':
            pipeline.create_archive()
        elif choice == '7':
            pipeline.run_dummy_pipeline()
        elif choice == '8':
            print("Exiting...")
            break
        else:
            print("Invalid choice")


if __name__ == "__main__":
    main()
