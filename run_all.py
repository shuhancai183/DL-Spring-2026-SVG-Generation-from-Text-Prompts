import os
import subprocess

def run_cmd(cmd):
    print(f"\n[RUN] {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {cmd}")

def main():
    # =========================
    # Step 0
    # =========================
    TEST_CSV = "test.csv"
    TRAIN_CSV = "train.csv"
    OUTPUT_DIR = "final_results"

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # =========================
    # Step 1
    # =========================
    print("\n===== Step 1: Extract prompts =====")
    run_cmd("python prompts.py")

    # =========================
    # Step 2
    # =========================
    print("\n===== Step 2: OmniSVG Inference =====")
    run_cmd(
        "python inference.py "
        "--task text-to-svg "
        "--input prompts.txt "
        "--output output_text "
        "--weight-path ./models/OmniSVG1.1_4B"
        "--model-path ./models/Qwen2.5-VL-3B-Instruct"
        "--model-size 4B"
    )

    # =========================
    # Step 3
    # =========================
    print("\n===== Step 3: Build submission.csv =====")
    run_cmd("python build_submission.py")

    # =========================
    # Step 4
    # =========================
    print("\n===== Step 4: Retrieval Replacement =====")
    run_cmd(
        f"python replace_submission_by_prompt_similarity.py "
        f"--submission_csv submission.csv "
        f"--test_csv {TEST_CSV} "
        f"--train_csv {TRAIN_CSV} "
        f"--output_dir {OUTPUT_DIR} "
        f"--thresholds 0.98"
    )

    print("\n===== DONE =====")
    print(f"Final result saved in: {OUTPUT_DIR}/submission_replace_th_0.98.csv")

if __name__ == "__main__":
    main()