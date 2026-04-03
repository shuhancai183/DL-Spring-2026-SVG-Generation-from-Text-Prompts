import os
import pandas as pd

INPUT_DIR = "output_text"
OUTPUT_FILE = "submission.csv"

def main():
    files = sorted([f for f in os.listdir(INPUT_DIR) if f.endswith(".svg")])

    rows = []
    for i, fname in enumerate(files):
        path = os.path.join(INPUT_DIR, fname)
        with open(path, "r", encoding="utf-8") as f:
            svg = f.read()
        rows.append({"id": str(i), "svg": svg})

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_FILE, index=False)

    print(f"Saved {OUTPUT_FILE}")

if __name__ == "__main__":
    main()