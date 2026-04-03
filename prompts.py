import pandas as pd

input_file = "test.csv"
output_file = "prompts.txt"

df = pd.read_csv(input_file)

with open(output_file, "w", encoding="utf-8") as f:
    for p in df["prompt"]:
        if pd.isna(p):
            continue
        f.write(str(p).strip() + "\n")

print(f"Done. Saved to {output_file}")