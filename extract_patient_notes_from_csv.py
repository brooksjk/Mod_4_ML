import os
import argparse
import pandas as pd
from pathlib import Path
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(
        description="Combine all notes per patient from CSVs and save as .txt files."
    )
    parser.add_argument("-i", "--input-dir", required=True, help="Directory containing input CSV files")
    parser.add_argument( "-o", "--output-dir", required=True, help="Directory to save combined patient .txt files")
    parser.add_argument("--id-col", default=None, help="Optional: name of the column containing patient IDs")
    parser.add_argument("--text-col", default=None, help="Optional: name of the column containing note text")

    args = parser.parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_files = list(input_dir.glob("*.csv"))
    if not csv_files:
        print(f"No CSV files found in {input_dir}")
        return

    for csv_file in csv_files:
        print(f"\nProcessing: {csv_file.name}")
        df = pd.read_csv(csv_file)

        text_col = args.text_col
        if text_col is None:
            for candidate in ["note_text", "text", "document", "notes"]:
                if candidate in df.columns:
                    text_col = candidate
                    break
        if text_col is None:
            print(f"Could not find a text column in {csv_file.name}, skipping.")
            continue

        id_col = args.id_col
        if id_col is None:
            for candidate in ["patient_id", "id", "record_id", "subject_id"]:
                if candidate in df.columns:
                    id_col = candidate
                    break
        if id_col is None:
            print(f"Could not find a patient ID column in {csv_file.name}, skipping.")
            continue

        # Group by patient
        grouped = df.groupby(id_col)[text_col].apply(
            lambda notes: "\n\n---\n\n".join(str(n) for n in notes if pd.notna(n))
        )

        for patient_id, combined_text in tqdm(grouped.items(), total=len(grouped)):
            if not combined_text.strip():
                continue

            safe_id = str(patient_id).replace("/", "_").replace(" ", "_")
            txt_path = output_dir / f"{csv_file.stem}_{safe_id}.txt"
            with open(txt_path, "w") as f:
                f.write(combined_text)

        print(f"Saved {len(grouped)} patient files to {output_dir}")

    print("\nAll CSVs processed successfully.")


if __name__ == "__main__":
    main()

"""
 python3 extract_patient_notes_from_csv.py \
  -i /scratch/jkbrook/NLP/Mod_4_ML/input \
  -o /scratch/jkbrook/NLP/Mod_4_ML/output \
  --id-col patient_id \
  --text-col text
"""