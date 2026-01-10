import os
import glob
import json
from time import sleep
from typing import Dict, Any, List

from dotenv import load_dotenv
from openai import OpenAI
import fitz
import pandas as pd
import argparse

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def extract_text_from_pdf(
    pdf_path: str,
    max_chars: int = 120000,
) -> str:

    doc = fitz.open(pdf_path)
    chunks = []

    for page in doc:
        text = page.get_text("text")
        if text:
            chunks.append(text)

    full_text = "\n\n".join(chunks)
    return full_text[:max_chars]


# ---------------------------------------------------------------------
#  SYSTEM PROMPT FROM EXTERNAL FILE
# ---------------------------------------------------------------------

def load_system_prompt(prompt_path: str = "prompt.txt") -> str:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(base_dir, prompt_path)

    if not os.path.exists(full_path):
        raise FileNotFoundError(f"System prompt file not found: {full_path}")

    with open(full_path, "r", encoding="utf-8") as f:
        return f.read()


SYSTEM_PROMPT = load_system_prompt()


# ---------------------------------------------------------------------
#  CALL OPENAI
# ---------------------------------------------------------------------

def extract_metadata_from_pdf(
    pdf_path: str,
    model: str = "gpt-5-mini",
) -> Dict[str, Any]:

    text = extract_text_from_pdf(pdf_path)

    if not text.strip():
        raise ValueError(f"No text extracted from: {pdf_path}")

    user_prompt = f"""
I will give you text extracted from an APSR article PDF.
Use it to fill in all JSON fields described in the system message.

PDF filename: {os.path.basename(pdf_path)}

EXTRACTED TEXT:
{text}
"""

    response = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
    )

    raw_text = response.output_text
    result = json.loads(raw_text)

    # Add file name for bookkeeping
    result["pdf_filename"] = os.path.basename(pdf_path)
    return result


# ---------------------------------------------------------------------
#  RUN PIPELINE AND SAVE CSV
# ---------------------------------------------------------------------

def run_pipeline(
    input_dir: str,
    output_csv: str,
    model: str = "gpt-5-mini",
    sleep_sec: float = 0.4,
):
    pdf_files = sorted(glob.glob(os.path.join(input_dir, "*.pdf")))
    rows: List[Dict[str, Any]] = []

    if not pdf_files:
        print(f"No PDFs found in {input_dir}")
        return

    print(f"Found {len(pdf_files)} PDF(s) in {input_dir}")
    print(f"Output CSV will be saved to: {output_csv}")

    for i, pdf_path in enumerate(pdf_files, start=1):
        print(f"[{i}/{len(pdf_files)}] Processing {pdf_path} ...")

        try:
            row = extract_metadata_from_pdf(pdf_path, model=model)
        except Exception as e:
            print(f"  ERROR for {pdf_path}: {e}")
            row = {
                "pdf_filename": os.path.basename(pdf_path),
                "error": str(e),
            }

        rows.append(row)
        sleep(sleep_sec)

    # Convert list of dicts -> DataFrame -> CSV
    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    print(f"CSV saved to {output_csv}")


if __name__ == "__main__":
    args = argparse.ArgumentParser(
        description="Extract metadata from APSR article PDFs and save to CSV."
    )
    args.add_argument(
        "--input_dir",
        type=str,
        default="data",
        help="Directory containing PDF files.",
    )
    args.add_argument(
        "--output_csv",
        type=str,
        default="articles_survey_metadata.csv",
        help="Output CSV file path.",
    )
    parsed_args = args.parse_args()
    run_pipeline(parsed_args.input_dir, parsed_args.output_csv, model="gpt-4.1-mini")