import os
import json
import zipfile
import pdfplumber
from datetime import datetime

ZIP_PATH = "testing/papers.zip"
EXTRACT_DIR = "testing/papers"
OUTPUT_FILE = "testing/dataset.json"


def log(message):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")


def unzip_if_needed():
    if not os.path.exists(EXTRACT_DIR):
        log("Extracting ZIP file...")
        with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(EXTRACT_DIR)
        log("Extraction complete.")
    else:
        log("Papers folder already exists. Skipping unzip.")


def extract_text_from_pdf(path):
    text = ""
    try:
        with pdfplumber.open(path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        log(f"Error reading {path}: {e}")
    return text.strip()


def main():
    unzip_if_needed()

    dataset = []
    files = sorted(os.listdir(EXTRACT_DIR))
    pdf_files = [f for f in files if f.endswith(".pdf")]

    log(f"Found {len(pdf_files)} PDF files.")

    for idx, filename in enumerate(pdf_files):
        path = os.path.join(EXTRACT_DIR, filename)

        log(f"Processing ({idx+1}/{len(pdf_files)}): {filename}")

        text = extract_text_from_pdf(path)

        if not text:
            log(f"WARNING: No text extracted from {filename}")
            continue

        dataset.append({
            "doc_id": f"doc_{idx+1:03d}",
            "filename": filename,
            "text": text
        })

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2)

    log(f"Dataset saved to {OUTPUT_FILE}")
    log(f"Total documents processed: {len(dataset)}")


if __name__ == "__main__":
    main()