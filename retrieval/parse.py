"""
retrieval/parse.py
==================
Module for parsing raw SEC EDGAR 10-K filings (.htm / .html).

Key Responsibilities:
---------------------
1. HTML Cleaning: Strip scripts, styling, and noise from SEC EDGAR documents.
2. Table Preservation: Convert HTML <table> structures into Markdown format
   so embedding models and LLMs keep column/row relationships intact.
3. Section Extraction: Extract core 10-K sections:
   - Item 1  : Business
   - Item 1A : Risk Factors
   - Item 7  : Management's Discussion and Analysis (MD&A)
   - Item 8  : Financial Statements and Supplementary Data
4. Structured Output: Save parsed text + tables as structured JSON in data/processed/.
"""

import json
import os
import re
from pathlib import Path
from bs4 import BeautifulSoup, Tag


# ─── Part 1: Table Preservation & HTML Cleaning ────────────────────────────────

def table_to_markdown(table_tag: Tag) -> str:
    """
    Converts an HTML <table> Tag into a clean Markdown table string.
    Preserves numeric and tabular relationships for RAG embeddings.
    """
    rows = []
    
    # Extract all <tr> (table row) elements
    for tr in table_tag.find_all("tr"):
        row_cells = []
        
        for cell in tr.find_all(["th", "td"]):
            # Extract cell text, clean non-breaking spaces and whitespace
            cell_text = cell.get_text(separator=" ", strip=True)
            cell_text = cell_text.replace("\xa0", " ").replace("\n", " ")
            cell_text = re.sub(r"\s+", " ", cell_text).strip()
            
            # Escape pipe '|' characters to avoid breaking Markdown table structure
            cell_text = cell_text.replace("|", "/")
            row_cells.append(cell_text)
            
        if any(row_cells):
            rows.append(row_cells)
            
    if not rows:
        return ""
        
    max_cols = max(len(r) for r in rows)
    if max_cols == 0:
        return ""
        
    # Standardize all rows to have exactly `max_cols` cells
    padded_rows = [r + [""] * (max_cols - len(r)) for r in rows]
    
    # Identify and drop columns that are empty across EVERY row (SEC HTML padding)
    non_empty_col_indices = [
        c for c in range(max_cols) if any(padded_rows[r][c] != "" for r in range(len(padded_rows)))
    ]
    if not non_empty_col_indices:
        return ""
        
    pruned_rows = [[r[c] for c in non_empty_col_indices] for r in padded_rows]
    
    # Merge standalone currency symbols with the adjacent number
    cleaned_rows = []
    for r in pruned_rows:
        new_row = []
        skip_next = False
        for i in range(len(r)):
            if skip_next:
                skip_next = False
                continue
            if r[i] in ["$", "€", "£"] and i + 1 < len(r) and r[i + 1] != "":
                new_row.append(f"{r[i]}{r[i + 1]}")
                skip_next = True
            else:
                new_row.append(r[i])
        cleaned_rows.append(new_row)
        
    num_cols = max(len(r) for r in cleaned_rows)
    standard_rows = [r + [""] * (num_cols - len(r)) for r in cleaned_rows]
    
    # Build Markdown table
    header_row = "| " + " | ".join(standard_rows[0]) + " |"
    delimiter_row = "| " + " | ".join(["---"] * num_cols) + " |"
    data_rows = ["| " + " | ".join(r) + " |" for r in standard_rows[1:]]
    
    markdown_lines = [header_row, delimiter_row] + data_rows
    return "\n" + "\n".join(markdown_lines) + "\n"


def clean_html(html_content: str) -> BeautifulSoup:
    """
    Parses raw HTML and removes unwanted script, style, and metadata tags.
    """
    soup = BeautifulSoup(html_content, "html.parser")
    for tag in soup(["script", "style", "noscript", "meta"]):
        tag.decompose()
    return soup


def convert_all_tables_to_markdown(soup: BeautifulSoup) -> None:
    """
    Replaces every <table> node inside the BeautifulSoup tree with its
    Markdown string representation in-place.
    """
    for table in soup.find_all("table"):
        md_table = table_to_markdown(table)
        table.replace_with(soup.new_string(f"\n\n{md_table}\n\n"))


# ─── Part 2: 10-K Section Boundary Extraction ──────────────────────────────────

def extract_sections(text: str) -> dict[str, str]:
    """
    Extracts key 10-K sections from parsed filing text.
    Handles Table of Contents (TOC) ambiguity by looking for substantial narrative text.
    
    Target Sections:
    - Item 1  : Business
    - Item 1A : Risk Factors
    - Item 7  : Management's Discussion and Analysis (MD&A)
    - Item 8  : Financial Statements and Supplementary Data
    """
    # Patterns for section start headers (flexible for spacing/kerning in SEC HTML)
    start_patterns = {
        "item_1": r"Item\s+1[\.\:\s\|]+B\s*usiness",
        "item_1a": r"Item\s+1A[\.\:\s\|]+Ris\s*k\s+Factors",
        "item_7": r"Item\s+7[\.\:\s\|]+Management",
        "item_8": r"Item\s+8[\.\:\s\|]+Financial\s+Stat\s*e?\s*ments",
    }
    
    # Patterns for subsequent sections that mark the boundary end
    end_markers = {
        "item_1": r"Item\s+1A[\.\:\s\|]+Ris\s*k",
        "item_1a": r"Item\s+(1B|1C|2)[\.\:\s\|]",
        "item_7": r"Item\s+(7A|8)[\.\:\s\|]",
        "item_8": r"Item\s+9[\.\:\s\|]",
    }
    
    def find_body_match(pattern: str, search_from: int = 0) -> int | None:
        """Finds the section start in the body, bypassing Table of Contents listings."""
        matches = list(re.finditer(pattern, text[search_from:], re.IGNORECASE))
        if not matches:
            return None
            
        for m in matches:
            pos = search_from + m.start()
            following_snippet = text[pos:pos + 400]
            # Table of Contents entries list multiple items in rapid succession (< 400 chars)
            if len(re.findall(r"Item\s+\d", following_snippet, re.IGNORECASE)) <= 1:
                return pos
                
        # If all matches have rapid item listings, take the last occurrence
        return search_from + matches[-1].start()

    extracted = {}
    
    for sec_key, start_pat in start_patterns.items():
        start_pos = find_body_match(start_pat)
        if start_pos is None:
            extracted[sec_key] = ""
            continue
            
        end_pat = end_markers[sec_key]
        end_pos = find_body_match(end_pat, search_from=start_pos + 100)
        
        if end_pos is not None and end_pos > start_pos:
            sec_text = text[start_pos:end_pos].strip()
        else:
            # Fallback: take the next 60,000 characters if no clean end marker found
            sec_text = text[start_pos:start_pos + 60000].strip()
            
        extracted[sec_key] = sec_text
        
    return extracted


# ─── Part 3: Pipeline Runner & JSON Storage ────────────────────────────────────

def parse_filing(file_path: Path) -> dict:
    """
    Parses a single 10-K filing file into structured sections and metadata.
    """
    # Derive ticker and fiscal year from parent folder or filename
    # e.g., data/raw/AAPL/AAPL_2024_10K.htm -> ticker='AAPL', year=2024
    ticker = file_path.parent.name
    year_match = re.search(r"(20\d\d)", file_path.name)
    fiscal_year = int(year_match.group(1)) if year_match else 0
    
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        html_content = f.read()
        
    # 1. Clean HTML noise
    soup = clean_html(html_content)
    
    # 2. In-place convert tables to Markdown
    convert_all_tables_to_markdown(soup)
    
    # 3. Extract text and normalize spacing
    text = soup.get_text(separator="\n")
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    
    # 4. Segment into core 10-K sections
    sections = extract_sections(text)
    
    return {
        "ticker": ticker,
        "fiscal_year": fiscal_year,
        "source_file": file_path.name,
        "sections": sections,
        "character_counts": {k: len(v) for k, v in sections.items()},
    }


def parse_all_filings(raw_dir: Path, output_dir: Path) -> list[Path]:
    """
    Scans data/raw/ for all .htm and .html files, parses each, and saves
    structured JSON files into data/processed/.
    """
    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    generated_files = []
    
    # Search for all .htm and .html files in data/raw/
    filing_paths = sorted(list(raw_dir.glob("*/*.htm")) + list(raw_dir.glob("*/*.html")))
    
    print(f"\n[FinSight Parser] Found {len(filing_paths)} raw filings to process.\n")
    
    for path in filing_paths:
        print(f"--> Parsing: {path.parent.name}/{path.name} ...", end=" ", flush=True)
        result = parse_filing(path)
        
        output_filename = f"{result['ticker']}_{result['fiscal_year']}_parsed.json"
        output_path = output_dir / output_filename
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
            
        generated_files.append(output_path)
        counts = result["character_counts"]
        print(f"DONE | Item 1: {counts['item_1']:,}c | Item 7: {counts['item_7']:,}c | Item 8: {counts['item_8']:,}c")
        
    return generated_files


if __name__ == "__main__":
    # Standard entry point when run directly: python retrieval/parse.py
    project_root = Path(__file__).resolve().parent.parent
    raw_directory = project_root / "data" / "raw"
    processed_directory = project_root / "data" / "processed"
    
    output_files = parse_all_filings(raw_directory, processed_directory)
    print(f"\n[FinSight Parser] Successfully parsed and saved {len(output_files)} filings to {processed_directory}")
