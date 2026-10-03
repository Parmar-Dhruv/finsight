"""
retrieval/download.py
=====================
SEC EDGAR Document Downloader for FinSight RAG Pipeline.

Downloads official 10-K annual filings directly from the SEC EDGAR API
and saves them to data/raw/<TICKER>/<TICKER>_<YEAR>_10K.htm in compliance
with SEC fair-access policies and project conventions.

Usage:
------
# 1. Download NVIDIA 2023 and 2024 10-K filings (default):
python retrieval/download.py

# 2. Download specific ticker and years via CLI:
python retrieval/download.py --ticker NVDA --years 2023 2024
python retrieval/download.py --ticker MSFT --years 2023 2024
"""

import argparse
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
from dotenv import load_dotenv

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Load .env configuration
load_dotenv(PROJECT_ROOT / ".env")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Common Tech / S&P 500 Ticker-to-CIK mapping for instant lookup
KNOWN_CIKS: Dict[str, str] = {
    "NVDA": "0001045810",
    "AAPL": "0000320193",
    "MSFT": "0000789019",
    "GOOGL": "0001652044",
    "GOOG": "0001652044",
    "AMZN": "0001018724",
    "META": "0001326801",
    "TSLA": "0001318605",
    "AMD": "0000002488",
    "INTC": "0000050863",
}


def get_user_agent() -> str:
    """
    Retrieves the SEC User-Agent header from .env.
    SEC EDGAR requires: 'Sample Company Name AdminContact@domain.com'.
    """
    user_agent = os.getenv("SEC_EDGAR_USER_AGENT")
    if not user_agent or "your_email@example.com" in user_agent:
        default_ua = "FinSight Research Project (contact: student@university.edu)"
        logger.warning(
            f"SEC_EDGAR_USER_AGENT not set in .env. Using fallback: '{default_ua}'. "
            "Please update your .env with your name and email."
        )
        return default_ua
    return user_agent.strip()


def resolve_cik(ticker: str, headers: Dict[str, str]) -> str:
    """
    Resolves a ticker symbol to a 10-digit CIK string.
    Checks local dictionary first, then falls back to SEC company_tickers.json.
    """
    ticker_upper = ticker.upper()
    if ticker_upper in KNOWN_CIKS:
        return KNOWN_CIKS[ticker_upper]

    logger.info(f"Looking up CIK for {ticker_upper} via SEC directory...")
    url = "https://www.sec.gov/files/company_tickers.json"
    resp = requests.get(url, headers=headers, timeout=15)
    resp.raise_for_status()

    tickers_data = resp.json()
    for item in tickers_data.values():
        if item.get("ticker", "").upper() == ticker_upper:
            cik_int = item.get("cik_str")
            cik_str = str(cik_int).zfill(10)
            KNOWN_CIKS[ticker_upper] = cik_str
            return cik_str

    raise ValueError(f"Could not find CIK for ticker: '{ticker_upper}'. Please check the ticker symbol.")


def get_filing_metadata(
    cik: str,
    target_years: List[int],
    headers: Dict[str, str],
) -> List[Dict[str, str]]:
    """
    Queries SEC EDGAR submissions API to find primary 10-K documents
    matching the specified calendar/fiscal filing years.
    """
    submissions_url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    logger.info(f"Fetching submissions metadata from SEC EDGAR for CIK {cik}...")
    resp = requests.get(submissions_url, headers=headers, timeout=20)
    resp.raise_for_status()

    data = resp.json()
    recent = data.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    accession_numbers = recent.get("accessionNumber", [])
    primary_docs = recent.get("primaryDocument", [])
    filing_dates = recent.get("filingDate", [])
    report_dates = recent.get("reportDate", [])

    matched_filings = []
    found_years = set()

    for idx, form in enumerate(forms):
        if form == "10-K":
            f_date = filing_dates[idx]  # e.g. "2024-02-21"
            r_date = report_dates[idx]  # e.g. "2024-01-28"

            # Determine filing year (prioritizing report date period, then filing date)
            year_candidate = None
            for d in [r_date, f_date]:
                m = re.match(r"^(\d{4})", d)
                if m:
                    year_candidate = int(m.group(1))
                    break

            if year_candidate in target_years and year_candidate not in found_years:
                acc_num = accession_numbers[idx]
                primary_doc = primary_docs[idx]
                acc_no_hyphen = acc_num.replace("-", "")
                cik_no_leading_zeros = str(int(cik))

                doc_url = (
                    f"https://www.sec.gov/Archives/edgar/data/"
                    f"{cik_no_leading_zeros}/{acc_no_hyphen}/{primary_doc}"
                )

                matched_filings.append({
                    "year": year_candidate,
                    "filing_date": f_date,
                    "report_date": r_date,
                    "accession_number": acc_num,
                    "primary_doc": primary_doc,
                    "url": doc_url,
                })
                found_years.add(year_candidate)

    return matched_filings


def download_sec_document(
    url: str,
    output_path: Path,
    headers: Dict[str, str],
) -> int:
    """
    Downloads document from SEC EDGAR with fair-access rate limit pause.
    Returns file size in bytes.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading from SEC EDGAR: {url}")
    
    resp = requests.get(url, headers=headers, timeout=60)
    resp.raise_for_status()

    content = resp.content
    with open(output_path, "wb") as f:
        f.write(content)

    # SEC enforces max 10 requests per second; 0.2s sleep ensures complete compliance
    time.sleep(0.2)
    return len(content)


def download_filings_for_ticker(
    ticker: str,
    years: List[int],
    output_dir: Optional[Path] = None,
) -> List[Path]:
    """
    Full workflow: resolve CIK -> get 10-K metadata -> download files -> save to data/raw/<TICKER>/.
    """
    ticker_upper = ticker.upper()
    if output_dir is None:
        output_dir = PROJECT_ROOT / "data" / "raw" / ticker_upper

    headers = {"User-Agent": get_user_agent()}
    cik = resolve_cik(ticker_upper, headers)
    logger.info(f"Resolved {ticker_upper} -> CIK: {cik}")

    filings = get_filing_metadata(cik, years, headers)
    if not filings:
        logger.error(f"No 10-K filings found for {ticker_upper} matching years: {years}")
        return []

    downloaded_paths = []
    for f in filings:
        year = f["year"]
        # Save according to FinSight naming convention: <TICKER>_<YEAR>_10K.htm
        target_file = output_dir / f"{ticker_upper}_{year}_10K.htm"
        logger.info(f"Fetching {ticker_upper} FY{year} 10-K (filed {f['filing_date']})...")
        size = download_sec_document(f["url"], target_file, headers)
        logger.info(f"Saved: {target_file} ({size:,} bytes)")
        downloaded_paths.append(target_file)

    return downloaded_paths


def main():
    parser = argparse.ArgumentParser(description="Download SEC 10-K filings for FinSight RAG Pipeline.")
    parser.add_argument(
        "--ticker",
        type=str,
        default="NVDA",
        help="Stock ticker symbol (e.g. NVDA, AAPL, MSFT). Default: NVDA",
    )
    parser.add_argument(
        "--years",
        type=int,
        nargs="+",
        default=[2023, 2024],
        help="Fiscal/reporting years to download (e.g. 2023 2024). Default: 2023 2024",
    )
    args = parser.parse_args()

    print("=" * 70)
    print(f"FinSight SEC 10-K Downloader — Target: {args.ticker.upper()} (Years: {args.years})")
    print("=" * 70)

    downloaded = download_filings_for_ticker(args.ticker, args.years)

    print("\n" + "=" * 70)
    print(f"DOWNLOAD COMPLETE: {len(downloaded)} files downloaded successfully.")
    for p in downloaded:
        print(f" - {p} ({p.stat().st_size:,} bytes)")
    print("=" * 70)
    print("Next step: Run `python retrieval/parse.py` to extract Item 1, 1A, 7, 8 sections.")


if __name__ == "__main__":
    main()
