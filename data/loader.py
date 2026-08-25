"""
Data loader module - loads PDFs with authority metadata and Excel structured data.
"""
import os
import pandas as pd
from pypdf import PdfReader
from pathlib import Path

DATA_DIR = Path(__file__).parent / "documents"

DOCUMENT_REGISTRY = [
    {
        "filename": "01_Support_Policy_v3_CURRENT.pdf",
        "doc_id": "support_policy_v3",
        "title": "ParcelPilot Support Policy v3",
        "status": "CURRENT",
        "doc_type": "policy",
        "authority": 2,  # 1=agreement(highest), 2=current policy, 3=product docs, 4=historical
        "effective_date": "2026-05-01",
        "notes": "Defines default severity and response targets. Supersedes v2.",
    },
    {
        "filename": "02_Support_Policy_v2_DEPRECATED.pdf",
        "doc_id": "support_policy_v2",
        "title": "ParcelPilot Support Policy v2 (DEPRECATED)",
        "status": "DEPRECATED",
        "doc_type": "policy",
        "authority": 99,  # Should never be used for current decisions
        "effective_date": "2025-01-01",
        "notes": "DEPRECATED. Retained for historical reference only. Do NOT use for current requests.",
    },
    {
        "filename": "03_Cancellation_and_Service_Credit_SOP_v4.pdf",
        "doc_id": "cancellation_sop_v4",
        "title": "Cancellation & Service Credit SOP v4",
        "status": "CURRENT",
        "doc_type": "sop",
        "authority": 2,
        "effective_date": "2026-06-15",
        "notes": "Cancellation fees, service credit eligibility, approval thresholds.",
    },
    {
        "filename": "04_Product_Operations_Guide_and_Known_Issues.pdf",
        "doc_id": "product_ops_guide",
        "title": "Product Operations Guide & Known Issues",
        "status": "CURRENT",
        "doc_type": "product_documentation",
        "authority": 3,
        "effective_date": "2026-08-14",
        "notes": "Plan capabilities, known issues (KI-208, KI-211), resolved issues.",
    },
    {
        "filename": "05_Northstar_Logistics_Enterprise_Agreement.pdf",
        "doc_id": "northstar_agreement",
        "title": "Northstar Logistics Enterprise Agreement",
        "status": "CURRENT",
        "doc_type": "customer_agreement",
        "authority": 1,  # Highest - overrides default policy
        "account_id": "ACCT-001",
        "effective_date": "2026-01-01",
        "notes": "Custom SLA, no cancellation fee for BOOKED shipments, INR 5000 monthly credit cap.",
    },
    {
        "filename": "06_LumenWorks_Service_Agreement.pdf",
        "doc_id": "lumenworks_agreement",
        "title": "LumenWorks Service Agreement",
        "status": "CURRENT",
        "doc_type": "customer_agreement",
        "authority": 1,
        "account_id": "ACCT-002",
        "effective_date": "2026-03-01",
        "notes": "4-hour pickup delay threshold, fixed INR 300 credit, no cancellation waiver.",
    },
]


def load_pdf_text(filename: str) -> str:
    filepath = DATA_DIR / filename
    reader = PdfReader(str(filepath))
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text.strip()


def load_all_documents() -> list[dict]:
    """Load all documents with their text and metadata."""
    documents = []
    for doc_meta in DOCUMENT_REGISTRY:
        text = load_pdf_text(doc_meta["filename"])
        documents.append({**doc_meta, "content": text})
    return documents


def load_structured_data() -> dict[str, pd.DataFrame]:
    """Load Excel workbook into DataFrames."""
    xlsx_path = DATA_DIR / "ParcelPilot_Assessment_Data.xlsx"
    return {
        "accounts": pd.read_excel(xlsx_path, sheet_name="accounts"),
        "orders": pd.read_excel(xlsx_path, sheet_name="orders"),
        "tickets": pd.read_excel(xlsx_path, sheet_name="tickets"),
    }


SNAPSHOT_TIME = "2026-08-16 11:00 Asia/Kolkata"
