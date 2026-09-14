import os
import re

import chromadb
import pdfplumber
from fastembed import TextEmbedding

CHUNK_SIZE = 300    # words per chunk
OVERLAP = 50        # word overlap between chunks
# Same model as the original design (sentence-transformers/all-MiniLM-L6-v2,
# 384-dim), served through fastembed's ONNX runtime instead of full PyTorch.
# This keeps the documented embedding choice but avoids pulling in torch +
# ~3GB of CUDA runtime libraries for what is pure CPU inference in a
# container — see EXPLANATION.md for the original model rationale.
EMBED_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
# Anchored to the project root so this always writes to the same path
# rag/retriever.py reads from, regardless of the process's cwd when this
# script is invoked.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(PROJECT_ROOT, "rag_db")
DOCS_DIR = os.path.join(PROJECT_ROOT, "docs")
COLLECTION = "loan_regulations"

# Every source document in the knowledge base, with the metadata every chunk
# from it must carry. loan_program drives program-aware retrieval (see
# rag/retriever.py) — "underwriting" chunks are only ever retrieved for a
# matching program; "general" material (fair-lending, reporting) supplements
# any program. See REPORT.md / the ingestion report for exact source URLs
# and how each document was obtained.
DOCUMENTS = [
    {
        "path": "fha_handbook.pdf",
        "loan_program": "fha",
        "issuer": "U.S. Department of Housing and Urban Development (HUD)",
        "document": "HUD Handbook 4000.1 — Single Family Housing Policy Handbook",
        "document_type": "underwriting",
        "source_url": "https://www.hud.gov/hud-partners/single-family-handbook-4000-1",
        "source_format": "pdf",
        "content_scope": "partial",
    },
    {
        "path": "hmda_guidelines.pdf",
        "loan_program": "general",
        "issuer": "Consumer Financial Protection Bureau (CFPB) / FFIEC",
        "document": "HMDA Filing Instructions Guide",
        "document_type": "reporting",
        "source_url": "https://ffiec.cfpb.gov/documentation/",
        "source_format": "pdf",
        "content_scope": "full",
    },
    {
        "path": "fannie_mae_selling_guide_excerpts.txt",
        "loan_program": "conventional",
        "issuer": "Fannie Mae",
        "document": "Fannie Mae Selling Guide",
        "document_type": "underwriting",
        "source_url": "https://selling-guide.fanniemae.com/",
        "source_format": "extracted_text",
        "content_scope": "partial",
    },
    {
        "path": "va_pamphlet_26_7_ch04_credit_underwriting.txt",
        "loan_program": "va",
        "issuer": "U.S. Department of Veterans Affairs (VA), Veterans Benefits Administration",
        "document": "VA Pamphlet 26-7 — Lender's Handbook, Chapter 4: Credit Underwriting",
        "document_type": "underwriting",
        "source_url": "https://www.knowva.ebenefits.va.gov/system/templates/selfservice/va_ssnew/help/customer/locale/en-US/portal/554400000001018/content/554400000330850/VA-Pamphlet-VAP26-7-Chapter-04-Credit-Underwriting",
        "source_format": "extracted_text",
        "content_scope": "partial",
    },
    {
        "path": "ecoa_regulation_b.txt",
        "loan_program": "general",
        "issuer": "Consumer Financial Protection Bureau (CFPB)",
        "document": "12 CFR Part 1002 — Equal Credit Opportunity Act (Regulation B)",
        "document_type": "fair_lending",
        "source_url": "https://www.consumerfinance.gov/rules-policy/regulations/1002/",
        "source_format": "extracted_text",
        "content_scope": "partial",
    },
    {
        "path": "usda_hb_1_3555_excerpts.txt",
        "loan_program": "usda",
        "issuer": "U.S. Department of Agriculture (USDA), Rural Housing Service",
        "document": "HB-1-3555 — Single Family Housing Guaranteed Loan Program Technical Handbook",
        "document_type": "underwriting",
        "source_url": "https://www.usda.gov/sites/default/files/guidance-documents/RHS%20Consolidated%20HB-1-3555.pdf",
        "source_format": "extracted_text",
        "content_scope": "partial",
    },
]

# The .txt sources above embed their own per-section markers (see docs/*.txt)
# so each chunk can carry the exact section and URL it came from, rather
# than only the whole-document default. PDFs have no such markers and are
# treated as a single section.
_SECTION_PATTERN = re.compile(
    r"^SECTION:\s*(?P<section>.+?)\s*\nSOURCE URL:\s*(?P<url>\S+)\s*\n=+\s*$",
    re.MULTILINE,
)


def extract_text(path: str) -> str:
    if path.lower().endswith(".pdf"):
        with pdfplumber.open(path) as pdf:
            return "\n".join(p.extract_text() or "" for p in pdf.pages)
    with open(path, encoding="utf-8") as f:
        return f.read()


def split_into_sections(raw_text: str, default_source_url: str) -> list[tuple[str | None, str, str]]:
    """Splits a document's text into (section_label, source_url, body) parts
    using embedded 'SECTION: ...' / 'SOURCE URL: ...' markers when present.
    Falls back to one (None, default_source_url, raw_text) part otherwise
    (plain PDFs, or a .txt file with no markers)."""
    matches = list(_SECTION_PATTERN.finditer(raw_text))
    if not matches:
        return [(None, default_source_url, raw_text)]

    parts = []
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(raw_text)
        body = raw_text[start:end].strip()
        # A section's body can end right up against the next section's own
        # "====" divider line (or the file's closing one), pulling it in as
        # trailing noise — strip any line that's purely "=" characters.
        body = re.sub(r"\n *=+\s*\n?", "\n", body).strip()
        if body:
            parts.append((m.group("section"), m.group("url"), body))
    return parts


def chunk_text(text: str) -> list[str]:
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunks.append(" ".join(words[i:i + CHUNK_SIZE]))
        i += CHUNK_SIZE - OVERLAP
    return chunks


def build_collection() -> None:
    """Rebuilds the collection from scratch. Re-running this script always
    starts from an empty collection instead of appending — the previous
    version had no deduplication, so re-ingestion would silently accumulate
    duplicate chunks (or error on colliding IDs) every run."""
    db_client = chromadb.PersistentClient(path=DB_PATH)
    try:
        db_client.delete_collection(COLLECTION)
        print(f"[ingest] Deleted existing '{COLLECTION}' collection for a clean rebuild.")
    except Exception:
        pass  # collection didn't exist yet — nothing to delete
    collection = db_client.create_collection(COLLECTION)

    model = TextEmbedding(model_name=EMBED_MODEL)
    total_chunks = 0

    for doc in DOCUMENTS:
        full_path = os.path.join(DOCS_DIR, doc["path"])
        if not os.path.exists(full_path):
            print(f"[ingest] WARNING: {full_path} not found — skipping.")
            continue

        print(f"[ingest] Ingesting {doc['path']} (loan_program={doc['loan_program']}, "
              f"document_type={doc['document_type']})...")
        text = extract_text(full_path)
        sections = split_into_sections(text, doc["source_url"])

        doc_chunks, doc_metadatas, doc_ids = [], [], []
        for section_index, (section_label, source_url, body) in enumerate(sections):
            for chunk_index, chunk in enumerate(chunk_text(body)):
                doc_chunks.append(chunk)
                doc_metadatas.append({
                    "loan_program": doc["loan_program"],
                    "issuer": doc["issuer"],
                    "document": doc["document"],
                    "document_type": doc["document_type"],
                    "source_url": source_url,
                    "section": section_label or "",
                    "source_format": doc["source_format"],
                    "content_scope": doc["content_scope"],
                })
                doc_ids.append(f"{doc['path']}::{section_index}::{chunk_index}")

        if not doc_chunks:
            print(f"[ingest] WARNING: No text extracted from {doc['path']}.")
            continue

        collection.add(
            documents=doc_chunks,
            embeddings=[vec.tolist() for vec in model.embed(doc_chunks)],
            metadatas=doc_metadatas,
            ids=doc_ids,
        )
        total_chunks += len(doc_chunks)
        print(f"[ingest] Added {len(doc_chunks)} chunk(s) from {doc['path']}.")

    print(f"[ingest] Done. Collection '{COLLECTION}' now has {collection.count()} chunk(s) "
          f"({total_chunks} added this run).")


if __name__ == "__main__":
    build_collection()
