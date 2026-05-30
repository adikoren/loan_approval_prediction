import anthropic

LOAN_TYPE_MAP = {1: "Conventional", 2: "FHA-insured", 3: "VA-guaranteed", 4: "FSA/RHS-guaranteed"}
PROPERTY_TYPE_MAP = {1: "One-to-four family dwelling", 2: "Manufactured housing", 3: "Multifamily dwelling"}
AGENCY_MAP = {
    1: "Office of the Comptroller of the Currency",
    2: "Federal Reserve System",
    3: "Federal Deposit Insurance Corporation",
    5: "National Credit Union Administration",
    6: "Department of Housing and Urban Development",
    7: "Consumer Financial Protection Bureau",
}

client = anthropic.Anthropic()


def _readable(features: dict) -> dict:
    """Convert integer codes to human-readable strings for use in prompts."""
    f = dict(features)
    if f.get("loan_type") is not None:
        f["loan_type"] = LOAN_TYPE_MAP.get(f["loan_type"], f["loan_type"])
    if f.get("property_type") is not None:
        f["property_type"] = PROPERTY_TYPE_MAP.get(f["property_type"], f["property_type"])
    if f.get("agency") is not None:
        f["agency"] = AGENCY_MAP.get(f["agency"], f["agency"])
    return f


def build_query(decision: str, features: dict) -> str:
    """Build a semantic search query from the decision and key applicant features."""
    f = _readable(features)
    parts = [f"Loan application {decision}."]
    if f.get("loan_type"):
        parts.append(f"Loan type: {f['loan_type']}.")
    if f.get("loan_purpose"):
        parts.append(f"Purpose: {f['loan_purpose']}.")
    if f.get("property_type"):
        parts.append(f"Property: {f['property_type']}.")
    if f.get("applicant_income") is not None:
        parts.append(f"Applicant income: ${f['applicant_income']}k.")
    if f.get("loan_amount") is not None:
        parts.append(f"Loan amount: ${f['loan_amount']}k.")
    if f.get("lien_status"):
        parts.append(f"Lien status: {f['lien_status']}.")
    return " ".join(parts)


def generate_explanation(decision: str, features: dict, chunks: list[str]) -> str:
    """Generate a compliance explanation grounded in regulation chunks using Claude."""
    f = _readable(features)

    context = "\n\n".join(chunks) if chunks else "No regulation context available."

    feature_summary = ", ".join(
        f"{k.replace('_', ' ')}: {v}"
        for k, v in f.items()
        if v is not None and k not in {"A", "B", "C", "D"}
    )

    message = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=300,
        messages=[
            {
                "role": "user",
                "content": (
                    f"You are a mortgage compliance officer. A loan application was {decision.upper()}.\n\n"
                    f"Applicant details: {feature_summary}\n\n"
                    f"Relevant regulation excerpts:\n{context}\n\n"
                    f"Write a concise 2-3 sentence explanation of why this application was {decision}, "
                    f"referencing specific regulation criteria where applicable."
                ),
            }
        ],
    )

    return message.content[0].text
