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


def build_query(decision: str, features: dict, confidence: float | None = None) -> str:
    """Build a semantic search query from the decision and key applicant features."""
    f = _readable(features)
    parts = [f"Loan application {decision}."]
    if confidence is not None:
        parts.append(f"Model confidence: {confidence * 100:.0f}%.")
    if f.get("loan_type"):
        parts.append(f"Loan type: {f['loan_type']}.")
    if f.get("loan_purpose"):
        parts.append(f"Purpose: {f['loan_purpose']}.")
    if f.get("property_type"):
        parts.append(f"Property: {f['property_type']}.")
    if f.get("owner_occupancy"):
        parts.append(f"Occupancy: {f['owner_occupancy']}.")
    if f.get("preapproval"):
        parts.append(f"Preapproval: {f['preapproval']}.")
    if f.get("applicant_income") is not None:
        parts.append(f"Applicant income: ${f['applicant_income']}k.")
    if f.get("loan_amount") is not None:
        parts.append(f"Loan amount: ${f['loan_amount']}k.")
    if f.get("lien_status"):
        parts.append(f"Lien status: {f['lien_status']}.")
    return " ".join(parts)


SYSTEM_PROMPT = """You are a fair-lending compliance assistant summarizing the output of an internal \
ML decision-support model for a lender. You are not a loan officer, and this is not a formal \
adverse-action notice or a legally binding underwriting decision.

Rules you must follow:
- Base your explanation only on the application data and regulation excerpts given below. Never \
invent facts (credit score, debt-to-income ratio, employment history, delinquency history, etc.) \
that are not present in the provided data.
- The application data may include HMDA-required demographic fields (race, ethnicity, sex). These \
exist because HMDA requires lenders to collect them, not because they are legitimate underwriting \
criteria. Never cite a protected characteristic as a reason for the outcome, and never imply one was \
an appropriate basis for the model's prediction.
- Clearly distinguish the ML model's statistical output from a legally valid, compliant underwriting \
decision — this is a research/portfolio demo, not a real adverse-action determination.
- Reference the retrieved regulation excerpts only where they are genuinely relevant to this case; do \
not fabricate a regulatory citation that isn't supported by the excerpts.
- If the available application data does not establish a clear, specific reason for the outcome, say \
so explicitly rather than guessing.
- Note that a human/legal fair-lending and compliance review would still be required before any real \
lending decision.
- Keep the tone professional and concise: 3-4 sentences."""


def generate_explanation(
    decision: str, features: dict, chunks: list[str], confidence: float | None = None
) -> str:
    """Generate a compliance explanation grounded in regulation chunks using Claude."""
    f = _readable(features)

    context = "\n\n".join(chunks) if chunks else "No regulation context available."

    feature_summary = ", ".join(
        f"{k.replace('_', ' ')}: {v}"
        for k, v in f.items()
        if v is not None and k not in {"A", "B", "C", "D"}
    )
    if not feature_summary:
        feature_summary = "No application fields were provided."

    confidence_line = (
        f"Model confidence: {confidence * 100:.0f}%\n" if confidence is not None else ""
    )

    try:
        message = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=300,
            system=SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Model outcome: {decision.upper()}\n"
                        f"{confidence_line}\n"
                        f"Submitted application data:\n{feature_summary}\n\n"
                        f"Retrieved regulation/compliance excerpts:\n{context}\n\n"
                        f"Write the compliance explanation now."
                    ),
                }
            ],
        )
        return message.content[0].text
    except TypeError as e:
        # The Anthropic SDK raises a plain TypeError client-side, before any
        # network call, when ANTHROPIC_API_KEY is entirely unset in the
        # environment (as opposed to set-but-invalid, which surfaces as
        # AuthenticationError below). This is the single most common
        # deployment failure — flag it loudly and specifically.
        print(
            "[rag.generator] ANTHROPIC_API_KEY is not set in this environment "
            f"— Claude client could not resolve credentials: {e}. "
            "Set ANTHROPIC_API_KEY in the deployment environment. "
            "Returning fallback explanation."
        )
        return (
            f"This application was {decision} based on the applicant's financial profile. "
            "A detailed regulation-grounded explanation is temporarily unavailable."
        )
    except anthropic.AuthenticationError as e:
        # ANTHROPIC_API_KEY is set but rejected by the API (invalid/revoked).
        print(
            "[rag.generator] ANTHROPIC_API_KEY was rejected by the Anthropic "
            f"API (invalid or revoked key): {e}. "
            "Returning fallback explanation."
        )
        return (
            f"This application was {decision} based on the applicant's financial profile. "
            "A detailed regulation-grounded explanation is temporarily unavailable."
        )
    except Exception as e:
        # Network hiccup, rate limit, or other transient provider failure —
        # degrade gracefully instead of failing the whole /predict request.
        print(f"[rag.generator] Claude call failed: {e}. Returning fallback explanation.")
        return (
            f"This application was {decision} based on the applicant's financial profile. "
            "A detailed regulation-grounded explanation is temporarily unavailable."
        )
