import os

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
# Human-readable program label per loan_type, for the "program-specific
# coverage" line told to the LLM — kept separate from
# rag/retriever.py's LOAN_TYPE_TO_PROGRAM (that one's values are the
# loan_program metadata values used to filter retrieval; these are just
# display labels).
LOAN_TYPE_TO_PROGRAM_LABEL = {
    "Conventional": "conventional",
    "FHA-insured": "FHA",
    "VA-guaranteed": "VA",
    "FSA/RHS-guaranteed": "USDA/FSA-RHS",
}

# Some Anthropic Console organizations issue API keys that are not scoped to
# a single workspace; the API then requires every request to carry an
# anthropic-workspace-id header naming which workspace to bill/attribute the
# call to (the API's own 400 response names this exact fix). Optional: if
# ANTHROPIC_WORKSPACE_ID isn't set, no header is sent and behavior is
# unchanged for keys that are already workspace-scoped.
_workspace_id = os.environ.get("ANTHROPIC_WORKSPACE_ID")
_default_headers = {"anthropic-workspace-id": _workspace_id} if _workspace_id else None

client = anthropic.Anthropic(default_headers=_default_headers)


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


def build_query(
    decision: str,
    features: dict,
    confidence: float | None = None,
    approval_probability: float | None = None,
) -> str:
    """Build a semantic search query from the decision and key applicant features.

    `confidence` and `approval_probability` are distinct: approval_probability
    is the raw model score for the positive (approved) class, while confidence
    is how sure the model is in whichever decision was actually made (i.e.
    1 - approval_probability when the decision is a denial). Conflating the
    two under one number previously made e.g. a denial with a 12% approval
    probability read as "12% confidence" — actually 88% confidence in denial.
    """
    f = _readable(features)
    parts = [f"Loan application {decision}."]
    if confidence is not None:
        parts.append(f"Model confidence in this {decision} decision: {confidence * 100:.0f}%.")
    if approval_probability is not None:
        parts.append(f"Raw model approval probability: {approval_probability * 100:.0f}%.")
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
- Never state or imply that a protected characteristic "played no role," "had no effect," or was \
"not a factor" in the outcome — that cannot be verified from the information available to you, and an \
unverified claim of that kind is itself misleading. If asked to address this, say plainly that whether \
protected characteristics influenced the model's output cannot be confirmed from this explanation alone \
and would require a formal fair-lending review of the model.
- Clearly distinguish the ML model's statistical output from a legally valid, compliant underwriting \
decision — this is a research/portfolio demo, not a real adverse-action determination.
- Reference the retrieved regulation excerpts only where they are genuinely relevant to this case; do \
not fabricate a regulatory citation that isn't supported by the excerpts.
- If the available application data does not establish a clear, specific reason for the outcome, say \
so explicitly rather than guessing.
- Note that a human/legal fair-lending and compliance review would still be required before any real \
lending decision.
- Treat the "Retrieved regulation/compliance excerpts" section below as the ONLY source of \
regulatory grounding. You must not introduce a specific underwriting requirement, threshold, or \
rule from your own general knowledge and present it as though it came from the retrieved corpus. \
If the "Program-specific underwriting coverage" line below says none was found, say so explicitly \
(e.g. "the retrieved knowledge base does not contain sufficient [program]-specific underwriting \
guidance to establish a formal underwriting rationale for this case") rather than filling that gap \
with something you happen to know about that loan program.
- Keep the tone professional and concise: 3-4 sentences.
- Write in plain prose only — no Markdown (no "**bold**", "#" headings, or "-"/"*" bullet lists)."""


def generate_explanation(
    decision: str,
    features: dict,
    chunks: list[dict],
    confidence: float | None = None,
    approval_probability: float | None = None,
) -> str:
    """Generate a compliance explanation grounded in regulation chunks using Claude.

    `chunks` is a list of dicts from rag/retriever.py — each carries "text"
    plus source metadata (document, issuer, document_type, loan_program,
    source_url, section). Program-specific ("underwriting") chunks are
    already restricted to the application's own loan program; "general"
    chunks (fair-lending/reporting) apply regardless of program.
    """
    f = _readable(features)

    program_chunks = [c for c in chunks if c["document_type"] == "underwriting"]
    program = LOAN_TYPE_TO_PROGRAM_LABEL.get(f.get("loan_type"), f.get("loan_type"))

    if chunks:
        context = "\n\n".join(
            f"[{c['source']} — issuer: {c['issuer']}, type: {c['document_type']}"
            + (f", section: {c['section']}" if c.get("section") else "")
            + f"]\n{c['text']}"
            for c in chunks
        )
    else:
        context = "No regulation context available."

    if program and not program_chunks:
        coverage_line = (
            f"Program-specific underwriting coverage: NONE FOUND. The knowledge base has no "
            f"{program}-specific underwriting document covering this case."
        )
    elif program_chunks:
        coverage_line = (
            f"Program-specific underwriting coverage: {len(program_chunks)} chunk(s) found from "
            + ", ".join(sorted({c['source'] for c in program_chunks}))
            + "."
        )
    else:
        coverage_line = "Program-specific underwriting coverage: not applicable (loan program unknown)."

    feature_summary = ", ".join(
        f"{k.replace('_', ' ')}: {v}"
        for k, v in f.items()
        if v is not None and k not in {"A", "B", "C", "D"}
    )
    if not feature_summary:
        feature_summary = "No application fields were provided."

    # See build_query() for why these are two distinct numbers, not one.
    outcome_lines = []
    if confidence is not None:
        outcome_lines.append(f"Model confidence in this {decision} decision: {confidence * 100:.0f}%")
    if approval_probability is not None:
        outcome_lines.append(f"Raw model approval probability: {approval_probability * 100:.0f}%")
    outcome_block = "\n".join(outcome_lines)

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
                        f"{outcome_block}\n\n"
                        f"Submitted application data:\n{feature_summary}\n\n"
                        f"{coverage_line}\n\n"
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
    except anthropic.BadRequestError as e:
        # Observed in production: some Anthropic Console organizations issue
        # API keys that aren't scoped to a single workspace, and the API
        # rejects every request with a 400 until either the key is re-issued
        # as workspace-scoped, or the request carries an
        # anthropic-workspace-id header (see ANTHROPIC_WORKSPACE_ID above).
        print(
            "[rag.generator] Claude API rejected the request as malformed "
            f"(400): {e}. If this mentions workspace scoping, either "
            "regenerate ANTHROPIC_API_KEY as a workspace-scoped key in the "
            "Anthropic Console, or set ANTHROPIC_WORKSPACE_ID in the "
            "deployment environment. Returning fallback explanation."
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
