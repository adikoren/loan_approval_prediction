from fpdf import FPDF

def create_fha_pdf():
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=12)
    text = """
    HUD Handbook 4000.1 - FHA Single Family Housing Policy Handbook
    
    Section II.A.4: Underwriting the Borrower
    
    Debt-to-Income (DTI) Ratios:
    The Mortgagee must determine that the Borrower has the ability to repay the mortgage. For FHA loans, the standard maximum back-end Debt-to-Income (DTI) ratio is 43%. Borrowers must demonstrate a back-end DTI below 43% unless compensating factors apply.
    If compensating factors (such as significant cash reserves or a high credit score) are present, the DTI may exceed 43%, up to a maximum of 50%.
    Applications with a DTI above 50% are typically denied.
    
    Credit Score (FICO) Requirements:
    To be eligible for maximum financing (3.5% down payment), the Borrower must have a Minimum Decision Credit Score of at least 580.
    Borrowers with a Minimum Decision Credit Score between 500 and 579 are limited to a maximum LTV of 90% (meaning a 10% down payment is required).
    Borrowers with a credit score below 500 are not eligible for FHA-insured financing.
    
    Down Payment Minimums:
    The standard minimum down payment for an FHA loan is 3.5% of the appraised value or purchase price, whichever is less.
    """
    pdf.multi_cell(0, 10, text)
    pdf.output("docs/fha_handbook.pdf")

def create_fannie_mae_pdf():
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", size=11)
    text = """Fannie Mae Selling Guide - Conventional Loan Underwriting Guidelines

PART B: ORIGINATING & UNDERWRITING

Section B3-1: Comprehensive Risk Assessment

Credit Score (FICO) Requirements:
The minimum credit score required for a conventional loan eligible for delivery to Fannie Mae is 620. Borrowers with a credit score below 620 are not eligible for conventional financing. For loans with LTV ratios above 75%, a minimum score of 640 is recommended. Borrowers with scores of 740 or above receive the most favorable pricing adjustments.

Debt-to-Income (DTI) Ratios:
The standard maximum DTI ratio for conventional conforming loans is 45% when assessed through Desktop Underwriter (DU). With compensating factors such as significant cash reserves, a high credit score (720+), or a low LTV, DU may approve loans up to 50% DTI. Loans with DTI above 50% are generally ineligible. The front-end (housing) DTI should not exceed 28% for manually underwritten loans.

Down Payment and Loan-to-Value (LTV):
The minimum down payment for a conventional loan is 3% for first-time homebuyers under the HomeReady program and 5% for standard primary residence purchases. Investment properties require a minimum 15% down payment. Second homes require at least 10% down. Private Mortgage Insurance (PMI) is required for LTV ratios above 80%.

Income and Employment Verification:
Borrowers must have a minimum of two years of stable employment history. Self-employed borrowers must provide two years of federal tax returns. Base salary, overtime, bonuses, and commission income are allowable if documented with pay stubs and W-2s. Rental income may be counted if supported by signed lease agreements and tax returns.

Property Eligibility:
Eligible properties include one-to-four family dwellings, condominiums, and planned unit developments (PUDs). Manufactured homes are eligible under specific guidelines with a minimum 5% down payment. Mixed-use properties are eligible if the residential use represents at least 51% of the total property square footage. Non-warrantable condominiums are ineligible.

Lien Status and Subordinate Financing:
First lien mortgages are the standard for conventional conforming loans. Subordinate lien financing is permitted provided the combined LTV (CLTV) does not exceed program maximums. Loans not secured by a lien are ineligible for standard conforming delivery.

Loan Purpose Guidelines:
Purchase transactions: Full appraisal required; purchase price or appraised value, whichever is lower, is used for LTV calculation.
Rate and term refinance: Must demonstrate a net tangible benefit to the borrower such as lower rate, shorter term, or elimination of mortgage insurance.
Cash-out refinance: Maximum LTV of 80% for primary residences; limited to 75% for second homes and 70% for investment properties.
Home improvement loans: Proceeds must be used for documented property improvements; contractor bids or invoices required.

Geographic and Census Tract Considerations:
Loans secured by properties in declining markets may be subject to additional LTV restrictions of 5%. High-cost areas have conforming loan limits above the standard national baseline. Properties in rural areas may qualify for special rural lending programs. Tract-to-MSAMD income ratios below 80% may indicate a low-to-moderate income (LMI) census tract eligible for CRA credit.

Agency and Regulatory Compliance:
All loans must comply with applicable federal, state, and local laws including the Truth in Lending Act (TILA), the Real Estate Settlement Procedures Act (RESPA), and the Equal Credit Opportunity Act (ECOA). Loans originated by institutions regulated by the OCC, Federal Reserve, FDIC, NCUA, HUD, or CFPB must meet both agency-specific and Fannie Mae guidelines. CRA performance obligations may influence lending patterns in specific geographies.

Owner Occupancy:
Loans secured by owner-occupied primary residences receive the most favorable underwriting treatment. Non-owner-occupied investment properties are subject to higher reserve requirements and more restrictive LTV limits. Occupancy misrepresentation is mortgage fraud and grounds for immediate loan repurchase.

Minority and Underserved Communities:
The HomeReady mortgage program is designed to serve low-to-moderate income borrowers, including those in minority census tracts. Borrowers in high-minority-population tracts may qualify for expanded DTI ratios and reduced mortgage insurance coverage under the HomeReady guidelines. Lenders are encouraged to actively market to underserved communities to fulfill fair lending obligations.
"""
    pdf.multi_cell(0, 8, text)
    pdf.output("docs/fannie_mae_selling_guide.pdf")

def create_hmda_pdf():
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=12)
    text = """
    Home Mortgage Disclosure Act (HMDA) Filing Instructions
    
    General Lending Rules and Compliance:
    
    Fair Lending and Non-Discrimination:
    Under the Equal Credit Opportunity Act (ECOA) and HMDA reporting rules, lenders must not discriminate against any applicant on a prohibited basis (including race, color, religion, national origin, sex, marital status, or age).
    Loan approvals must be based solely on objective, verifiable financial criteria such as the applicant's credit history, capacity to repay (measured by DTI and income), and the value of collateral.
    
    Documentation Requirements:
    All income used to qualify for the loan must be thoroughly documented using W-2 forms, tax returns, and current pay stubs.
    Asset verification is required to confirm the borrower has sufficient funds for the down payment and closing costs.
    """
    pdf.multi_cell(0, 10, text)
    pdf.output("docs/hmda_guidelines.pdf")

if __name__ == "__main__":
    # FHA and HMDA are downloaded from official sources — only regenerate Fannie Mae
    create_fannie_mae_pdf()
    print("Fannie Mae PDF generated in docs/")
