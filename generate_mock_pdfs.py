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
    pdf.add_page()
    pdf.set_font("Helvetica", size=12)
    text = """
    Fannie Mae Selling Guide
    
    Conventional Loan Guidelines:
    
    Credit Score (FICO) Requirements:
    The minimum credit score required for a conventional loan eligible for delivery to Fannie Mae is typically 620.
    Borrowers with a credit score below 620 are generally not eligible for conventional financing.
    
    Debt-to-Income (DTI) Ratios:
    The standard maximum DTI ratio for conventional conforming loans is 36%. However, with a strong credit profile and compensating factors, Desktop Underwriter (DU) may approve loans with a maximum DTI of up to 45% or 50%.
    A higher DTI ratio often requires a higher minimum FICO score (e.g., 680 to 700+) to receive an Approve/Eligible recommendation.
    
    Down Payment Minimums:
    The minimum down payment for a conventional loan is usually 3% for first-time homebuyers and 5% for standard primary residence purchases.
    """
    pdf.multi_cell(0, 10, text)
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
    create_fha_pdf()
    create_fannie_mae_pdf()
    create_hmda_pdf()
    print("PDFs generated in docs/")
