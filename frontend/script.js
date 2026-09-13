// Real records from the model's held-out test set. Field keys match the
// existing form input IDs exactly, and values use each field's existing
// payload representation (see script.js's submit handler) — e.g. loan_amount
// and applicant_income are in the same raw ($ thousands) units the model
// was trained on, with no client-side unit conversion.
const exampleApplications = [
    {
        id: 338455,
        label: 'Conventional Preapproval',
        description: '$285K conventional home purchase with requested preapproval',
        fields: {
            loan_amount: 285.0, applicant_income: 121.0, population: 4515.0,
            minority_population: 9.86, hud_median_family_income: 65500.0,
            tract_to_msamd_income: 134.32, number_of_owner_occupied_units: 1334.0,
            census_tract_number: 20.12, county: 'Manatee County',
            msamd: 'North Port, Sarasota, Bradenton - FL',
            property_type: 'One-to-four family dwelling (other than manufactured housing)',
            lien_status: 'Secured by a first lien',
            agency: 'Department of Housing and Urban Development',
            loan_type: 'Conventional', loan_purpose: 'Home purchase',
            owner_occupancy: 'Owner-occupied as a principal dwelling',
            preapproval: 'Preapproval was requested',
            applicant_ethnicity: 'Not Hispanic or Latino', applicant_race_name_1: 'White',
            applicant_sex: 'Male', co_applicant_ethnicity: 'Not Hispanic or Latino',
            co_applicant_race_name_1: 'White', co_applicant_sex: 'Female',
        },
    },
    {
        id: 306559,
        label: 'FHA Miami Application',
        description: '$316K FHA-insured purchase in Miami-Dade',
        fields: {
            loan_amount: 316.0, applicant_income: 122.0, population: 5203.0,
            minority_population: 99.31, hud_median_family_income: 51800.0,
            tract_to_msamd_income: 139.61, number_of_owner_occupied_units: 1165.0,
            census_tract_number: 99.03, county: 'Miami-Dade County',
            msamd: 'Miami, Miami Beach, Kendall - FL',
            property_type: 'One-to-four family dwelling (other than manufactured housing)',
            lien_status: 'Secured by a first lien',
            agency: 'Consumer Financial Protection Bureau',
            loan_type: 'FHA-insured', loan_purpose: 'Home purchase',
            owner_occupancy: 'Owner-occupied as a principal dwelling',
            preapproval: 'Preapproval was not requested',
            applicant_ethnicity: 'Not Hispanic or Latino', applicant_race_name_1: 'Black or African American',
            applicant_sex: 'Female', co_applicant_ethnicity: 'Not Hispanic or Latino',
            co_applicant_race_name_1: 'Black or African American', co_applicant_sex: 'Male',
        },
    },
    {
        id: 655272,
        label: 'VA Jacksonville Application',
        description: '$291K VA-guaranteed purchase in the Jacksonville metro',
        fields: {
            loan_amount: 291.0, applicant_income: 95.0, population: 26173.0,
            minority_population: 16.16, hud_median_family_income: 64300.0,
            tract_to_msamd_income: 160.10, number_of_owner_occupied_units: 7554.0,
            census_tract_number: 209.02, county: 'St. Johns County',
            msamd: 'Jacksonville - FL',
            property_type: 'One-to-four family dwelling (other than manufactured housing)',
            lien_status: 'Secured by a first lien',
            agency: 'Federal Reserve System',
            loan_type: 'VA-guaranteed', loan_purpose: 'Home purchase',
            owner_occupancy: 'Owner-occupied as a principal dwelling',
            preapproval: 'Preapproval was not requested',
            applicant_ethnicity: 'Not Hispanic or Latino', applicant_race_name_1: 'White',
            applicant_sex: 'Male', co_applicant_ethnicity: 'No co-applicant',
            co_applicant_race_name_1: 'No co-applicant', co_applicant_sex: 'No co-applicant',
        },
    },
    {
        id: 283547,
        label: 'Broward Refinance',
        description: '$101K conventional refinance in Broward County',
        fields: {
            loan_amount: 101.0, applicant_income: 87.0, population: 6526.0,
            minority_population: 57.48, hud_median_family_income: 64100.0,
            tract_to_msamd_income: 104.36, number_of_owner_occupied_units: 1725.0,
            census_tract_number: 202.05, county: 'Broward County',
            msamd: 'Fort Lauderdale, Pompano Beach, Deerfield Beach - FL',
            property_type: 'One-to-four family dwelling (other than manufactured housing)',
            lien_status: 'Secured by a first lien',
            agency: 'Consumer Financial Protection Bureau',
            loan_type: 'Conventional', loan_purpose: 'Refinancing',
            owner_occupancy: 'Owner-occupied as a principal dwelling',
            preapproval: 'Not applicable',
            applicant_ethnicity: 'Not Hispanic or Latino', applicant_race_name_1: 'White',
            applicant_sex: 'Male', co_applicant_ethnicity: 'Not Hispanic or Latino',
            co_applicant_race_name_1: 'White', co_applicant_sex: 'Female',
        },
    },
    {
        id: 212325,
        label: 'Investment Property Purchase',
        description: '$290K non-owner-occupied property purchase',
        fields: {
            loan_amount: 290.0, applicant_income: 216.0, population: 7215.0,
            minority_population: 49.12, hud_median_family_income: 58400.0,
            tract_to_msamd_income: 90.42, number_of_owner_occupied_units: 876.0,
            census_tract_number: 408.02, county: 'Osceola County',
            msamd: 'Orlando, Kissimmee, Sanford - FL',
            property_type: 'One-to-four family dwelling (other than manufactured housing)',
            lien_status: 'Secured by a first lien',
            agency: 'Federal Reserve System',
            loan_type: 'Conventional', loan_purpose: 'Home purchase',
            owner_occupancy: 'Not owner-occupied as a principal dwelling',
            preapproval: 'Not applicable',
            applicant_ethnicity: 'Not Hispanic or Latino', applicant_race_name_1: 'White',
            applicant_sex: 'Male', co_applicant_ethnicity: 'No co-applicant',
            co_applicant_race_name_1: 'No co-applicant', co_applicant_sex: 'No co-applicant',
        },
    },
    {
        id: 664185,
        label: 'Manufactured Home Purchase',
        description: '$96K manufactured-home purchase',
        fields: {
            loan_amount: 96.0, applicant_income: 24.0, population: 5183.0,
            minority_population: 32.88, hud_median_family_income: 57300.0,
            tract_to_msamd_income: 82.61, number_of_owner_occupied_units: 898.0,
            census_tract_number: 8.03, county: 'Bay County',
            msamd: 'Panama City - FL',
            property_type: 'Manufactured housing',
            lien_status: 'Secured by a first lien',
            agency: 'Department of Housing and Urban Development',
            loan_type: 'Conventional', loan_purpose: 'Home purchase',
            owner_occupancy: 'Owner-occupied as a principal dwelling',
            preapproval: 'Not applicable',
            applicant_ethnicity: 'Not Hispanic or Latino', applicant_race_name_1: 'American Indian or Alaska Native',
            applicant_sex: 'Female', co_applicant_ethnicity: 'No co-applicant',
            co_applicant_race_name_1: 'No co-applicant', co_applicant_sex: 'No co-applicant',
        },
    },
    {
        id: 499523,
        label: 'Unsecured Home Improvement',
        description: '$10K unsecured home-improvement application',
        fields: {
            loan_amount: 10.0, applicant_income: 48.0, population: 3076.0,
            minority_population: 87.26, hud_median_family_income: 64300.0,
            tract_to_msamd_income: 30.39, number_of_owner_occupied_units: 546.0,
            census_tract_number: 26.0, county: 'Duval County',
            msamd: 'Jacksonville - FL',
            property_type: 'One-to-four family dwelling (other than manufactured housing)',
            lien_status: 'Not secured by a lien',
            agency: 'Consumer Financial Protection Bureau',
            loan_type: 'Conventional', loan_purpose: 'Home improvement',
            owner_occupancy: 'Owner-occupied as a principal dwelling',
            preapproval: 'Not applicable',
            applicant_ethnicity: 'Not Hispanic or Latino', applicant_race_name_1: 'Black or African American',
            applicant_sex: 'Female', co_applicant_ethnicity: 'No co-applicant',
            co_applicant_race_name_1: 'No co-applicant', co_applicant_sex: 'No co-applicant',
        },
    },
    {
        id: 492084,
        label: 'Subordinate-Lien Improvement',
        description: '$50K subordinate-lien home-improvement loan',
        fields: {
            loan_amount: 50.0, applicant_income: 85.0, population: 5521.0,
            minority_population: 6.83, hud_median_family_income: 64300.0,
            tract_to_msamd_income: 88.71, number_of_owner_occupied_units: 1251.0,
            census_tract_number: 301.03, county: 'Clay County',
            msamd: 'Jacksonville - FL',
            property_type: 'One-to-four family dwelling (other than manufactured housing)',
            lien_status: 'Secured by a subordinate lien',
            agency: 'National Credit Union Administration',
            loan_type: 'Conventional', loan_purpose: 'Home improvement',
            owner_occupancy: 'Owner-occupied as a principal dwelling',
            preapproval: 'Not applicable',
            applicant_ethnicity: 'Not Hispanic or Latino', applicant_race_name_1: 'White',
            applicant_sex: 'Female', co_applicant_ethnicity: 'Not Hispanic or Latino',
            co_applicant_race_name_1: 'White', co_applicant_sex: 'Male',
        },
    },
    {
        id: 404851,
        label: 'Naples Refinance',
        description: '$309K conventional refinance',
        fields: {
            loan_amount: 309.0, applicant_income: 23.0, population: 8556.0,
            minority_population: 13.21, hud_median_family_income: 68300.0,
            tract_to_msamd_income: 138.22, number_of_owner_occupied_units: 3180.0,
            census_tract_number: 112.01, county: 'Collier County',
            msamd: 'Naples, Immokalee, Marco Island - FL',
            property_type: 'One-to-four family dwelling (other than manufactured housing)',
            lien_status: 'Secured by a first lien',
            agency: 'Consumer Financial Protection Bureau',
            loan_type: 'Conventional', loan_purpose: 'Refinancing',
            owner_occupancy: 'Owner-occupied as a principal dwelling',
            preapproval: 'Not applicable',
            applicant_ethnicity: 'Hispanic or Latino', applicant_race_name_1: 'White',
            applicant_sex: 'Male', co_applicant_ethnicity: 'Hispanic or Latino',
            co_applicant_race_name_1: 'White', co_applicant_sex: 'Female',
        },
    },
    {
        id: 517925,
        label: 'FSA/RHS Home Purchase',
        description: '$152K FSA/RHS-guaranteed home purchase',
        fields: {
            loan_amount: 152.0, applicant_income: 41.0, population: 4352.0,
            minority_population: 36.40, hud_median_family_income: 65200.0,
            tract_to_msamd_income: 96.90, number_of_owner_occupied_units: 1217.0,
            census_tract_number: 22.1, county: 'Alachua County',
            msamd: 'Gainesville - FL',
            property_type: 'One-to-four family dwelling (other than manufactured housing)',
            lien_status: 'Secured by a first lien',
            agency: 'Department of Housing and Urban Development',
            loan_type: 'FSA/RHS-guaranteed', loan_purpose: 'Home purchase',
            owner_occupancy: 'Owner-occupied as a principal dwelling',
            preapproval: 'Not applicable',
            applicant_ethnicity: 'Information not provided by applicant in mail, Internet, or telephone application',
            applicant_race_name_1: 'Information not provided by applicant in mail, Internet, or telephone application',
            applicant_sex: 'Information not provided',
            co_applicant_ethnicity: 'No co-applicant', co_applicant_race_name_1: 'No co-applicant',
            co_applicant_sex: 'No co-applicant',
        },
    },
];

// Sets a form field to an example's value, matching it against a <select>'s
// existing option values case-insensitively when an exact match isn't found
// (e.g. minor case differences in long HMDA category strings), then fires a
// change event in case any future logic depends on it.
function setFieldValue(fieldId, value) {
    const el = document.getElementById(fieldId);
    if (!el) return;

    if (el.tagName === 'SELECT') {
        const target = String(value).toLowerCase();
        const match = Array.from(el.options).find(
            (opt) => opt.value.toLowerCase() === target
        );
        el.value = match ? match.value : '';
    } else {
        el.value = value;
    }
    el.dispatchEvent(new Event('change', { bubbles: true }));
}

function loadExampleApplication(example) {
    Object.entries(example.fields).forEach(([fieldId, value]) => {
        setFieldValue(fieldId, value);
    });
}

// Fills the "Select an application" dropdown with the 10 held-out test
// cases, in order, after the existing "Fill manually" option. Each option's
// value is its index into exampleApplications, so the change handler can
// look the case back up with exampleApplications[select.value].
function populateApplicationSelect() {
    const select = document.getElementById('application-select');
    if (!select) return;

    exampleApplications.forEach((example, index) => {
        const option = document.createElement('option');
        option.value = String(index);
        option.textContent = example.label;
        select.appendChild(option);
    });
}

document.addEventListener('DOMContentLoaded', () => {
    populateApplicationSelect();

    const form = document.getElementById('loan-form');
    const resultContainer = document.getElementById('result-container');
    const submitBtn = document.getElementById('submit-btn');
    const resetBtn = document.getElementById('reset-btn');
    const applicationSelect = document.getElementById('application-select');
    const btnText = submitBtn.querySelector('.btn-text');
    const spinner = submitBtn.querySelector('.spinner');

    const decisionBadge = document.getElementById('decision-badge');
    const confidenceFill = document.getElementById('confidence-fill');
    const confidenceVal = document.getElementById('confidence-val');
    const explanationText = document.getElementById('explanation-text');

    // Shared by the "New Prediction" button and the "Fill manually" dropdown
    // option so both return the page to the same clean manual-entry state.
    // application-select lives outside #loan-form, so form.reset() here never
    // touches the dropdown's own value.
    function resetForm() {
        resultContainer.classList.add('hidden');
        form.classList.remove('hidden');
        form.reset();
        confidenceFill.style.width = '0%';
    }

    applicationSelect.addEventListener('change', () => {
        const selectedValue = applicationSelect.value;
        resetForm();
        if (selectedValue !== 'manual') {
            loadExampleApplication(exampleApplications[Number(selectedValue)]);
        }
    });

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        // Setup UI for loading
        btnText.classList.add('hidden');
        spinner.classList.remove('hidden');
        submitBtn.disabled = true;
        resultContainer.classList.add('hidden');
        
        // Gather data
        const payload = {
            loan_amount: document.getElementById('loan_amount').value ? parseFloat(document.getElementById('loan_amount').value) : null,
            applicant_income: document.getElementById('applicant_income').value ? parseFloat(document.getElementById('applicant_income').value) : null,
            population: document.getElementById('population').value ? parseFloat(document.getElementById('population').value) : null,
            minority_population: document.getElementById('minority_population').value ? parseFloat(document.getElementById('minority_population').value) : null,
            hud_median_family_income: document.getElementById('hud_median_family_income').value ? parseFloat(document.getElementById('hud_median_family_income').value) : null,
            tract_to_msamd_income: document.getElementById('tract_to_msamd_income').value ? parseFloat(document.getElementById('tract_to_msamd_income').value) : null,
            number_of_owner_occupied_units: document.getElementById('number_of_owner_occupied_units').value ? parseFloat(document.getElementById('number_of_owner_occupied_units').value) : null,
            A: null,
            B: null,
            C: null,
            property_type: document.getElementById('property_type').value || null,
            loan_purpose: document.getElementById('loan_purpose').value || null,
            owner_occupancy: document.getElementById('owner_occupancy').value || null,
            preapproval: document.getElementById('preapproval').value || null,
            applicant_ethnicity: document.getElementById('applicant_ethnicity').value || null,
            applicant_race_name_1: document.getElementById('applicant_race_name_1').value || null,
            co_applicant_ethnicity: document.getElementById('co_applicant_ethnicity').value || null,
            co_applicant_race_name_1: document.getElementById('co_applicant_race_name_1').value || null,
            census_tract_number: document.getElementById('census_tract_number').value ? parseFloat(document.getElementById('census_tract_number').value) : null,
            county: document.getElementById('county').value || null,
            msamd: document.getElementById('msamd').value || null,
            lien_status: document.getElementById('lien_status').value || null,
            applicant_sex: document.getElementById('applicant_sex').value || null,
            co_applicant_sex: document.getElementById('co_applicant_sex').value || null,
            agency: document.getElementById('agency').value || null,
            D: null,
            loan_type: document.getElementById('loan_type').value || null
        };

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(payload)
            });

            if (!response.ok) {
                throw new Error('API request failed');
            }

            const data = await response.json();
            
            // Format UI
            form.classList.add('hidden');
            resultContainer.classList.remove('hidden');
            
            decisionBadge.textContent = data.decision;
            decisionBadge.className = 'decision-badge ' + (data.decision.toLowerCase() === 'approved' ? 'decision-approved' : 'decision-denied');
            
            // Animate confidence bar
            const confPct = Math.round(data.confidence * 100);
            confidenceVal.textContent = "0%";
            
            // Small delay to allow CSS transition
            setTimeout(() => {
                confidenceFill.style.width = confPct + '%';
                // Animate number
                let curr = 0;
                const interval = setInterval(() => {
                    curr += Math.ceil(confPct / 20) || 1;
                    if (curr >= confPct) {
                        curr = confPct;
                        clearInterval(interval);
                    }
                    confidenceVal.textContent = curr + '%';
                }, 40);
            }, 100);

            explanationText.textContent = data.explanation;

        } catch (error) {
            console.error('Error:', error);
            alert('Failed to get prediction. Ensure the backend is running.');
        } finally {
            // Restore button
            btnText.classList.remove('hidden');
            spinner.classList.add('hidden');
            submitBtn.disabled = false;
        }
    });

    resetBtn.addEventListener('click', () => {
        applicationSelect.value = 'manual';
        resetForm();
    });
});
