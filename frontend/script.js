document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('loan-form');
    const resultContainer = document.getElementById('result-container');
    const submitBtn = document.getElementById('submit-btn');
    const resetBtn = document.getElementById('reset-btn');
    const btnText = submitBtn.querySelector('.btn-text');
    const spinner = submitBtn.querySelector('.spinner');

    const decisionBadge = document.getElementById('decision-badge');
    const confidenceFill = document.getElementById('confidence-fill');
    const confidenceVal = document.getElementById('confidence-val');
    const explanationText = document.getElementById('explanation-text');

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        // Setup UI for loading
        btnText.classList.add('hidden');
        spinner.classList.remove('hidden');
        submitBtn.disabled = true;
        resultContainer.classList.add('hidden');
        
        // Gather data
        const payload = {
            loan_amount: parseFloat(document.getElementById('loan_amount').value),
            applicant_income: parseFloat(document.getElementById('applicant_income').value),
            population: parseFloat(document.getElementById('population').value),
            minority_population: parseFloat(document.getElementById('minority_population').value),
            hud_median_family_income: parseFloat(document.getElementById('hud_median_family_income').value),
            tract_to_msamd_income: parseFloat(document.getElementById('tract_to_msamd_income').value),
            number_of_owner_occupied_units: parseFloat(document.getElementById('number_of_owner_occupied_units').value),
            A: null,
            B: null,
            C: null,
            property_type: parseInt(document.getElementById('property_type').value),
            preapproval: document.getElementById('preapproval').value,
            applicant_ethnicity: document.getElementById('applicant_ethnicity').value,
            applicant_race_name_1: document.getElementById('applicant_race_name_1').value,
            co_applicant_ethnicity: document.getElementById('co_applicant_ethnicity').value,
            co_applicant_race_name_1: document.getElementById('co_applicant_race_name_1').value,
            census_tract_number: parseFloat(document.getElementById('census_tract_number').value),
            county: parseFloat(document.getElementById('county').value),
            msamd: parseFloat(document.getElementById('msamd').value),
            lien_status: parseInt(document.getElementById('lien_status').value),
            applicant_sex: document.getElementById('applicant_sex').value,
            co_applicant_sex: document.getElementById('co_applicant_sex').value,
            agency: parseInt(document.getElementById('agency').value),
            D: null,
            loan_type: parseInt(document.getElementById('loan_type').value)
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
        resultContainer.classList.add('hidden');
        form.classList.remove('hidden');
        form.reset();
        confidenceFill.style.width = '0%';
    });
});
