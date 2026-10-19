document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('predictionForm');
    const btnSubmit = document.getElementById('btnSubmit');
    const btnLoader = document.getElementById('btnLoader');
    const emptyState = document.getElementById('emptyState');
    const resultsContent = document.getElementById('resultsContent');
    const resultCard = document.getElementById('resultCard');

    const gaugeFill = document.getElementById('gaugeFill');
    const riskPctText = document.getElementById('riskPct');
    const riskLevelLabel = document.getElementById('riskLevelLabel');

    const riskSummaryBox = document.getElementById('riskSummaryBox');
    const summaryHeading = document.getElementById('summaryHeading');
    const summaryText = document.getElementById('summaryText');
    const recommendationList = document.getElementById('recommendationList');

    const GAUGE_CIRCUMFERENCE = 597;

    form.addEventListener('submit', async (e) => {
        e.preventDefault();

        btnSubmit.disabled = true;
        btnLoader.style.display = 'block';
        btnSubmit.querySelector('.btn-text').style.opacity = '0.5';

        const payload = {
            gender: document.getElementById('gender').value,
            age: parseFloat(document.getElementById('age').value),
            bmi: parseFloat(document.getElementById('bmi').value),
            smoking_history: document.getElementById('smoking_history').value,
            HbA1c_level: parseFloat(document.getElementById('HbA1c_level').value),
            blood_glucose_level: parseInt(document.getElementById('blood_glucose_level').value, 10),
            hypertension: document.getElementById('hypertension').checked ? 1 : 0,
            heart_disease: document.getElementById('heart_disease').checked ? 1 : 0
        };

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(payload)
            });

            const result = await response.json();

            if (!response.ok) {
                throw new Error(result.error || 'Server error occurred');
            }

            displayResults(result);

        } catch (error) {
            console.error('Prediction error:', error);
            alert(`Diagnostics Failed: ${error.message}`);
        } finally {
            btnSubmit.disabled = false;
            btnLoader.style.display = 'none';
            btnSubmit.querySelector('.btn-text').style.opacity = '1';
        }
    });

    function displayResults(data) {
        emptyState.classList.add('hidden');
        resultsContent.classList.remove('hidden');

        const probability = data.probability;
        const pct = Math.round(probability * 100);

        animateValue(riskPctText, 0, pct, 800);

        const offset = GAUGE_CIRCUMFERENCE - (probability * GAUGE_CIRCUMFERENCE);
        gaugeFill.style.strokeDashoffset = offset;

        let colorClass = 'risk-low';
        let strokeColor = '#00f2a7';
        let headingText = 'Low Risk Profile';
        let descText = 'Based on the parameters analyzed by the Random Forest model, the patient is within a low-risk threshold. Continue monitoring.';

        if (data.risk_level === 'Moderate') {
            colorClass = 'risk-med';
            strokeColor = '#ffaa00';
            headingText = 'Moderate Risk Profile';
            descText = 'The analysis reports mild elevations in bio-markers or pre-existing clinical risk parameters. Increased vigilance is advised.';
        } else if (data.risk_level === 'High') {
            colorClass = 'risk-high';
            strokeColor = '#ff2a5f';
            headingText = 'High Risk Profile (Diabetic Likely)';
            descText = 'Multiple indicators exceed normal physiological ranges. Prompt medical consultation is strongly advised for further screening.';
        }

        riskLevelLabel.textContent = `${data.risk_level} Risk`;
        gaugeFill.style.stroke = strokeColor;

        riskSummaryBox.className = `risk-summary ${colorClass}`;
        summaryHeading.textContent = headingText;
        summaryText.textContent = descText;

        recommendationList.innerHTML = '';
        data.recommendations.forEach(rec => {
            const li = document.createElement('li');
            li.textContent = rec;
            recommendationList.appendChild(li);
        });

        resultCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }

    function animateValue(obj, start, end, duration) {
        let startTimestamp = null;
        const step = (timestamp) => {
            if (!startTimestamp) startTimestamp = timestamp;
            const progress = Math.min((timestamp - startTimestamp) / duration, 1);
            obj.innerHTML = Math.floor(progress * (end - start) + start) + "%";
            if (progress < 1) {
                window.requestAnimationFrame(step);
            }
        };
        window.requestAnimationFrame(step);
    }
});
