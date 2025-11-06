# Model Card: Credit Risk Prediction

## Model Overview
- **Type:** Gradient Boosted Decision Trees (XGBoost)
- **Purpose:** Predict probability of loan default for consumer credit applicants
- **Inputs:** Demographics, financial history, credit bureau data
- **Outputs:** Risk score (0-1), explainability report

## Performance Metrics
- **AUC-ROC:** 0.92 (industry benchmark: 0.85)
- **Accuracy:** 89%
- **Recall (Default):** 0.81
- **Precision (Default):** 0.77
- **F1 Score:** 0.79
- **Coverage:** 100% of applicants

## Fairness & Bias
- Model tested for bias across age, gender, and region
- No statistically significant disparate impact detected
- Regular fairness audits recommended

## Explainability
- SHAP values provided for every prediction
- Feature importance visualizations available
- Full audit trail for regulatory compliance

## Model Lifecycle
- **Training Data:** 50,000+ anonymized credit applications
- **Retraining Frequency:** Quarterly or as needed
- **Monitoring:** Automated drift detection and alerting

## Limitations
- Not suitable for commercial lending without retraining
- Requires up-to-date applicant data for best results

## Responsible AI Statement
This model is designed and maintained according to best practices in responsible AI, including transparency, fairness, and ongoing monitoring.

---