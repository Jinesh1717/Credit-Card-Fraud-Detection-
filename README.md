# ML-03 Credit Card Fraud Detection

Reproducible FinTech fraud-risk pipeline with schema validation, stratified splitting, leakage-safe preprocessing, imbalance strategies, PR-AUC evaluation, cost-based thresholding, CLI scoring, and Streamlit review workflow.

## Run

```bash
pip install -r requirements.txt
python -m src.train
python -m src.score --input sample.csv --output scored.csv
streamlit run app/streamlit_app.py
pytest -q
```

Place Kaggle's `creditcard.csv` in `data/raw/` (it is intentionally not tracked). If it is absent, training uses a labeled synthetic dataset only to validate code paths. Fraud is highly imbalanced, so accuracy is misleading; compare precision, recall, F2, PR-AUC, MCC, false positives and false negatives. The project is a demonstration: PCA-anonymized features, a short historical window, concept drift, unknown real costs, and customer friction limit production conclusions.
