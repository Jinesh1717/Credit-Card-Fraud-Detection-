# Business summary

The pipeline detects rare card fraud using PR-AUC as the primary metric and selects a review threshold on validation data using configurable false-alarm and missed-fraud costs. Run `python -m src.train` to generate current metrics. The default fallback is synthetic and must not be used for business decisions; replace it with Kaggle's `creditcard.csv`. Production use requires drift monitoring, human review controls, privacy protection, and periodic retraining.
