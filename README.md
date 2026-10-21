# Online Shoppers Purchase Intention Prediction

## Project Overview

This project uses machine learning to predict whether an online shopping session is likely to result in a purchase.

The application accepts visitor session details and returns a predicted purchase outcome along with estimated probabilities.

## Dataset

The project uses the Online Shoppers Purchasing Intention dataset.

## Machine Learning Models

The following models were evaluated:

* Logistic Regression
* K-Nearest Neighbors (KNN)
* Support Vector Machine (SVM)
* Decision Tree
* Random Forest
* Gradient Boosting

## Selected Model

Gradient Boosting was selected based on its F1-score, recall, and ROC-AUC results.

Reported test-set results:

* Accuracy: 90.25%
* F1-score: 67.13%
* ROC-AUC: 93.63%

## Technologies Used

* Python
* Pandas
* Scikit-learn
* Joblib
* Gradio

## Application Features

* Accepts online shopping session details.
* Predicts whether a purchase is likely.
* Displays estimated probabilities for purchase and no purchase.

## Run Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the application:

```bash
python app.py
```

Then open the local URL displayed in the terminal.

## Project Files

* `app.py` — Gradio application
* `requirements.txt` — Python dependencies
* `gradient_boosting_model.pkl` — Trained model
* `preprocessor.pkl` — Fitted preprocessing pipeline
* `README.md` — Project documentation

## Disclaimer

Predictions are estimates produced by a machine learning model and are not guarantees of customer behavior.
