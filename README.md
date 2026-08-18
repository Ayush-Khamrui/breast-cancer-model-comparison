# ML Assignment 2 - Classification Model Comparison

**Name:** Ayush Khamrui  
**Student ID:** 2025AC05152
- **GitHub repository:** [breast-cancer-model-comparison](https://github.com/Ayush-Khamrui/breast-cancer-model-comparison)
- **Live Streamlit application:** [Diagnostic Model Studio](https://2025ac05152ayushkhamrui.streamlit.app/)
  
<img width="1710" height="1075" alt="2026-08-18_22-12-14" src="https://github.com/user-attachments/assets/bf73a54b-31df-402c-9368-34ba7b6a6ccc" />

## a. Problem statement

For this assignment, I trained and compared six classification models using the same dataset. I evaluated each model using Accuracy, AUC, Precision, Recall, F1 Score, and Matthews Correlation Coefficient (MCC).

I also created a Streamlit application where I can upload test data, select a model, view its evaluation scores, and check the confusion matrix and classification report.

The assignment asks for six models, but it specifically lists only five. I implemented all five listed models and used Support Vector Machine (SVM) as the additional sixth model.

## b. Dataset description

I used the Breast Cancer Wisconsin (Diagnostic) dataset. I chose it because it is a clear binary classification problem and it satisfies the assignment requirement of at least 500 instances and 12 features. It also has no missing values, so I could focus on model comparison instead of spending most of the work on data cleaning.

The classification task is also easy to understand: the model predicts whether a tumour is malignant or benign. This made metrics such as Recall and Precision meaningful while comparing the models.

- **Source:** UCI Machine Learning Repository, loaded using `sklearn.datasets.load_breast_cancer`
- **Number of instances:** 569
- **Number of features:** 30 numeric features
- **Target values:** `0 = malignant` and `1 = benign`
- **Training and test split:** 80% training and 20% testing
- **Training rows:** 455
- **Test rows:** 114
- **Random state:** 42
- **Missing values:** None

I used a stratified split so that the proportion of malignant and benign cases remained similar in both the training and test data.

## c. Links

- **GitHub repository:** I will add the public repository link here after pushing the final version.
- **Live Streamlit application:** I will add the deployed application link here after deployment.

## d. Models used and results

I trained every model on the same training data and evaluated it on the same test data. For Logistic Regression, kNN, Naive Bayes, and SVM, I used `StandardScaler` inside a pipeline. This ensures that scaling is learned only from the training data.

For AUC, Precision, Recall, and F1 Score, I treated class `1` (benign) as the positive class.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9825 | 0.9957 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |
| Decision Tree | 0.9035 | 0.9373 | 0.9420 | 0.9028 | 0.9220 | 0.7969 |
| kNN | 0.9737 | 0.9884 | 0.9600 | 1.0000 | 0.9796 | 0.9442 |
| Naive Bayes | 0.9298 | 0.9868 | 0.9444 | 0.9444 | 0.9444 | 0.8492 |
| Random Forest (Ensemble) | 0.9474 | 0.9940 | 0.9583 | 0.9583 | 0.9583 | 0.8869 |
| Support Vector Machine (Additional) | 0.9737 | 0.9947 | 0.9859 | 0.9722 | 0.9790 | 0.9439 |

## My observations

### Logistic Regression

Logistic Regression gave me the best overall result. It had the highest Accuracy, AUC, F1 Score, and MCC. Its Precision and Recall were also almost equal, which shows that its predictions were well balanced on this test data.

### Decision Tree

The Decision Tree gave the lowest result among the six models. It was easy to interpret, but its lower Recall and MCC show that it did not generalize as well as the other models on this split.

### k-Nearest Neighbours

kNN achieved a Recall of 1.0000 for the benign class. However, its Precision was slightly lower than Logistic Regression, which means it made a few more false-benign predictions.

### Naive Bayes

Naive Bayes produced a good AUC score, but its Accuracy and MCC were lower than Logistic Regression and kNN. One possible reason is that many of the measurements in this dataset are related to each other, while Naive Bayes assumes that features are independent.

### Random Forest

Random Forest gave the same Precision and Recall of 0.9583 and also had a high AUC score. It performed well, but on this particular test split it did not perform better than Logistic Regression or kNN.

### Support Vector Machine

SVM achieved 97.37% Accuracy and a high AUC score. Its result was close to kNN, although its F1 Score was slightly lower.

### Overall result

I selected Logistic Regression as the overall winner because it produced the best combination of Accuracy, AUC, F1 Score, and MCC on my test data.

## What I found challenging

The main challenge was making the comparison fair. Some models work better after feature scaling, while tree-based models do not need it. I used pipelines for the models that require scaling and kept the same train-test split for all six models.

Another challenge was handling uploaded CSV files in the Streamlit app. I added checks for missing columns, invalid target values, non-numeric data, empty values, infinity, file size, and number of rows. This prevents the application from failing with an unclear error when the uploaded file is not in the expected format.

## Streamlit application

The application provides the following features:

- Upload test data in CSV format
- Download the sample test data
- Select any one of the six models
- Compare the results of all models
- View Accuracy, AUC, Precision, Recall, F1 Score, and MCC
- View the confusion matrix and classification report
- Preview and download model predictions

The uploaded CSV is checked before it is passed to a model. Uploads are limited to 5 MB and 5,000 rows. The application uses only the required feature columns and does not save uploaded data to disk.

## Files in the repository

```text
ml_assignment_2/
|-- app.py
|-- requirements.txt
|-- README.md
|-- test_data.csv
|-- model/
|   |-- __init__.py
|   `-- train_models.py
|-- .streamlit/
|   `-- config.toml
`-- .gitignore
```

## Running the project on the BITS Virtual Lab VM

From the project folder, I use the following commands:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run app.py
```

Streamlit normally displays the application at `http://localhost:8501`. I only need to run `app.py` because it automatically calls the training code from `model/train_models.py` and loads the included `test_data.csv`.

If I want to view all model scores directly in the terminal, I can also run:

```bash
python model/train_models.py
```

## Before final submission

Before submitting the assignment, I still need to:

1. Add the final GitHub repository link.
2. Deploy the application and add its Streamlit link.
3. Run the project on the BITS Virtual Lab and capture the required screenshot.
4. Include the links, screenshot, and this README content in the final PDF.
