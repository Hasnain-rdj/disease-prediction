import streamlit as st
import pandas as pd
import numpy as np
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from scipy.sparse import hstack, csr_matrix

st.title("Disease Classification: KNN & Logistic Regression Experiments")

@st.cache_data
def load_data():
    df = pd.read_csv("disease_features.csv")
    encoded_df = pd.read_csv("encoded_output2.csv")
    return df, encoded_df

df, encoded_df = load_data()

def parse_and_join(text):
    lst = ast.literal_eval(text)
    return " ".join(lst)

for col in ["Risk Factors", "Symptoms", "Signs"]:
    df[col] = df[col].apply(parse_and_join)

tfidf_rf = TfidfVectorizer()
tfidf_symptoms = TfidfVectorizer()
tfidf_signs = TfidfVectorizer()
matrix_rf = tfidf_rf.fit_transform(df["Risk Factors"])
matrix_symptoms = tfidf_symptoms.fit_transform(df["Symptoms"])
matrix_signs = tfidf_signs.fit_transform(df["Signs"])
tfidf_matrix = hstack([matrix_rf, matrix_symptoms, matrix_signs])

onehot_numeric = encoded_df.apply(pd.to_numeric, errors="coerce").fillna(0).values.astype(float)
onehot_sparse = csr_matrix(onehot_numeric)

if "Category" in df.columns:
    target = df["Category"]
elif "Subtype" in df.columns:
    target = df["Subtype"]
else:
    st.error("No target column found for classification. Please include 'Category' or 'Subtype' in your dataset.")
    target = pd.Series(["unknown"] * df.shape[0])

def run_knn_experiments(matrix, target, encoding_label):
    ks = [3, 5, 7]
    distance_metrics = {"Euclidean": "euclidean", "Manhattan": "manhattan", "Cosine": "cosine"}
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = []
    for dist_name, metric in distance_metrics.items():
        for k in ks:
            if metric == "cosine":
                knn = KNeighborsClassifier(n_neighbors=k, metric=metric, algorithm="brute")
            else:
                knn = KNeighborsClassifier(n_neighbors=k, metric=metric)
            scoring = {
                "accuracy": "accuracy",
                "precision": "precision_macro",
                "recall": "recall_macro",
                "f1": "f1_macro"
            }
            scores = cross_validate(knn, matrix, target, cv=skf, scoring=scoring, n_jobs=-1)
            results.append({
                "Model": "KNN",
                "Encoding": encoding_label,
                "K": k,
                "Distance Metric": dist_name,
                "Accuracy": np.mean(scores["test_accuracy"]),
                "Precision": np.mean(scores["test_precision"]),
                "Recall": np.mean(scores["test_recall"]),
                "F1-Score": np.mean(scores["test_f1"])
            })
    return pd.DataFrame(results)

def run_logreg_experiments(matrix, target, encoding_label):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    logreg = LogisticRegression(max_iter=1000, solver="liblinear")
    scoring = {"accuracy": "accuracy", "f1": "f1_macro"}
    scores = cross_validate(logreg, matrix, target, cv=skf, scoring=scoring, n_jobs=-1)
    return {
        "Model": "Logistic Regression",
        "Encoding": encoding_label,
        "Accuracy": np.mean(scores["test_accuracy"]),
        "F1-Score": np.mean(scores["test_f1"])
    }

st.header("KNN Experiments")
knn_results_tfidf = run_knn_experiments(tfidf_matrix, target, "TF-IDF")
knn_results_onehot = run_knn_experiments(onehot_numeric, target, "One-hot")
knn_results = pd.concat([knn_results_tfidf, knn_results_onehot], ignore_index=True)
st.dataframe(knn_results)

st.header("Logistic Regression Experiments")
logreg_result_tfidf = run_logreg_experiments(tfidf_matrix, target, "TF-IDF")
logreg_result_onehot = run_logreg_experiments(onehot_numeric, target, "One-hot")
logreg_results = pd.DataFrame([logreg_result_tfidf, logreg_result_onehot])
st.dataframe(logreg_results)

st.header("Summary Comparison")
st.write("The tables above provide the cross-validated metrics for each configuration. You can compare results across:")
st.write("- **Encoding Methods:** TF-IDF vs. One-hot")
st.write("- **Distance Metrics (KNN only):** Euclidean, Manhattan, and Cosine")
st.write("- **Model Types:** KNN vs. Logistic Regression")
st.write("Generally, higher Accuracy and F1-Score indicate better performance. Analyze which configurations yield the best separability for your disease classification task.")
