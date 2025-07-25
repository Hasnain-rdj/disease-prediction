import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, KFold
from sklearn.metrics import precision_score, recall_score, f1_score, make_scorer
from sklearn.preprocessing import normalize, StandardScaler, Normalizer
from sklearn.pipeline import make_pipeline

# --------------------
# Label Generator Function
# --------------------
def generate_disease_labels(df, output_path="disease_labels.csv"):
    if "Disease" in df.columns:
        labels = df["Disease"]
    elif "Subtypes" in df.columns:
        labels = df["Subtypes"]
    else:
        labels = pd.Series(["unknown"] * df.shape[0])

    max_labels = 50
    if labels.nunique() > max_labels:
        top_labels = labels.value_counts().index[:max_labels]
        labels = labels.apply(lambda x: x if x in top_labels else "Other")

    labels_df = pd.DataFrame({"Label": labels})
    labels_df.to_csv(output_path, index=False)
    return labels_df["Label"]

# --------------------
# Data Loading Functions
# --------------------
@st.cache_data
def load_csv(file_path):
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        st.error(f"Error loading {file_path}: {e}")
        return None

def load_data():
    df_text = load_csv("disease_features.csv")
    df_onehot = load_csv("encoded_output2.csv")
    if df_text is None or df_onehot is None:
        st.stop()
    return df_text, df_onehot

# --------------------
# Data Preparation
# --------------------
def prepare_text_data(df_text, text_col, label_col):
    vectorizer = TfidfVectorizer()
    X_tfidf = vectorizer.fit_transform(df_text[text_col])
    y = df_text[label_col]
    return X_tfidf, y

def prepare_onehot_data(df_onehot, label_col):
    X_onehot = df_onehot.drop(label_col, axis=1)
    # Select only numeric columns (float or int)
    numeric_cols = X_onehot.select_dtypes(include=[np.number]).columns
    dropped_cols = set(X_onehot.columns) - set(numeric_cols)
    if dropped_cols:
        st.warning(f"Dropped non-numeric columns in one-hot data: {dropped_cols}")
    X_onehot = X_onehot[numeric_cols]
    y = df_onehot[label_col]
    return X_onehot, y

# --------------------
# Model Evaluation
# --------------------
def check_for_nans(X):
    """Optional helper to remove or replace NaNs or inf."""
    if hasattr(X, "toarray"):  # it's sparse
        X_arr = X.toarray()
    else:
        X_arr = np.array(X)

    # If needed, you can replace with 0 or drop them
    # In this example, we only check but do not fix:
    if np.isnan(X_arr).any() or np.isinf(X_arr).any():
        raise ValueError("Data contains NaN or inf. Fix or remove them before training.")
    return X  # or X_arr if you want to directly convert

def compute_knn_results(X, y, k_values, metrics_list, cv, scoring):
    """
    Runs KNN with cross-validation for each combination of k in k_values and
    distance metric in metrics_list, returning a dictionary of results.
    Uses scikit-learn pipelines to ensure proper preprocessing for each metric.
    """
    from sklearn.preprocessing import StandardScaler, Normalizer, LabelEncoder
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import cross_validate
    from sklearn.pipeline import make_pipeline
    
    # Check for NaNs or infinite values
    check_for_nans(X)
    
    # Encode string labels to integers for KNN
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    results = {}
    for metric in metrics_list:
        for k in k_values:
            try:                # Initialize with default values in case of error
                results[(k, metric)] = {m: 0.0 for m in scoring}
                # Check if data is sparse
                is_sparse = hasattr(X, "toarray")
                
                if metric == "manhattan":
                    # For manhattan, handle sparse matrices differently
                    if is_sparse:
                        # Always use StandardScaler without centering for sparse matrices
                        pipeline = make_pipeline(
                            StandardScaler(with_mean=False),
                            KNeighborsClassifier(n_neighbors=k, metric="minkowski", p=1)
                        )
                        cv_results = cross_validate(pipeline, X, y_encoded, cv=cv, scoring=scoring)
                    else:
                        # For dense data, use regular StandardScaler
                        pipeline = make_pipeline(
                            StandardScaler(),
                            KNeighborsClassifier(n_neighbors=k, metric="minkowski", p=1)
                        )
                        cv_results = cross_validate(pipeline, X, y_encoded, cv=cv, scoring=scoring)
                    # Store results and continue to next iteration
                    results[(k, metric)] = {m: np.mean(cv_results[f'test_{m}']) for m in scoring}
                    continue
                elif metric == "cosine":
                    # For cosine, use normalization
                    pipeline = make_pipeline(
                        Normalizer(),
                        KNeighborsClassifier(n_neighbors=k, metric=metric)
                    )
                else:
                    # For euclidean and other metrics
                    pipeline = make_pipeline(
                        KNeighborsClassifier(n_neighbors=k, metric=metric)
                    )
                
                # Run cross-validation with the pipeline
                cv_results = cross_validate(pipeline, X, y_encoded, cv=cv, scoring=scoring)
                results[(k, metric)] = {m: np.mean(cv_results[f'test_{m}']) for m in scoring}
                
            except Exception as e:
                # Log error for debugging
                st.error(f"[ERROR] KNN: k={k}, metric={metric} => {type(e).__name__}: {e}")
                # Default values were already initialized above
                pass
    
    return results

def compute_logistic_results(X, y, cv, scoring):
    from sklearn.preprocessing import LabelEncoder
    
    # Encode string labels to integers for Logistic Regression
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    lr = LogisticRegression(max_iter=1000)
    cv_results = cross_validate(lr, X, y_encoded, cv=cv, scoring=scoring)
    return {m: np.mean(cv_results[f'test_{m}']) for m in scoring}

def format_results_dict(results_dict):
    rows = []
    for (k, metric), metrics in results_dict.items():
        row = {"k": k, "Distance Metric": metric}
        row.update(metrics)
        rows.append(row)
    return pd.DataFrame(rows)

def plot_metric_bar_chart(df, metric, title):
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Define consistent colors for each distance metric
    colors = {'euclidean': 'blue', 'manhattan': 'red', 'cosine': 'green'}
    
    # Get unique k values and metrics for positioning
    k_values = sorted(df["k"].unique())
    metrics = sorted(df["Distance Metric"].unique())
    
    # Calculate positions for grouped bars
    bar_width = 0.25
    x = np.arange(len(k_values))
    
    # Plot each metric side by side
    for i, dist_metric in enumerate(metrics):
        metric_data = df[df["Distance Metric"] == dist_metric]
        values = []
        for k in k_values:
            row = metric_data[metric_data["k"] == k]
            if not row.empty:
                values.append(row[metric].values[0])
            else:
                values.append(0)
        
        # Position the bars side by side
        pos = x + (i - 1) * bar_width
        ax.bar(
            pos, 
            values, 
            bar_width,
            label=dist_metric,
            color=colors.get(dist_metric, 'gray')
        )
    
    # Set the x-axis labels
    ax.set_xticks(x)
    ax.set_xticklabels([str(k) for k in k_values])
    
    ax.set_title(title)
    ax.set_xlabel("k Value")
    ax.set_ylabel(metric.capitalize())
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.6)
    st.pyplot(fig)

# --------------------
# Main Streamlit App
# --------------------
def main():
    st.set_page_config(page_title="Disease Model Evaluation", layout="wide")
    st.title("Disease Model Evaluation")
    
    df_text, df_onehot = load_data()
    generate_disease_labels(df_text)
    
    with st.sidebar:
        st.title("Configuration")

        with st.expander("1. Select Data Columns"):
            text_col = st.selectbox("Text Column", df_text.columns.tolist())
            label_col_text = st.selectbox("Label Column (Text File)", df_text.columns.tolist())
            label_col_onehot = st.selectbox("Label Column (One-Hot File)", df_onehot.columns.tolist())

        with st.expander("2. Filter by Disease"):
            diseases = sorted(df_text[label_col_text].unique().tolist())
            selected_diseases = st.multiselect("Select Diseases", diseases, default=diseases)
            df_text = df_text[df_text[label_col_text].isin(selected_diseases)]
            df_onehot = df_onehot[df_onehot[label_col_onehot].isin(selected_diseases)]

        with st.expander("3. Model and Encoding Options"):
            encoding_option = st.radio("Encoding", ["TF‑IDF", "One‑hot", "Both"])
            model_option = st.radio("Model", ["KNN", "Logistic Regression", "Both"])

        with st.expander("🔍 Data Preview", expanded=False):
            st.write("**Text Data Sample:**")
            st.dataframe(df_text.head())
            st.write("**One-Hot Encoded Sample:**")
            st.dataframe(df_onehot.head())

    X_tfidf, y_tfidf = None, None
    X_onehot, y_onehot = None, None    
    if encoding_option in ("TF‑IDF", "Both"):
        X_tfidf, y_tfidf = prepare_text_data(df_text, text_col, label_col_text)
        st.write("TF‑IDF matrix shape:", X_tfidf.shape)
    if encoding_option in ("One‑hot", "Both") and df_onehot.shape[0] >= 5:
        X_onehot, y_onehot = prepare_onehot_data(df_onehot, label_col_onehot)
        st.write("One‑hot features shape:", X_onehot.shape)
        
    k_values = [3, 5, 7]
    distance_metrics = ["euclidean", "manhattan", "cosine"]
    scoring = {
        "accuracy": "accuracy",
        "precision": make_scorer(precision_score, average="weighted", zero_division=0, pos_label=None),
        "recall": make_scorer(recall_score, average="weighted", zero_division=0, pos_label=None),
        "f1": make_scorer(f1_score, average="weighted", zero_division=0)
    }
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    tabs = st.tabs(["KNN Model", "Logistic Regression", "Comparison"])

    with tabs[0]:
        st.header("KNN Model Results")
        # TF‑IDF Results
        if encoding_option in ("TF‑IDF", "Both") and model_option in ("KNN", "Both"):
            st.subheader("TF‑IDF")
            results = compute_knn_results(X_tfidf, y_tfidf, k_values, distance_metrics, cv, scoring)
            df_results = format_results_dict(results)
            st.dataframe(df_results)
            plot_metric_bar_chart(df_results, "accuracy", "KNN Accuracy (TF‑IDF)")

        # One‑hot Results
        if encoding_option in ("One‑hot", "Both") and model_option in ("KNN", "Both") and X_onehot is not None:
            st.subheader("One‑hot")
            results = compute_knn_results(X_onehot, y_onehot, k_values, distance_metrics, cv, scoring)
            df_results = format_results_dict(results)
            st.dataframe(df_results)
            plot_metric_bar_chart(df_results, "accuracy", "KNN Accuracy (One‑hot)")

    with tabs[1]:
        st.header("Logistic Regression Results")
        # TF‑IDF Results
        if encoding_option in ("TF‑IDF", "Both") and model_option in ("Logistic Regression", "Both"):
            st.subheader("TF‑IDF")
            results = compute_logistic_results(X_tfidf, y_tfidf, cv, scoring)
            st.dataframe(pd.DataFrame([results]))

        # One‑hot Results
        if encoding_option in ("One‑hot", "Both") and model_option in ("Logistic Regression", "Both") and X_onehot is not None:
            st.subheader("One‑hot")
            results = compute_logistic_results(X_onehot, y_onehot, cv, scoring)
            st.dataframe(pd.DataFrame([results]))

    with tabs[2]:
        st.header("Summary & Comparison")
        st.markdown("Compare results of models using different encodings.")
        st.info("Use the tabs above to explore KNN and Logistic Regression results.")

if __name__ == '__main__':
    main()
