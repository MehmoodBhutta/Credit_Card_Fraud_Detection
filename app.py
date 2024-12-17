import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt

# Set up the Streamlit app
st.title("Credit Card Fraud Detection")
st.sidebar.title("Navigation")
st.sidebar.header("Upload Dataset")

# File upload
uploaded_file = st.sidebar.file_uploader("Upload your credit card dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    # Load dataset
    df = pd.read_csv(uploaded_file)
    st.write("### Dataset Overview")
    st.write(df.head())

    # Display dataset information
    st.write("### Dataset Summary")
    st.write(df.describe())

    # EDA Section
    st.sidebar.header("EDA Options")
    eda_option = st.sidebar.radio("Select an option", ("Class Distribution", "Correlation Heatmap", "Feature Distributions"))

    if eda_option == "Class Distribution":
        st.write("### Class Distribution")
        class_counts = df['Class'].value_counts()
        fig, ax = plt.subplots()
        class_counts.plot(kind='bar', ax=ax, color=['skyblue', 'orange'])
        plt.title("Class Distribution")
        plt.xlabel("Class (0 = Non-Fraud, 1 = Fraud)")
        plt.ylabel("Count")
        st.pyplot(fig)

    elif eda_option == "Correlation Heatmap":
        st.write("### Correlation Heatmap")
        corr = df.corr()
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(corr, cmap='coolwarm', annot=False, ax=ax)
        st.pyplot(fig)

    elif eda_option == "Feature Distributions":
        selected_feature = st.sidebar.selectbox("Select a feature to visualize", df.columns)
        st.write(f"### Distribution of {selected_feature}")
        fig, ax = plt.subplots()
        sns.histplot(df[selected_feature], bins=50, kde=True, ax=ax)
        st.pyplot(fig)

    # Data Preprocessing
    st.write("### Data Preprocessing")
    df_cleaned = df.drop_duplicates()
    st.write(f"Removed duplicates. Remaining rows: {len(df_cleaned)}")

    X = df_cleaned.drop(columns=['Class'])
    y = df_cleaned['Class']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train[['Time', 'Amount']] = scaler.fit_transform(X_train[['Time', 'Amount']])
    X_test[['Time', 'Amount']] = scaler.transform(X_test[['Time', 'Amount']])

    st.write("Data split into training and test sets, and scaled successfully.")

    # Model Selection
    st.sidebar.header("Model Options")
    model_choice = st.sidebar.selectbox("Choose a model", ["Logistic Regression", "Random Forest"])

    if model_choice == "Logistic Regression":
        model = LogisticRegression(random_state=42, max_iter=1000)
    elif model_choice == "Random Forest":
        n_estimators = st.sidebar.slider("Number of Trees", min_value=10, max_value=200, value=100, step=10)
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)

    # Model Training
    st.write(f"### Training {model_choice}")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Metrics
    st.write("### Model Evaluation")
    st.write("Classification Report:")
    st.text(classification_report(y_test, y_pred))

    st.write("ROC-AUC Score:")
    st.write(roc_auc_score(y_test, y_proba))

    # Confusion Matrix
    st.write("### Confusion Matrix")
    conf_matrix = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-Fraud', 'Fraud'], yticklabels=['Non-Fraud', 'Fraud'])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    st.pyplot(fig)

    # Predict on Custom Input
    st.sidebar.header("Fraud Prediction on Custom Input")
    time_input = st.sidebar.number_input("Time (seconds)", min_value=0.0, max_value=172800.0, value=0.0)
    amount_input = st.sidebar.number_input("Transaction Amount", min_value=0.0, value=100.0)
    custom_data = pd.DataFrame([[time_input, amount_input]], columns=["Time", "Amount"])
    custom_data_scaled = scaler.transform(custom_data)
    custom_pred = model.predict(custom_data_scaled)

    st.sidebar.write("### Fraud Prediction Result")
    st.sidebar.write("Prediction: Fraud" if custom_pred[0] == 1 else "Prediction: Non-Fraud")

else:
    st.write("Upload a dataset to get started.")
