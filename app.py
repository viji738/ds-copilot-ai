import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

st.title("DS Copilot AI")

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

# ------------------ FILE CHECK ------------------
if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    # 🔥 SPEED FIX (optional but important)
    df = df.sample(min(1000, len(df)), random_state=42)

    st.subheader("Dataset Preview")
    st.write(df.head())

    st.subheader("Dataset Information")
    st.write("Rows:", df.shape[0])
    st.write("Columns:", df.shape[1])

    st.subheader("Missing Values")
    st.write(df.isnull().sum())

    st.subheader("Data Types")
    st.write(df.dtypes)

    # ------------------ SUMMARY ------------------
    missing = df.isnull().sum().sum()

    if missing == 0:
        missing_text = "No missing values detected."
    else:
        missing_text = f"{missing} missing values found."

    st.info(
        f"This dataset contains {df.shape[0]} rows and {df.shape[1]} columns. "
        f"{missing_text} "
        "The dataset is ready for machine learning analysis."
    )

    # ------------------ TARGET ------------------
    target = st.selectbox("Select Target Column", df.columns)

    if df[target].dtype == 'object':
        st.success("Problem Type: Classification")
        st.write("Suggested Models:")
        st.write("- Logistic Regression")
        st.write("- Random Forest Classifier")
    else:
        st.success("Problem Type: Regression")
        st.write("Suggested Models:")
        st.write("- Linear Regression")
        st.write("- Random Forest Regressor")

    st.info("AI Insight: Random Forest works well for complex datasets.")

    # ------------------ TRAIN MODEL ------------------
    if st.button("Train Model"):

        # 🔥 spinner ADDED (IMPORTANT FIX)
        with st.spinner("Training models... Please wait ⏳"):

            X = df.drop(columns=[target])
            X = pd.get_dummies(X)
            X = X.fillna(0)
            y = df[target].fillna(0)

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            # Linear Regression
            lr_model = LinearRegression()
            lr_model.fit(X_train, y_train)
            lr_predictions = lr_model.predict(X_test)
            lr_score = r2_score(y_test, lr_predictions)

            # 🔥 OPTIMIZED Random Forest
            rf_model = RandomForestRegressor(
                n_estimators=50,
                random_state=42
            )
            rf_model.fit(X_train, y_train)
            rf_predictions = rf_model.predict(X_test)
            rf_score = r2_score(y_test, rf_predictions)

        st.success("Models Trained Successfully! 🚀")

        st.write("Linear Regression Score:", lr_score)
        st.write("Random Forest Score:", rf_score)

        if lr_score > rf_score:
            st.info("Best Model: Linear Regression")
        else:
            st.info("Best Model: Random Forest")

        # ------------------ MODEL COMPARISON ------------------
        comparison_df = pd.DataFrame({
            "Model": ["Linear Regression", "Random Forest"],
            "Score": [lr_score, rf_score]
        })

        st.subheader("Model Comparison")
        st.bar_chart(comparison_df.set_index("Model"))

        csv = comparison_df.to_csv(index=False)

        st.download_button(
            label="Download Model Results",
            data=csv,
            file_name="model_results.csv",
            mime="text/csv"
        )

        # ------------------ VISUALIZATION ------------------
        st.subheader("Data Visualization")

numeric_df = df.select_dtypes(include=['number'])

if numeric_df.shape[1] >= 2:

    fig, ax = plt.subplots()

    ax.scatter(numeric_df.iloc[:, 0], numeric_df.iloc[:, 1])

    ax.set_xlabel(numeric_df.columns[0])
    ax.set_ylabel(numeric_df.columns[1])

    st.pyplot(fig)

else:
    st.warning("Not enough numeric columns for scatter plot")