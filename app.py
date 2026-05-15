import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

st.title("DS Copilot AI 🤖")

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

# ------------------ FILE CHECK ------------------
if uploaded_file is not None:

    try:
        df = pd.read_csv(uploaded_file)

        # limit rows for speed
        df = df.sample(min(1000, len(df)), random_state=42)

        # ------------------ DATASET PREVIEW ------------------
        st.subheader("Dataset Preview")
        st.write(df.head())

        # ------------------ DATASET INFO ------------------
        st.subheader("Dataset Information")
        st.write("Rows:", df.shape[0])
        st.write("Columns:", df.shape[1])

        # ------------------ MISSING VALUES ------------------
        st.subheader("Missing Values")
        st.write(df.isnull().sum())

        # ------------------ DATA TYPES ------------------
        st.subheader("Data Types")
        st.write(df.dtypes)

        # ------------------ AI SUMMARY ------------------
        st.subheader("AI Dataset Summary 🤖")

        missing = df.isnull().sum().sum()

        if missing == 0:
            st.success("Dataset looks clean with no missing values.")
        else:
            st.warning(f"Dataset contains {missing} missing values.")

        st.info(
            f"The dataset contains {df.shape[0]} rows and "
            f"{df.shape[1]} columns."
        )

        # ------------------ TARGET COLUMN ------------------
        target = st.selectbox(
            "Select Target Column",
            df.columns
        )

        # ------------------ MODEL SUGGESTION ------------------
        st.subheader("Suggested Models")

        if df[target].dtype == 'object':
            st.write("- Logistic Regression")
            st.write("- Random Forest Classifier")
        else:
            st.write("- Linear Regression")
            st.write("- Random Forest Regressor")

        # ------------------ TRAIN BUTTON ------------------
        if st.button("Train Model"):

            with st.spinner("Training models... Please wait ⏳"):

                try:

                    # ------------------ CLEAN DATA ------------------
                    X = df.drop(columns=[target])

                    # convert categorical values
                    X = pd.get_dummies(X)

                    # fill missing values
                    X = X.fillna(0)

                    # clean column names
                    X.columns = [
                        re.sub(r'[\[\]<>]', '', str(col))
                        for col in X.columns
                    ]

                    # keep only numeric columns
                    X = X.select_dtypes(include=['number'])

                    # target column
                    y = df[target]

                    # fill missing target safely
                    if y.dtype == 'object':
                        y = y.fillna(y.mode()[0])
                    else:
                        y = y.fillna(y.median())

                    # ------------------ SPLIT DATA ------------------
                    X_train, X_test, y_train, y_test = train_test_split(
                        X,
                        y,
                        test_size=0.2,
                        random_state=42
                    )

                    # ------------------ LINEAR REGRESSION ------------------
                    lr = LinearRegression()

                    lr.fit(X_train, y_train)

                    lr_predictions = lr.predict(X_test)

                    lr_score = r2_score(
                        y_test,
                        lr_predictions
                    )

                    # ------------------ RANDOM FOREST ------------------
                    rf = RandomForestRegressor(
                        n_estimators=50,
                        random_state=42
                    )

                    rf.fit(X_train, y_train)

                    rf_predictions = rf.predict(X_test)

                    rf_score = r2_score(
                        y_test,
                        rf_predictions
                    )

                    # ------------------ SCORES ------------------
                    scores = {
                        "Linear Regression": lr_score,
                        "Random Forest": rf_score
                    }

                    # ------------------ BEST MODEL ------------------
                    best_model = max(
                        scores,
                        key=scores.get
                    )

                    # ------------------ RESULTS ------------------
                    st.success("Training Completed Successfully 🚀")

                    st.write(
                        "Linear Regression Score:",
                        lr_score
                    )

                    st.write(
                        "Random Forest Score:",
                        rf_score
                    )

                    st.info(
                        f"🏆 Best Model: {best_model}"
                    )

                    # ------------------ MODEL COMPARISON GRAPH ------------------
                    comparison_df = pd.DataFrame({
                        "Model": list(scores.keys()),
                        "Score": list(scores.values())
                    })

                    st.subheader("Model Comparison 📊")

                    # ✅ FIXED GRAPH (NO ALTair ERROR)
                    fig_bar, ax_bar = plt.subplots()

                    ax_bar.bar(
                        comparison_df["Model"],
                        comparison_df["Score"]
                    )

                    ax_bar.set_xlabel("Models")
                    ax_bar.set_ylabel("Scores")

                    st.pyplot(fig_bar)

                    # ------------------ AI INSIGHT ------------------
                    st.subheader("AI Insight 🤖")

                    if rf_score > lr_score:
                        st.success(
                            "Random Forest performs better on this dataset."
                        )
                    else:
                        st.success(
                            "Linear Regression performs better on this dataset."
                        )

                    # ------------------ FEATURE IMPORTANCE ------------------
                    st.subheader("Feature Importance 🔥")

                    try:

                        importance_df = pd.DataFrame({
                            "Feature": X.columns,
                            "Importance": rf.feature_importances_
                        })

                        importance_df = importance_df.sort_values(
                            by="Importance",
                            ascending=False
                        )

                        st.write(
                            importance_df.head(10)
                        )

                        fig2, ax2 = plt.subplots(
                            figsize=(8, 5)
                        )

                        ax2.barh(
                            importance_df["Feature"].head(10),
                            importance_df["Importance"].head(10)
                        )

                        ax2.set_xlabel("Importance")
                        ax2.set_ylabel("Feature")

                        st.pyplot(fig2)

                    except Exception as e:
                        st.warning(
                            f"Feature Importance Error: {e}"
                        )

                except Exception as e:
                    st.error(
                        "Training failed but app did NOT crash 👍"
                    )
                    st.error(e)

    except Exception as e:
        st.error("Dataset loading error")
        st.error(e)

    # ------------------ DATA VISUALIZATION ------------------
    st.subheader("Data Visualization 📈")

    try:

        numeric_df = df.select_dtypes(
            include=['number']
        )

        if numeric_df.shape[1] >= 2:

            # ------------------ SCATTER PLOT ------------------
            fig, ax = plt.subplots()

            ax.scatter(
                numeric_df.iloc[:, 0],
                numeric_df.iloc[:, 1]
            )

            ax.set_xlabel(
                numeric_df.columns[0]
            )

            ax.set_ylabel(
                numeric_df.columns[1]
            )

            st.pyplot(fig)

            # ------------------ HEATMAP ------------------
            st.subheader("Correlation Heatmap 📊")

            corr = numeric_df.corr()

            fig3, ax3 = plt.subplots(
                figsize=(8, 6)
            )

            sns.heatmap(
                corr,
                annot=True,
                cmap="coolwarm",
                ax=ax3
            )

            st.pyplot(fig3)

        else:
            st.warning(
                "Not enough numeric columns for visualization"
            )

    except Exception as e:
        st.warning(
            f"Visualization Error: {e}"
        )

else:
    st.info("👆 Please upload a CSV file to start")
