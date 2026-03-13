import streamlit as st
import pandas as pd
import plotly.express as px
from textblob import TextBlob
from collections import Counter
import re
from sklearn.model_selection import train_test_split

# Regression imports
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Classification imports
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Page configuration
st.set_page_config(page_title="Universal EDA & ML Tool", layout="wide")

st.title("📊 Universal EDA & Machine Learning App")
st.markdown("Upload any CSV or Excel file to begin your analysis and train models.")

# 1. File Uploader
uploaded_file = st.file_uploader("Choose a file", type=['csv', 'xlsx'])

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        st.success("File uploaded successfully!")
        
        st.sidebar.header("Settings")
        show_raw_data = st.sidebar.checkbox("Show Raw Data", False)
        
        tab_general, tab_text, tab_ml = st.tabs(["📊 General EDA", "📝 Text & Sentiment", "🏆 Model Comparison (ML)"])
        
        # ==========================================
        # TAB 1: GENERAL EDA
        # ==========================================
        with tab_general:
            st.header("🔍 Data Overview")
            col1, col2, col3 = st.columns(3)
            col1.metric("Rows", df.shape[0])
            col2.metric("Columns", df.shape[1])
            col3.metric("Duplicate Rows", df.duplicated().sum())

            if show_raw_data:
                st.subheader("Raw Dataframe")
                st.dataframe(df.head(100))

            st.header("📈 Statistical Summary")
            st.write(df.describe(include='all').fillna(''))

            st.header("🎨 Interactive Visuals")
            columns = df.columns.tolist()
            viz_col1, viz_col2 = st.columns([1, 2])
            
            with viz_col1:
                chart_type = st.selectbox("Select Chart Type", ["Scatter", "Bar", "Histogram", "Boxplot"])
                x_axis = st.selectbox("Select X-axis", columns)
                y_axis = st.selectbox("Select Y-axis", columns)
                color_by = st.selectbox("Color by (Optional)", [None] + columns)

            with viz_col2:
                if chart_type == "Scatter":
                    fig = px.scatter(df, x=x_axis, y=y_axis, color=color_by, template="plotly_white")
                elif chart_type == "Bar":
                    fig = px.bar(df, x=x_axis, y=y_axis, color=color_by, template="plotly_white")
                elif chart_type == "Histogram":
                    fig = px.histogram(df, x=x_axis, color=color_by, template="plotly_white")
                elif chart_type == "Boxplot":
                    fig = px.box(df, x=x_axis, y=y_axis, color=color_by, template="plotly_white")
                
                st.plotly_chart(fig, use_container_width=True)

            st.header("Missing Values")
            null_counts = df.isnull().sum()
            if null_counts.sum() > 0:
                st.warning(f"Found {null_counts.sum()} missing values.")
                st.bar_chart(null_counts[null_counts > 0])
            else:
                st.info("No missing values detected!")

        # ==========================================
        # TAB 2: TEXT & SENTIMENT ANALYSIS
        # ==========================================
        with tab_text:
            st.header("📝 Text Analysis & Sentiment")
            text_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()
            
            if not text_cols:
                st.info("No text columns detected in the dataset.")
            else:
                text_col = st.selectbox("Select a column to analyze:", text_cols)
                
                if st.button("Run Text Analysis"):
                    with st.spinner("Analyzing text and calculating sentiment..."):
                        df[['Polarity', 'Subjectivity']] = df[text_col].dropna().astype(str).apply(
                            lambda x: pd.Series([TextBlob(x).sentiment.polarity, TextBlob(x).sentiment.subjectivity])
                        )
                        
                        col_sent1, col_sent2 = st.columns(2)
                        with col_sent1:
                            fig_pol = px.histogram(df, x='Polarity', nbins=20, title="Polarity (-1 Neg to 1 Pos)")
                            st.plotly_chart(fig_pol, use_container_width=True)
                        with col_sent2:
                            fig_sub = px.histogram(df, x='Subjectivity', nbins=20, title="Subjectivity (0 Obj to 1 Sub)")
                            st.plotly_chart(fig_sub, use_container_width=True)
                        
                        st.subheader("Most Frequent Words")
                        all_text = " ".join(df[text_col].dropna().astype(str).tolist()).lower()
                        words = re.findall(r'\b[a-z]{4,}\b', all_text)
                        
                        word_counts = Counter(words).most_common(20)
                        freq_df = pd.DataFrame(word_counts, columns=['Word', 'Frequency'])
                        fig_words = px.bar(freq_df, x='Word', y='Frequency', title="Top 20 Most Common Words (4+ chars)")
                        st.plotly_chart(fig_words, use_container_width=True)

        # ==========================================
        # TAB 3: MODEL COMPARISON (UPDATED METRICS)
        # ==========================================
        with tab_ml:
            st.header("🏆 Machine Learning Bake-off")
            
            task_type = st.radio("Select ML Task Type", ["Regression (Predict a Number)", "Classification (Predict a Category)"], horizontal=True)
            is_regression = "Regression" in task_type
            
            if is_regression:
                target_options = df.select_dtypes(include=['number']).columns.tolist()
                available_models = {
                    "Linear Regression": LinearRegression(),
                    "Random Forest": RandomForestRegressor(random_state=42),
                    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
                    "Support Vector Machine (SVR)": SVR(),
                    "K-Nearest Neighbors": KNeighborsRegressor()
                }
            else:
                target_options = df.columns.tolist() 
                available_models = {
                    "Logistic Regression": LogisticRegression(max_iter=1000),
                    "Random Forest": RandomForestClassifier(random_state=42),
                    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
                    "Support Vector Machine (SVC)": SVC(),
                    "K-Nearest Neighbors": KNeighborsClassifier()
                }

            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            
            if len(numeric_cols) < 1:
                st.warning("You need at least one numeric column to act as a feature.")
            else:
                col_x, col_y = st.columns(2)
                
                with col_y:
                    target_var = st.selectbox("Select Target Variable (Y)", target_options)
                with col_x:
                    feature_options = [c for c in numeric_cols if c != target_var]
                    features = st.multiselect("Select Feature Variables (X)", feature_options, default=feature_options)
                
                if not features:
                    st.warning("Please select at least one feature variable.")
                else:
                    col_settings, col_models = st.columns(2)
                    with col_settings:
                        test_size = st.slider("Test Set Size (%)", 10, 50, 20) / 100
                    with col_models:
                        selected_model_names = st.multiselect("Select Models to Compare", list(available_models.keys()), default=list(available_models.keys())[:2])
                        
                    if st.button("Run Model Bake-off"):
                        if not selected_model_names:
                            st.error("Please select at least one model to train.")
                        else:
                            ml_data = df[features + [target_var]].dropna()
                            
                            if len(ml_data) < 10:
                                st.error("Not enough valid data points remaining after removing missing values.")
                            else:
                                X = ml_data[features]
                                y = ml_data[target_var]
                                
                                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
                                
                                results = []
                                
                                with st.spinner("Training models and calculating advanced metrics..."):
                                    for name in selected_model_names:
                                        model = available_models[name]
                                        model.fit(X_train, y_train)
                                        preds = model.predict(X_test)
                                        
                                        if is_regression:
                                            r2 = r2_score(y_test, preds)
                                            mse = mean_squared_error(y_test, preds)
                                            mae = mean_absolute_error(y_test, preds)
                                            results.append({"Model": name, "R² Score": r2, "MSE": mse, "MAE": mae})
                                        else:
                                            # Using average='weighted' safely handles both binary and multi-class target variables
                                            acc = accuracy_score(y_test, preds)
                                            prec = precision_score(y_test, preds, average='weighted', zero_division=0)
                                            rec = recall_score(y_test, preds, average='weighted', zero_division=0)
                                            f1 = f1_score(y_test, preds, average='weighted', zero_division=0)
                                            results.append({"Model": name, "Accuracy": acc, "Precision": prec, "Recall": rec, "F1-Score": f1})
                                
                                st.success("Bake-off complete!")
                                
                                # Display Results Leaderboard
                                st.subheader("🏆 Model Leaderboard")
                                results_df = pd.DataFrame(results)
                                
                                if is_regression:
                                    results_df = results_df.sort_values(by="R² Score", ascending=False)
                                    st.dataframe(results_df.style.highlight_max(subset=['R² Score'], color='lightgreen'))
                                    
                                    # Plot Comparison using a grouped bar chart
                                    df_melted = results_df.melt(id_vars="Model", value_vars=["R² Score"], var_name="Metric", value_name="Score")
                                    fig_comp = px.bar(df_melted, x="Model", y="Score", color="Metric", barmode="group", title="R² Score Comparison (Higher is Better)")
                                    st.plotly_chart(fig_comp, use_container_width=True)
                                else:
                                    results_df = results_df.sort_values(by="F1-Score", ascending=False)
                                    # Highlight the max values across the primary metrics
                                    st.dataframe(results_df.style.highlight_max(subset=['Accuracy', 'Precision', 'Recall', 'F1-Score'], color='lightgreen'))
                                    
                                    # Plot Comparison using a grouped bar chart for all 4 metrics
                                    df_melted = results_df.melt(id_vars="Model", value_vars=["Accuracy", "Precision", "Recall", "F1-Score"], var_name="Metric", value_name="Score")
                                    fig_comp = px.bar(df_melted, x="Model", y="Score", color="Metric", barmode="group", title="Comprehensive Metric Comparison (Higher is Better)")
                                    st.plotly_chart(fig_comp, use_container_width=True)

    except Exception as e:
        st.error(f"Error processing file: {e}")
else:
    st.info("Waiting for a dataset to be uploaded...")
