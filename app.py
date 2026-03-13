import streamlit as st
import pandas as pd
import plotly.express as px
from textblob import TextBlob
from collections import Counter
import re
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Page configuration
st.set_page_config(page_title="Universal EDA & ML Tool", layout="wide")

st.title("📊 Universal EDA & Machine Learning App")
st.markdown("Upload any CSV or Excel file to begin your analysis and train models.")

# 1. File Uploader
uploaded_file = st.file_uploader("Choose a file", type=['csv', 'xlsx'])

if uploaded_file is not None:
    # Load data
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        st.success("File uploaded successfully!")
        
        # --- Sidebar Navigation ---
        st.sidebar.header("Settings")
        show_raw_data = st.sidebar.checkbox("Show Raw Data", False)
        
        # Create Tabs for cleaner UI
        tab_general, tab_text, tab_ml = st.tabs(["📊 General EDA", "📝 Text & Sentiment", "🤖 Machine Learning"])
        
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
                        
                        st.subheader("Sentiment Distribution")
                        col_sent1, col_sent2 = st.columns(2)
                        
                        with col_sent1:
                            fig_pol = px.histogram(df, x='Polarity', nbins=20, 
                                                   title="Polarity (-1 Negative to 1 Positive)",
                                                   color_discrete_sequence=['#2ecc71'])
                            st.plotly_chart(fig_pol, use_container_width=True)
                            
                        with col_sent2:
                            fig_sub = px.histogram(df, x='Subjectivity', nbins=20, 
                                                   title="Subjectivity (0 Objective to 1 Subjective)",
                                                   color_discrete_sequence=['#3498db'])
                            st.plotly_chart(fig_sub, use_container_width=True)
                        
                        st.subheader("Most Frequent Words")
                        all_text = " ".join(df[text_col].dropna().astype(str).tolist()).lower()
                        words = re.findall(r'\b[a-z]{4,}\b', all_text)
                        
                        word_counts = Counter(words).most_common(20)
                        freq_df = pd.DataFrame(word_counts, columns=['Word', 'Frequency'])
                        
                        fig_words = px.bar(freq_df, x='Word', y='Frequency', 
                                           title="Top 20 Most Common Words (4+ characters)",
                                           template="plotly_white")
                        st.plotly_chart(fig_words, use_container_width=True)

        # ==========================================
        # TAB 3: MACHINE LEARNING (NEW)
        # ==========================================
        with tab_ml:
            st.header("🤖 Machine Learning Model Training")
            st.markdown("Train a basic regression model directly from your dataset.")
            
            # Auto-detect numeric columns only (to prevent string conversion errors)
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            
            if len(numeric_cols) < 2:
                st.warning("You need at least two numeric columns in your dataset to train a model.")
            else:
                col_x, col_y = st.columns(2)
                
                with col_y:
                    target_var = st.selectbox("Select Target Variable (Y - What you want to predict)", numeric_cols)
                with col_x:
                    # Exclude the target variable from default feature selection
                    feature_options = [c for c in numeric_cols if c != target_var]
                    features = st.multiselect("Select Feature Variables (X - What you will use to predict)", feature_options, default=feature_options)
                
                if not features:
                    st.warning("Please select at least one feature variable.")
                else:
                    col_settings, col_model = st.columns(2)
                    with col_settings:
                        test_size = st.slider("Test Set Size (%)", 10, 50, 20) / 100
                    with col_model:
                        model_type = st.selectbox("Select Model Algorithm", ["Linear Regression", "Random Forest Regressor"])
                        
                    if st.button("Train Model"):
                        # Data Prep: drop rows with missing values in the selected columns
                        ml_data = df[features + [target_var]].dropna()
                        
                        if len(ml_data) < 10:
                            st.error("Not enough valid data points remaining after removing missing values.")
                        else:
                            X = ml_data[features]
                            y = ml_data[target_var]
                            
                            # Split Data
                            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
                            
                            # Initialize Model
                            if model_type == "Linear Regression":
                                model = LinearRegression()
                            else:
                                model = RandomForestRegressor(random_state=42)
                                
                            with st.spinner(f"Training {model_type}..."):
                                model.fit(X_train, y_train)
                                predictions = model.predict(X_test)
                                
                                # Calculate Metrics
                                mse = mean_squared_error(y_test, predictions)
                                r2 = r2_score(y_test, predictions)
                                
                                st.success(f"{model_type} trained successfully on {len(X_train)} rows!")
                                
                                st.subheader("Model Performance")
                                m_col1, m_col2 = st.columns(2)
                                m_col1.metric("R² Score (Closer to 1 is better)", round(r2, 4))
                                m_col2.metric("Mean Squared Error (MSE)", round(mse, 4))
                                
                                # Plot Actual vs Predicted
                                res_df = pd.DataFrame({"Actual": y_test, "Predicted": predictions})
                                fig_ml = px.scatter(res_df, x="Actual", y="Predicted", 
                                                    title="Actual vs Predicted Values", 
                                                    template="plotly_white")
                                
                                # Add ideal prediction trend line
                                min_val = min(res_df.min())
                                max_val = max(res_df.max())
                                fig_ml.add_shape(type="line", x0=min_val, y0=min_val, x1=max_val, y1=max_val, 
                                                 line=dict(color="red", dash="dash"))
                                
                                st.plotly_chart(fig_ml, use_container_width=True)

    except Exception as e:
        st.error(f"Error processing file: {e}")
else:
    st.info("Waiting for a dataset to be uploaded...")
