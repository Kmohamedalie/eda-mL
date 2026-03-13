import streamlit as st
import pandas as pd
import plotly.express as px
from textblob import TextBlob
from collections import Counter
import re

# Page configuration
st.set_page_config(page_title="Universal EDA Tool", layout="wide")

st.title("📊 Universal Exploratory Data Analysis")
st.markdown("Upload any CSV or Excel file to begin your analysis.")

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
        tab_general, tab_text = st.tabs(["📊 General EDA", "📝 Text & Sentiment Analysis"])
        
        # ==========================================
        # TAB 1: GENERAL EDA
        # ==========================================
        with tab_general:
            # 2. Data Overview
            st.header("🔍 Data Overview")
            col1, col2, col3 = st.columns(3)
            col1.metric("Rows", df.shape[0])
            col2.metric("Columns", df.shape[1])
            col3.metric("Duplicate Rows", df.duplicated().sum())

            if show_raw_data:
                st.subheader("Raw Dataframe")
                st.dataframe(df.head(100))

            # 3. Statistical Summary
            st.header("📈 Statistical Summary")
            st.write(df.describe(include='all').fillna(''))

            # 4. Interactive Visualizations
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

            # 5. Missing Values Analysis
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
            
            # Automatically detect columns that contain text/strings
            text_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()
            
            if not text_cols:
                st.info("No text columns detected in the dataset. Please upload a dataset with text data.")
            else:
                text_col = st.selectbox("Select a column to analyze:", text_cols)
                
                # Use a button to trigger analysis (prevents app from lagging on large datasets)
                if st.button("Run Text Analysis"):
                    with st.spinner("Analyzing text and calculating sentiment..."):
                        
                        # Calculate Polarity (-1 to 1) and Subjectivity (0 to 1)
                        # We use apply with a lambda function for row-by-row processing
                        df[['Polarity', 'Subjectivity']] = df[text_col].dropna().astype(str).apply(
                            lambda x: pd.Series([TextBlob(x).sentiment.polarity, TextBlob(x).sentiment.subjectivity])
                        )
                        
                        # Display Sentiment Distributions
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
                        
                        # Word Frequency Analysis
                        st.subheader("Most Frequent Words")
                        
                        # Combine all text, lowercase it, and extract words using regex
                        all_text = " ".join(df[text_col].dropna().astype(str).tolist()).lower()
                        words = re.findall(r'\b[a-z]{4,}\b', all_text) # Only grab words 4+ letters long
                        
                        # Count words and get top 20
                        word_counts = Counter(words).most_common(20)
                        freq_df = pd.DataFrame(word_counts, columns=['Word', 'Frequency'])
                        
                        fig_words = px.bar(freq_df, x='Word', y='Frequency', 
                                           title="Top 20 Most Common Words (4+ characters)",
                                           template="plotly_white")
                        st.plotly_chart(fig_words, use_container_width=True)
                        
                        # Show Data with new Sentiment Scores
                        st.subheader("Data with Sentiment Scores")
                        st.dataframe(df[[text_col, 'Polarity', 'Subjectivity']].head(100))

    except Exception as e:
        st.error(f"Error processing file: {e}")
else:
    st.info("Waiting for a dataset to be uploaded...")
