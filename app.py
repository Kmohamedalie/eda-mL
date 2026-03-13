import streamlit as st
import pandas as pd
import plotly.express as px

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

    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.info("Waiting for a dataset to be uploaded...")
