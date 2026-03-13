import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import os

# --- Page Configuration ---
st.set_page_config(page_title="Data Explorer", layout="wide", page_icon="📊")

# Title
st.title("📊 Universal Exploratory Data Analysis Tool")

# --- Sidebar: Data Loading Logic ---
st.sidebar.header("Data Settings")
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

@st.cache_data
def load_data(file_input):
    """
    Safely loads data. 
    Checks for upload first, then local file, else returns None.
    """
    # 1. Check if user uploaded a file
    if file_input is not None:
        try:
            return pd.read_csv(file_input)
        except Exception as e:
            st.sidebar.error(f"Error reading uploaded file: {e}")
            return None
            
    # 2. Check if default Gapminder.csv exists locally
    if os.path.exists('Gapminder.csv'):
        try:
            return pd.read_csv('Gapminder.csv')
        except Exception as e:
            st.sidebar.error(f"Error reading Gapminder.csv: {e}")
            return None
            
    # 3. If neither, return None
    return None

# Call the loader
df = load_data(uploaded_file)

# --- CRITICAL FIX: Handle the ValueError/Missing Data ---
if df is None:
    st.warning("👈 Please upload a CSV file in the sidebar to begin your analysis.")
    st.info("The app is currently waiting for a data source.")
    # This stops the script here and prevents "ValueError" in subsequent lines
    st.stop() 

# --- Clean data (Optional: remove index columns) ---
if 'Unnamed: 0' in df.columns:
    df = df.drop(columns=['Unnamed: 0'])

# --- Sidebar: Filters (Only runs if df is NOT None) ---
st.sidebar.markdown("---")
st.sidebar.header("Global Filters")
cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

if cat_cols:
    filter_col = st.sidebar.selectbox("Filter by Category:", cat_cols)
    unique_vals = df[filter_col].unique().tolist()
    selected_vals = st.sidebar.multiselect(f"Select {filter_col}", unique_vals, default=unique_vals)
    df = df[df[filter_col].isin(selected_vals)]

# Final check if filtered data is empty
if df.empty:
    st.error("No data matches the selected filters.")
    st.stop()

# --- Main Dashboard Layout ---
st.header("🔍 Dataset Overview")
col1, col2, col3 = st.columns(3)
col1.metric("Rows", df.shape[0])
col2.metric("Columns", df.shape[1])
col3.metric("Missing Values", df.isna().sum().sum())

with st.expander("Preview Dataset"):
    st.dataframe(df.head(100), use_container_width=True)

# --- Visualization Section ---
st.header("🎨 Interactive Visualizations")
num_cols = df.select_dtypes(include=['number']).columns.tolist()

if not num_cols:
    st.warning("No numeric columns found for visualization.")
else:
    viz_col1, viz_col2 = st.columns([1, 2])
    
    with viz_col1:
        chart_type = st.radio("Chart Type", ["Scatter Plot", "Histogram", "Box Plot"])
        x_axis = st.selectbox("X-Axis", df.columns)
        y_axis = st.selectbox("Y-Axis (Numeric)", num_cols)
        color_opt = st.selectbox("Color By", [None] + cat_cols)

    with viz_col2:
        if chart_type == "Scatter Plot":
            fig = px.scatter(df, x=x_axis, y=y_axis, color=color_opt, template="plotly_white")
        elif chart_type == "Histogram":
            fig = px.histogram(df, x=x_axis, color=color
