import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import os

# --- Page Configuration ---
st.set_page_config(page_title="Universal EDA Explorer", layout="wide", page_icon="📊")

# Custom CSS to improve UI
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0px 2px 10px rgba(0,0,0,0.05);
    }
    </style>
    """, unsafe_allow_stdio=True)

st.title("📊 Universal Exploratory Data Analysis Tool")
st.markdown("Upload any CSV to begin. If no file is uploaded, the app will attempt to load the default **Gapminder** dataset.")

# --- Sidebar: File Upload ---
st.sidebar.header("Data Settings")
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

@st.cache_data
def load_data(file):
    """Safely loads data from upload or local fallback."""
    try:
        if file is not None:
            return pd.read_csv(file)
        elif os.path.exists('Gapminder.csv'):
            return pd.read_csv('Gapminder.csv')
        else:
            return None
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

df = load_data(uploaded_file)

# --- Handling Missing Data / Initial State ---
if df is None:
    st.info("👋 **Welcome!** Please upload a CSV file in the sidebar to start your analysis.")
    st.image("https://streamlit.io/images/brand/streamlit-mark-color.png", width=100)
    st.stop()  # Prevents the rest of the app from running and showing errors

# --- Data Cleaning (Removing index columns if they exist) ---
if 'Unnamed: 0' in df.columns:
    df = df.drop(columns=['Unnamed: 0'])

# --- 1. Key Metrics Dashboard ---
st.header("🔍 Quick Summary")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Total Rows", df.shape[0])
m2.metric("Total Columns", df.shape[1])
m3.metric("Numeric Features", len(df.select_dtypes(include=['number']).columns))
m4.metric("Missing Values", df.isna().sum().sum())

# --- 2. Data Preview ---
with st.expander("👀 View Raw Dataframe"):
    st.dataframe(df, use_container_width=True)

# --- 3. Statistical Analysis ---
st.header("📈 Statistical Insights")
tab1, tab2 = st.tabs(["Summary Statistics", "Correlation Heatmap"])

with tab1:
    st.write(df.describe())

with tab2:
    numeric_df = df.select_dtypes(include=['number'])
    if not numeric_df.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.heatmap(numeric_df.corr(), annot=True, cmap="mako", fmt=".2f", ax=ax)
        st.pyplot(fig)
    else:
        st.write("No numerical columns found for correlation.")

# --- 4. Interactive Visualization Builder ---
st.header("🎨 Interactive Visual Builder")

col_a, col_b = st.columns([1, 3])

with col_a:
    viz_type = st.selectbox("Select Chart Type", 
                            ["Scatter Plot", "Bar Chart", "Line Chart", "Box Plot", "Histogram"])
    
    all_cols = df.columns.tolist()
    num_cols = df.select_dtypes(include=['number']).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    x_axis = st.selectbox("X Axis", all_cols)
    y_axis = st.selectbox("Y Axis", num_cols)
    color_by = st.selectbox("Color/Group By (Optional)", [None] + cat_cols)

with col_b:
    if viz_type == "Scatter Plot":
        fig = px.scatter(df, x=x_axis, y=y_axis, color=color_by, template="plotly_white")
    elif viz_type == "Bar Chart":
        fig = px.bar(df, x=x_axis, y=y_axis, color=color_by, template="plotly_white")
    elif viz_type == "Line Chart":
        fig = px.line(df, x=x_axis, y=y_axis, color=color_by, template="plotly_white")
    elif viz_type == "Box Plot":
        fig = px.box(df, x=x_axis, y=y_axis, color=color_by, template="plotly_white")
    else:
        fig = px.histogram(df, x=x_axis, template="plotly_white")
    
    st.plotly_chart(fig, use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.write("Created with Streamlit 🚀")
