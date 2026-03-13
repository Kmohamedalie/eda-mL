import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# Set page config
st.set_page_config(page_title="Universal EDA Tool", layout="wide")

st.title("📊 Universal Exploratory Data Analysis Tool")
st.markdown("""
Upload any CSV dataset to perform an instant exploratory analysis. 
If no file is uploaded, the **Gapminder** dataset will be used by default.
""")

# --- 1. File Upload Logic ---
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

@st.cache_data
def load_data(file):
    if file is not None:
        return pd.read_csv(file)
    else:
        # Default to Gapminder if no file is uploaded
        return pd.read_csv('Gapminder.csv')

df = load_data(uploaded_file)

# --- 2. Data Overview Section ---
st.header("🔍 Data Overview")
col1, col2, col3 = st.columns(3)
col1.metric("Rows", df.shape[0])
col2.metric("Columns", df.shape[1])
col3.metric("Missing Values", df.isna().sum().sum())

if st.checkbox("Show Raw Data"):
    st.dataframe(df.head(50))

# --- 3. Statistics Section ---
st.header("📈 Summary Statistics")
st.write(df.describe())

# --- 4. Interactive Visualizations ---
st.header("🎨 Interactive Visualizations")

viz_type = st.selectbox("Select Chart Type", 
                        ["Scatter Plot", "Line Chart", "Bar Chart", "Box Plot", "Histogram", "Correlation Heatmap"])

# Get numerical and categorical columns for selectors
numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
all_cols = df.columns.tolist()

if viz_type == "Scatter Plot":
    col_x = st.selectbox("X Axis", numeric_cols)
    col_y = st.selectbox("Y Axis", numeric_cols)
    color_col = st.selectbox("Color By (Optional)", [None] + categorical_cols)
    fig = px.scatter(df, x=col_x, y=col_y, color=color_col, title=f"{col_y} vs {col_x}")
    st.plotly_chart(fig, use_container_width=True)

elif viz_type == "Line Chart":
    col_x = st.selectbox("X Axis (Usually Time)", all_cols)
    col_y = st.selectbox("Y Axis", numeric_cols)
    group_col = st.selectbox("Group By", [None] + categorical_cols)
    fig = px.line(df, x=col_x, y=col_y, color=group_col, title=f"{col_y} over {col_x}")
    st.plotly_chart(fig, use_container_width=True)

elif viz_type == "Bar Chart":
    col_x = st.selectbox("X Axis (Category)", categorical_cols)
    col_y = st.selectbox("Y Axis (Value)", numeric_cols)
    fig = px.bar(df, x=col_x, y=col_y, title=f"{col_y} by {col_x}")
    st.plotly_chart(fig, use_container_width=True)

elif viz_type == "Box Plot":
    col_x = st.selectbox("Category", categorical_cols)
    col_y = st.selectbox("Numerical Value", numeric_cols)
    fig = px.box(df, x=col_x, y=col_y, title=f"Distribution of {col_y} by {col_x}")
    st.plotly_chart(fig, use_container_width=True)

elif viz_type == "Histogram":
    col_x = st.selectbox("Select Column", numeric_cols)
    fig = px.histogram(df, x=col_x, nbins=30, title=f"Frequency Distribution of {col_x}")
    st.plotly_chart(fig, use_container_width=True)

elif viz_type == "Correlation Heatmap":
    if len(numeric_cols) > 1:
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
        st.pyplot(fig)
    else:
        st.warning("Not enough numerical columns for a correlation heatmap.")

# --- 5. Data Filtering ---
st.sidebar.header("Filter Data")
if categorical_cols:
    filter_col = st.sidebar.selectbox("Filter by Category", categorical_cols)
    unique_vals = df[filter_col].unique().tolist()
    selected_vals = st.sidebar.multiselect(f"Select {filter_col}", unique_vals, default=unique_vals[:2])
    
    filtered_df = df[df[filter_col].isin(selected_vals)]
    st.subheader(f"Filtered Data (by {filter_col})")
    st.write(filtered_df)
