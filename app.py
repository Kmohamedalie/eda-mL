import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from textblob import TextBlob
from collections import Counter
import re
import pickle # Added for model exporting

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc

# Page configuration
st.set_page_config(page_title="Universal EDA & ML Tool", layout="wide")

st.title("📊 Universal EDA & Machine Learning App")
st.markdown("Upload any CSV or Excel file to begin your analysis and train models.")

# Initialize session state for ML results to prevent unwanted refreshes
if 'ml_run' not in st.session_state:
    st.session_state.ml_run = False

# 1. File Uploader
uploaded_file = st.file_uploader("Choose a file", type=['csv', 'xlsx'])

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
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
        # TAB 3: MODEL COMPARISON & TUNING
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
                    "Support Vector Machine (SVC)": SVC(probability=True), 
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
                        
                    # ==========================================
                    # INTERACTIVE HYPERPARAMETER TUNING
                    # ==========================================
                    with st.expander("⚙️ Advanced Model Tuning"):
                        st.markdown("Fine-tune the parameters for your selected models. (Leave as is to use standard defaults)")
                        
                        col_tune1, col_tune2 = st.columns(2)
                        
                        with col_tune1:
                            if "Random Forest" in selected_model_names:
                                st.markdown("**🌲 Random Forest**")
                                rf_n_estimators = st.slider("Number of Trees", min_value=10, max_value=500, value=100, step=10)
                                rf_max_depth = st.slider("Max Depth", min_value=2, max_value=50, value=15)
                                
                                if is_regression:
                                    available_models["Random Forest"] = RandomForestRegressor(n_estimators=rf_n_estimators, max_depth=rf_max_depth, random_state=42)
                                else:
                                    available_models["Random Forest"] = RandomForestClassifier(n_estimators=rf_n_estimators, max_depth=rf_max_depth, random_state=42)

                            if "Logistic Regression" in selected_model_names and not is_regression:
                                st.markdown("**📈 Logistic Regression**")
                                lr_c = st.select_slider("Inverse of Reg. Strength (C)", options=[0.01, 0.1, 1.0, 10.0, 100.0], value=1.0)
                                available_models["Logistic Regression"] = LogisticRegression(C=lr_c, max_iter=1000)

                        with col_tune2:
                            if "K-Nearest Neighbors" in selected_model_names:
                                st.markdown("**📍 K-Nearest Neighbors**")
                                knn_neighbors = st.slider("Number of Neighbors (K)", min_value=1, max_value=20, value=5)
                                
                                if is_regression:
                                    available_models["K-Nearest Neighbors"] = KNeighborsRegressor(n_neighbors=knn_neighbors)
                                else:
                                    available_models["K-Nearest Neighbors"] = KNeighborsClassifier(n_neighbors=knn_neighbors)
                                    
                            if "Support Vector Machine (SVC)" in selected_model_names or "Support Vector Machine (SVR)" in selected_model_names:
                                st.markdown("**🛣️ Support Vector Machine**")
                                svm_c = st.select_slider("Regularization (C)", options=[0.1, 1.0, 10.0, 100.0], value=1.0)
                                svm_kernel = st.selectbox("Kernel", ["rbf", "linear", "poly"])
                                
                                if is_regression:
                                    available_models["Support Vector Machine (SVR)"] = SVR(C=svm_c, kernel=svm_kernel)
                                else:
                                    available_models["Support Vector Machine (SVC)"] = SVC(C=svm_c, kernel=svm_kernel, probability=True)

                    # ==========================================
                    # MODEL TRAINING & SESSION STATE STORAGE
                    # ==========================================
                    if st.button("Run Model Bake-off"):
                        if not selected_model_names:
                            st.error("Please select at least one model to train.")
                        else:
                            ml_data = df[features + [target_var]].dropna()
                            
                            if len(ml_data) < 10:
                                st.error("Not enough valid data points remaining after removing missing values.")
                            else:
                                X = ml_data[features]
                                
                                if not is_regression:
                                    le = LabelEncoder()
                                    y = le.fit_transform(ml_data[target_var])
                                    is_binary = len(np.unique(y)) == 2
                                else:
                                    y = ml_data[target_var]
                                    is_binary = False
                                
                                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
                                
                                results = []
                                trained_models = {} 
                                
                                with st.spinner("Training models and calculating advanced metrics..."):
                                    for name in selected_model_names:
                                        model = available_models[name]
                                        model.fit(X_train, y_train)
                                        preds = model.predict(X_test)
                                        trained_models[name] = model
                                        
                                        if is_regression:
                                            r2 = r2_score(y_test, preds)
                                            mse = mean_squared_error(y_test, preds)
                                            mae = mean_absolute_error(y_test, preds)
                                            results.append({"Model": name, "R² Score": r2, "MSE": mse, "MAE": mae})
                                        else:
                                            acc = accuracy_score(y_test, preds)
                                            prec = precision_score(y_test, preds, average='weighted', zero_division=0)
                                            rec = recall_score(y_test, preds, average='weighted', zero_division=0)
                                            f1 = f1_score(y_test, preds, average='weighted', zero_division=0)
                                            results.append({"Model": name, "Accuracy": acc, "Precision": prec, "Recall": rec, "F1-Score": f1})
                                
                                # Save to memory
                                st.session_state.results = results
                                st.session_state.trained_models = trained_models
                                st.session_state.X_test = X_test
                                st.session_state.y_test = y_test
                                st.session_state.features = features
                                st.session_state.is_regression_run = is_regression
                                st.session_state.is_binary_run = is_binary
                                st.session_state.selected_model_names_run = selected_model_names
                                st.session_state.ml_run = True 

                    # ==========================================
                    # RENDER RESULTS FROM SESSION STATE
                    # ==========================================
                    if st.session_state.ml_run:
                        st.success("Bake-off complete!")
                        
                        st.subheader("🏆 Model Leaderboard")
                        results_df = pd.DataFrame(st.session_state.results)
                        
                        if st.session_state.is_regression_run:
                            results_df = results_df.sort_values(by="R² Score", ascending=False)
                            st.dataframe(results_df.style.highlight_max(subset=['R² Score'], color='lightgreen'))
                            
                            df_melted = results_df.melt(id_vars="Model", value_vars=["R² Score"], var_name="Metric", value_name="Score")
                            fig_comp = px.bar(df_melted, x="Model", y="Score", color="Metric", barmode="group", title="R² Score Comparison (Higher is Better)")
                            st.plotly_chart(fig_comp, use_container_width=True)
                        else:
                            results_df = results_df.sort_values(by="F1-Score", ascending=False)
                            st.dataframe(results_df.style.highlight_max(subset=['Accuracy', 'Precision', 'Recall', 'F1-Score'], color='lightgreen'))
                            
                            df_melted = results_df.melt(id_vars="Model", value_vars=["Accuracy", "Precision", "Recall", "F1-Score"], var_name="Metric", value_name="Score")
                            fig_comp = px.bar(df_melted, x="Model", y="Score", color="Metric", barmode="group", title="Comprehensive Metric Comparison (Higher is Better)")
                            st.plotly_chart(fig_comp, use_container_width=True)
                        
                        st.markdown("---")
                        st.header("🔬 Deep Dive: Model Diagnostics")
                        
                        col_diag1, col_diag2 = st.columns(2)
                        
                        with col_diag1:
                            st.subheader("Feature Importance")
                            model_to_explain = st.selectbox("Select a model to view feature importance:", st.session_state.selected_model_names_run)
                            selected_model = st.session_state.trained_models[model_to_explain]
                            
                            importances = None
                            if hasattr(selected_model, 'feature_importances_'):
                                importances = selected_model.feature_importances_
                            elif hasattr(selected_model, 'coef_'):
                                if len(selected_model.coef_.shape) > 1 and selected_model.coef_.shape[0] > 1:
                                    importances = np.mean(np.abs(selected_model.coef_), axis=0)
                                else:
                                    importances = np.abs(selected_model.coef_[0]) if len(selected_model.coef_.shape) > 1 else np.abs(selected_model.coef_)
                                    
                            if importances is not None:
                                fi_df = pd.DataFrame({'Feature': st.session_state.features, 'Importance': importances})
                                fi_df = fi_df.sort_values(by='Importance', ascending=True)
                                fig_fi = px.bar(fi_df, x='Importance', y='Feature', orientation='h', 
                                                title=f"Feature Importance ({model_to_explain})",
                                                color='Importance', color_continuous_scale='viridis')
                                st.plotly_chart(fig_fi, use_container_width=True)
                            else:
                                st.info(f"Feature importance is not natively supported for {model_to_explain} (e.g., standard KNN or SVR).")

                        with col_diag2:
                            if not st.session_state.is_regression_run and st.session_state.is_binary_run:
                                st.subheader("ROC-AUC Curve")
                                fig_roc = go.Figure()
                                fig_roc.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)

                                for name in st.session_state.selected_model_names_run:
                                    model = st.session_state.trained_models[name]
                                    if hasattr(model, "predict_proba"):
                                        y_pred_proba = model.predict_proba(st.session_state.X_test)[:, 1]
                                        fpr, tpr, _ = roc_curve(st.session_state.y_test, y_pred_proba)
                                        roc_auc = auc(fpr, tpr)
                                        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'{name} (AUC = {roc_auc:.3f})'))

                                fig_roc.update_layout(
                                    xaxis_title='False Positive Rate', 
                                    yaxis_title='True Positive Rate',
                                    legend=dict(x=0.6, y=0.1),
                                    template="plotly_white",
                                    margin=dict(l=20, r=20, t=40, b=20)
                                )
                                st.plotly_chart(fig_roc, use_container_width=True)
                            elif not st.session_state.is_regression_run and not st.session_state.is_binary_run:
                                st.info("ROC-AUC curve is currently only generated for binary (2-class) classification tasks in this app.")
                            else:
                                st.info("ROC-AUC is not applicable for Regression tasks.")

                        # ==========================================
                        # EXPORT / DOWNLOAD BEST MODEL
                        # ==========================================
                        st.markdown("---")
                        st.header("💾 Save Best Model")
                        
                        # Identify best model based on primary metric from the sorted results dataframe
                        best_model_name = results_df.iloc[0]['Model']
                        primary_metric = "R² Score" if st.session_state.is_regression_run else "F1-Score"
                        best_score = results_df.iloc[0][primary_metric]
                        
                        st.success(f"**Best Model Identified:** {best_model_name} (with {primary_metric}: {best_score:.4f})")
                        st.markdown("Download the best performing model as a `.pkl` file to use it in your own Python scripts or production environments.")
                        
                        col_export1, col_export2 = st.columns([1, 2])
                        
                        with col_export1:
                            # Default the dropdown to the best model identified above
                            try:
                                best_index = st.session_state.selected_model_names_run.index(best_model_name)
                            except ValueError:
                                best_index = 0

                            model_to_download = st.selectbox(
                                "Select a model to export (Defaults to Best Model):", 
                                st.session_state.selected_model_names_run,
                                index=best_index,
                                key="export_select"
                            )
                            
                            model_obj = st.session_state.trained_models[model_to_download]
                            model_bytes = pickle.dumps(model_obj)
                            safe_filename = f"{model_to_download.replace(' ', '_').lower()}_model.pkl"
                            
                            st.download_button(
                                label=f"⬇️ Download {model_to_download}.pkl",
                                data=model_bytes,
                                file_name=safe_filename,
                                mime="application/octet-stream"
                            )
                            
                        with col_export2:
                            st.info(f"**How to use your downloaded model locally:**\n\n"
                                    f"```python\n"
                                    f"import pickle\n"
                                    f"import pandas as pd\n\n"
                                    f"# 1. Load the model\n"
                                    f"with open('{safe_filename}', 'rb') as f:\n"
                                    f"    model = pickle.load(f)\n\n"
                                    f"# 2. Make predictions on new data\n"
                                    f"predictions = model.predict(new_data)\n"
                                    f"```")

    except Exception as e:
        st.error(f"Error processing file: {e}")
else:
    st.info("Waiting for a dataset to be uploaded...")
