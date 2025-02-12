#!/usr/bin/env python
# coding: utf-8

# Idea 
# You might also want to allow users to filter, sort, or search within the data to make it more interactive.

# In[155]:


streamlit_code = '''
import streamlit as st
import pandas as pd
import pickle
from xgboost import XGBRegressor
import plotly.express as px
import holidays
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO
from fpdf import FPDF
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import logging
import datetime
import toml
import getpass
import bcrypt
import streamlit_authenticator as stauth

# Store user credentials (for simplicity, using plain passwords here)
users = {
    "exec_user": {
        "password": "password123",
        "name": "Executive User"
    },
    "finance_user": {
        "password": "securepass456",
        "name": "Finance User"
    },
    "data_user": {
        "password": "datapass789",
        "name": "Data User"
    }
}

# Set up the login form
st.title("User Login")

# Get username and password input
username_input = st.text_input("Username")
password_input = st.text_input("Password", type="password")

# Initialize session state for login tracking
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

# Check if login button is pressed
login_button = st.button("Login")

if login_button:
    # Check if the username exists in the users dictionary
    if username_input in users:
        # Check if the password is correct
        if users[username_input]["password"] == password_input:
            st.session_state.authenticated = True  # Set authentication state to True
            st.success(f"Welcome {users[username_input]['name']}!")
            st.rerun()
        else:
            st.error("Incorrect password!")
    else:
        st.error("Invalid credentials")
        
    st.title("User Login")

# Only show tabs if the user is authenticated
if st.session_state.authenticated:
    st.title("Box Office Revenue Prediction")

    # Define tabs
    tab_names = [
        "Upload Data", "EDA", "Data Cleaning", "Feature Engineering",
        "Model Training", "Predictions", "Performance", "Download Report"
    ]
    tabs = st.tabs(tab_names)

    with tabs[0]:  # Upload Data
        st.header("Upload Your Data")
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
            except Exception as e:
                st.error(f"An error occurred while reading the file: {e}")
                st.session_state.df = df
                st.success("File uploaded successfully.")
                st.dataframe(df.head())
                
    # Handle errors for empty files or missing data
        if df.empty:
            st.error("Please upload a valid CSV file.")
        else:
            st.success("File uploaded successfully.")

    if 'df' in st.session_state:
        df = st.session_state.df

    # Load the trained model
    try:
        with open('xgboost_model.pkl', 'rb') as model_file:
            model = pickle.load(model_file)
    except FileNotFoundError:
        st.error("Model file not found!")

    st.write("Data Preview")
    st.dataframe(df.head())  # Use st.dataframe for better readability
    
    
    with tabs[1]:  # EDA
        st.header("Exploratory Data Analysis (EDA)")
        st.write("Basic Statistics")
        st.write(df.describe())

        st.write("Correlation Heatmap")
        # Select only numeric columns for correlation heatmap
        numeric_df = df.select_dtypes(include=['number'])
        corr = numeric_df.corr()
        fig = px.imshow(corr, text_auto=True, color_continuous_scale="Viridis")
        st.plotly_chart(fig)

        # Data Exploration: Additional Features
        st.header('Data Exploration')

        # Option for users to choose the type of plot for numeric columns
        plot_type = st.radio("Choose a plot type", ['Histogram', 'Box Plot'])

        # Numeric feature selection (Budget, Revenue, etc.)
        numeric_feature = st.selectbox("Choose a numeric feature to explore", 
                                        ['Production Budget (USD)', 'Domestic Gross (USD)', 'Opening Weekend (USD)'])

        # Display Histogram or Box Plot based on user selection
        if plot_type == 'Histogram':
            st.write(f"Histogram of {numeric_feature}:")
            fig = px.histogram(df, x=numeric_feature, nbins=50, title=f"Histogram of {numeric_feature}")
            st.plotly_chart(fig)
        else:
            st.write(f"Box Plot of {numeric_feature}:")
            fig = px.box(df, y=numeric_feature, title=f"Box Plot of {numeric_feature}")
            st.plotly_chart(fig)

        # Categorical feature for Bar Chart (e.g., Genre)
        st.header("Bar Chart for Categorical Features")

        categorical_feature = st.selectbox("Choose a categorical feature to explore", 
                                    ['Genre', 'Certificates', 'Source'])  # Adjust these based on your data

        st.write(f"Bar chart of {categorical_feature}:")
        bar_fig = px.bar(df, x=categorical_feature, title=f"Bar Chart of {categorical_feature}", 
                    category_orders={categorical_feature: df[categorical_feature].value_counts().index.tolist()})
        st.plotly_chart(bar_fig)

        # Scatter Plot to visualize relationships between two numeric variables (e.g., Budget vs. Box Office Revenue)
        st.header("Scatter Plot to Visualize Relationships")

        x_feature = st.selectbox("Choose the X-axis feature", 
                            ['Production Budget (USD)', 'MetaScore', 'IMDb Rating', 'Opening Weekend (USD)'])

        y_feature = st.selectbox("Choose the Y-axis feature", 
                            ['Domestic Gross (USD)', 'Production Budget (USD)', 'MetaScore', 'IMDb Rating', 'Opening Weekend (USD)'])

        st.write(f"Scatter plot between {x_feature} and {y_feature}:")
        scatter_fig = px.scatter(df, x=x_feature, y=y_feature, title=f"Scatter Plot: {x_feature} vs {y_feature}")
        st.plotly_chart(scatter_fig)
        
    with tabs[2]:  # Data Cleaning
        st.header("Data Cleaning")
        st.write("Raw Data Preview:")
        st.dataframe(df.head())

        ### Step 1: Remove Invalid Data Points
        with st.expander("Data Cleaning: Removing Invalid Data Points"):
            df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
            df['release_year'] = df['release_date'].dt.year

            # Explanation for removing 2020 movies with $0 Domestic Gross
            st.write("Movies released in 2020 with a Domestic Gross of $0 are removed due to potential COVID-19 impacts on box office performance, leading to unreliable revenue data.")

            invalid_movies = df[(df['Domestic Gross (USD)'] == 0) & (df['release_year'] == 2020)]
            if not invalid_movies.empty:
                st.write("Titles being removed:", invalid_movies['Title'].tolist())

            df = df[~df.index.isin(invalid_movies.index)]
            df.drop(columns=['release_year'], inplace=True)

            st.success("Movies with $0 Domestic Gross from 2020 have been successfully removed.")

        ### Step 2: Handle Missing Values
        with st.expander("Handle Missing Values"):
            numeric_fill = st.radio("Numeric columns fill method", ['Median', 'Mean'])
            categorical_fill = st.radio("Categorical columns fill method", ['Mode', 'Custom'])

            numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
            if numeric_fill == 'Median':
                df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
            else:
                df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

            categorical_cols = df.select_dtypes(include=['object']).columns
            if categorical_fill == 'Mode':
                df[categorical_cols] = df[categorical_cols].fillna(df[categorical_cols].mode().iloc[0])
            else:
                custom_fill_value = st.text_input("Enter custom value for missing categorical data")
                if custom_fill_value:
                    df[categorical_cols] = df[categorical_cols].fillna(custom_fill_value)

        ### Step 3: Feature Engineering
        with st.expander("Feature Engineering"):
            if st.checkbox("Extract Release Year and Month"):
                df['Release Year'] = df['release_date'].dt.year
                df['Release Month'] = df['release_date'].dt.month

            if st.checkbox("Add Holiday Release Feature"):
                us_holidays = holidays.US()
                df['Holiday_Release'] = df['release_date'].apply(lambda x: 1 if x in us_holidays else 0)

            if st.checkbox("Add Week of Year Feature"):
                df['Week_of_Year'] = df['release_date'].dt.isocalendar().week

        ### Step 4: Encoding
        with st.expander("Encoding"):
            if st.checkbox("Enable Label Encoding for Genre and Director"):
                label_enc_cols = ['Genre', 'Director']
                for col in label_enc_cols:
                    encoder = LabelEncoder()
                    df[col] = encoder.fit_transform(df[col])

            if st.checkbox("Enable One-Hot Encoding for 'Certificates', 'Language', and 'Source'"):
                df = pd.get_dummies(df, columns=['Certificates', 'original_language', 'Source'])

        ### Step 5: Log Transformation (Optional)
        with st.expander("Log Transformation (Optional)"):
            apply_log_transform = st.checkbox("Apply Log Transform to Skewed Columns")
            if apply_log_transform:
                skewed_cols = df.select_dtypes(include=['float64', 'int64']).apply(lambda x: x.skew()).abs()
                high_skew = skewed_cols[skewed_cols > 0.75].index
                df[high_skew] = df[high_skew].apply(lambda x: np.log1p(x))

        st.subheader("Final Processed Data")
        st.dataframe(df.head())

        # Download processed data
        st.subheader("Download Processed Data")
        st.download_button("Download Processed CSV", df.to_csv(index=False), "processed_data.csv")



# Analytical Tools
st.title("Box Office Revenue Prediction - Analytical Tools")

# Section: Data Overview
st.header("Data Overview")
if st.checkbox('Show Data Overview'):
    st.write(df.head())  # Show top rows of the data
    st.write(df.describe())  # Display data statistics
    st.write(df.info())  # Display data types and missing values

# Ensure the target variable is defined
target = 'Domestic Gross (USD)'

# Dynamically generate a list of all features (excluding the target)
all_features = [col for col in df.columns if col != target]

# Set default selected features (choose key features by default)
default_features = ['Opening Weekend (USD)']

# Ensure the selected default features exist in the dataframe
features = [col for col in default_features if col in all_features]

# Section: Feature Selection
st.header("Select Features for Model")
selected_features = st.multiselect(
    "Select the features you want to include in the model:",
    options=all_features,  # Use all features except the target
    default=features  # Use only the valid default features
)

# Ensure at least one feature is selected
if not selected_features:
    st.warning("Please select at least one feature.")
    selected_features = features  # Ensure a default set of features is used

# Define X and y after selection
X = df[selected_features]
y = df[target]


# Split the data into training and testing sets (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Section: Correlation Analysis
st.header("Correlation Analysis")
if st.checkbox('Show Correlation Heatmap'):
    corr = df[selected_features + [target]].corr()  # Ensure correlation matches selected features
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr, annot=True, cmap='seismic', ax=ax)
    st.pyplot(fig)

            # Section: Model Training & Evaluation
    st.title("Train and Evaluate Box Office Revenue Model")

    # Select model type (user option)
    model_option = st.selectbox("Select a Model to Train", ["XGBoost", "Random Forest", "Decision Tree", "Linear Regression"])

    # Train and test models based on selection
    if model_option == "XGBoost":
        # Initialize XGBoost Regressor
        xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6)
        xgb_model.fit(X_train, y_train)

        # Predictions and evaluation
        y_pred_xgb = xgb_model.predict(X_test)
        mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
        r2_xgb = r2_score(y_test, y_pred_xgb)

        # Display results
        st.subheader("Model Evaluation (XGBoost)")
        st.write(f"Mean Absolute Error (MAE): {mae_xgb:.2f}")
        st.write(f"R-squared (R²): {r2_xgb:.2f}")
        st.session_state.mae_xgb = mae_xgb
        st.session_state.r2_xgb = r2_xgb

        # Visualize Actual vs Predicted (XGBoost)
        st.subheader("Actual vs Predicted (XGBoost)")
        results_df_xgb = pd.DataFrame({"Actual": y_test, "Predicted": y_pred_xgb})
        fig_xgb = px.scatter(results_df_xgb, x="Actual", y="Predicted", title="Actual vs Predicted (XGBoost)")
        st.plotly_chart(fig_xgb)

    elif model_option == "Random Forest":
        # Initialize RandomForest Regressor
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)

        # Predictions and evaluation
        y_pred_rf = rf_model.predict(X_test)
        mae_rf = mean_absolute_error(y_test, y_pred_rf)
        r2_rf = r2_score(y_test, y_pred_rf)

        # Display results
        st.subheader("Model Evaluation (Random Forest)")
        st.write(f"Mean Absolute Error (MAE): {mae_rf:.2f}")
        st.write(f"R-squared (R²): {r2_rf:.2f}")
        st.session_state.mae_rf = mae_rf
        st.session_state.r2_rf = r2_rf

        # Visualize Actual vs Predicted (Random Forest)
        st.subheader("Actual vs Predicted (Random Forest)")
        results_df_rf = pd.DataFrame({"Actual": y_test, "Predicted": y_pred_rf})
        fig_rf = px.scatter(results_df_rf, x="Actual", y="Predicted", title="Actual vs Predicted (Random Forest)")
        st.plotly_chart(fig_rf)

        # Store Feature Importances in session_state
        st.session_state.importance_rf = rf_model.feature_importances_

        # Ensure correct feature names and importance values
        if len(selected_features) == len(st.session_state.importance_rf):
            importance_df_rf = pd.DataFrame({
                'Feature': selected_features,
                'Importance': st.session_state.importance_rf
            })

            # Display Feature Importance
            st.subheader("Random Forest Feature Importance")
            st.write(importance_df_rf)

            fig_rf_imp = px.bar(importance_df_rf, x='Feature', y='Importance', title="Feature Importance (Random Forest)")
            st.plotly_chart(fig_rf_imp)
        else:
            st.error("The number of selected features and feature importances do not match!")

    elif model_option == "Decision Tree":
        # Initialize DecisionTree Regressor
        dt_model = DecisionTreeRegressor(random_state=42)
        dt_model.fit(X_train, y_train)

        # Predictions and evaluation
        y_pred_dt = dt_model.predict(X_test)
        mae_dt = mean_absolute_error(y_test, y_pred_dt)
        r2_dt = r2_score(y_test, y_pred_dt)

        # Display results
        st.subheader("Model Evaluation (Decision Tree)")
        st.write(f"Mean Absolute Error (MAE): {mae_dt:.2f}")
        st.write(f"R-squared (R²): {r2_dt:.2f}")
        st.session_state.mae_dt = mae_dt
        st.session_state.r2_dt = r2_dt

        # Visualize Actual vs Predicted (Decision Tree)
        st.subheader("Actual vs Predicted (Decision Tree)")
        results_df_dt = pd.DataFrame({"Actual": y_test, "Predicted": y_pred_dt})
        fig_dt = px.scatter(results_df_dt, x="Actual", y="Predicted", title="Actual vs Predicted (Decision Tree)")
        st.plotly_chart(fig_dt)

    elif model_option == "Linear Regression":
        # Initialize and train the Linear Regression model
        lr_model = LinearRegression()
        lr_model.fit(X_train, y_train)

        # Make predictions on the test set
        y_pred_lr = lr_model.predict(X_test)

        # Calculate model performance metrics
        mae_lr = mean_absolute_error(y_test, y_pred_lr)
        r2_lr = r2_score(y_test, y_pred_lr)

        # Store results in session_state
        st.session_state.mae_lr = mae_lr
        st.session_state.r2_lr = r2_lr

        # Display Results
        st.subheader("Model Evaluation (Linear Regression)")
        st.write(f"Mean AbsR-squared (R²): {r2_lr:.2f}")
        st.write(f"R-squared (R²): {r2_lr:.2f}")

        # Visualize Actual vs Predicted
        st.subheader("Actual vs Predicted (Linear Regression)")
        results_df_lr = pd.DataFrame({"Actual": y_test, "Predicted": y_pred_lr})
        fig_lr = px.scatter(results_df_lr, x="Actual", y="Predicted", title="Actual vs Predicted (Linear Regression)")
        st.plotly_chart(fig_lr)

        # Section: Model Evaluation Summary
    st.header("Model Comparison")

    # Ensure session state is properly initialized
    if "results" not in st.session_state:
        st.session_state.results = {}

    # Check if features have changed and clear only relevant session state values
    if "prev_features" not in st.session_state:
        st.session_state.prev_features = selected_features
    elif st.session_state.prev_features != selected_features:
        st.session_state.results.clear()  # Clear only model results, not everything
        st.session_state.prev_features = selected_features

    # Dictionary to store model results
    model_results = {}

    # Train selected models dynamically
    if model_option == "Linear Regression":
        lr_model = LinearRegression()
        lr_model.fit(X_train, y_train)
        y_pred_lr = lr_model.predict(X_test)
        model_results["Linear Regression"] = {
            "MAE": mean_absolute_error(y_test, y_pred_lr),
            "R²": r2_score(y_test, y_pred_lr),
        }

    if model_option == "Decision Tree":
        dt_model = DecisionTreeRegressor()
        dt_model.fit(X_train, y_train)
        y_pred_dt = dt_model.predict(X_test)
        model_results["Decision Tree"] = {
            "MAE": mean_absolute_error(y_test, y_pred_dt),
            "R²": r2_score(y_test, y_pred_dt),
        }

    if model_option == "Random Forest":
        rf_model = RandomForestRegressor()
        rf_model.fit(X_train, y_train)
        y_pred_rf = rf_model.predict(X_test)
        model_results["Random Forest"] = {
            "MAE": mean_absolute_error(y_test, y_pred_rf),
            "R²": r2_score(y_test, y_pred_rf),
        }

    if model_option == "XGBoost":
        xgb_model = XGBRegressor()
        xgb_model.fit(X_train, y_train)
        y_pred_xgb = xgb_model.predict(X_test)
        model_results["XGBoost"] = {
            "MAE": mean_absolute_error(y_test, y_pred_xgb),
            "R²": r2_score(y_test, y_pred_xgb),
        }

    # Update session state with the latest results
    st.session_state.results.update(model_results)

    # Display results dynamically
    if st.session_state.results:
        eval_df = pd.DataFrame([
            {"Model": model, "MAE": res["MAE"], "R²": res["R²"]}
            for model, res in st.session_state.results.items()
        ])
        st.write(eval_df)


    if 'lr_model' not in st.session_state:
        st.session_state.lr_model = LinearRegression().fit(X_train, y_train)

    if 'dt_model' not in st.session_state:
        st.session_state.dt_model = DecisionTreeRegressor().fit(X_train, y_train)

    if 'rf_model' not in st.session_state:
        st.session_state.rf_model = RandomForestRegressor().fit(X_train, y_train)

    if 'xgb_model' not in st.session_state:
        st.session_state.xgb_model = XGBRegressor().fit(X_train, y_train)

    # Now create models dictionary
    models = {
        'Linear Regression': st.session_state.lr_model,
        'Decision Tree': st.session_state.dt_model,
        'Random Forest': st.session_state.rf_model,
        'XGBoost': st.session_state.xgb_model
    }

    # Remove any models that are None
    models = {k: v for k, v in models.items() if v is not None}

    def generate_report(selected_models, metrics, X_test, y_test, y_pred_dict):
        """
        Generates a PDF report summarizing model performance.

        Parameters:
        - selected_models: List of selected model names.
        - metrics: List of evaluation metrics (e.g., ["MAE", "R²"])
        - X_test: Features used for testing.
        - y_test: Actual target values.
        - y_pred_dict: Dictionary containing model predictions.

        Returns:
        - BytesIO object containing the PDF report.
        """
        buffer = BytesIO()
        c = canvas.Canvas(buffer, pagesize=letter)
        width, height = letter

        c.setFont("Helvetica-Bold", 16)
        c.drawString(100, height - 50, "Model Performance Report")

        y_position = height - 80

        c.setFont("Helvetica", 12)
        c.drawString(100, y_position, f"Selected Models: {', '.join(selected_models)}")
        y_position -= 20

        # Add evaluation metrics
        c.setFont("Helvetica-Bold", 14)
        c.drawString(100, y_position, "Evaluation Metrics")
        y_position -= 20
        c.setFont("Helvetica", 12)

        for model_name in selected_models:
            mae = st.session_state.get(f"mae_{model_name.lower().replace(' ', '_')}", "N/A")
            r2 = st.session_state.get(f"r2_{model_name.lower().replace(' ', '_')}", "N/A")
            c.drawString(100, y_position, f"{model_name}: MAE = {mae}, R² = {r2}")
            y_position -= 20

        # Add spacing
        y_position -= 10

        # Actual vs. Predicted section
        c.setFont("Helvetica-Bold", 14)
        c.drawString(100, y_position, "Actual vs. Predicted")
        y_position -= 20

        for model_name in selected_models:
            if model_name in y_pred_dict:
                c.drawString(100, y_position, f"Results for {model_name}:")
                y_position -= 20
                c.drawString(120, y_position, f"First 5 Predictions: {y_pred_dict[model_name][:5]}")
                y_position -= 20

        # Feature Importance for Tree-Based Models
        c.setFont("Helvetica-Bold", 14)
        c.drawString(100, y_position, "Feature Importance (Tree-Based Models)")
        y_position -= 20
        for model_name in selected_models:
            if model_name in ["Random Forest", "XGBoost"]:
                c.drawString(100, y_position, f"Feature importance available for {model_name}")
                y_position -= 20

        c.save()
        buffer.seek(0)
        return buffer

    st.header("Model Evaluation Summary")

    # Allow users to select models for evaluation
    selected_models = st.multiselect("Select Models for Report", options=models.keys())

    if st.button("Generate Report"):
        if not selected_models:
            st.warning("Please select at least one model to generate the report.")
        else:
            eval_data = []
            y_pred_dict = {}

            for model_name in selected_models:
                y_pred = models[model_name].predict(X_test)
                y_pred_dict[model_name] = y_pred

                if model_name == "Linear Regression":
                    eval_data.append(["Linear Regression", st.session_state.mae_lr, st.session_state.r2_lr])
                elif model_name == "Decision Tree":
                    eval_data.append(["Decision Tree", st.session_state.mae_dt, st.session_state.r2_dt])
                elif model_name == "Random Forest":
                    eval_data.append(["Random Forest", st.session_state.mae_rf, st.session_state.r2_rf])
                elif model_name == "XGBoost":
                    eval_data.append(["XGBoost", st.session_state.mae_xgb, st.session_state.r2_xgb])

            eval_df = pd.DataFrame(eval_data, columns=["Model", "MAE", "R²"])
            st.write(eval_df)

            # Generate report and store it in session state
            report_pdf = generate_report(selected_models, ["MAE", "R²"], X_test, y_test, y_pred_dict)
            st.session_state.report = report_pdf
            st.success("Report generated! You can now download it.")

    # Button to download report
    if "report" in st.session_state and st.session_state.report:
        st.download_button("Download PDF Report", 
                           data=st.session_state.report, 
                           file_name="model_performance_report.pdf", 
                           mime="application/pdf")
    else:
        st.warning("No report generated yet. Click 'Generate Report' first.")
'''

# Write the code to the app.py file
with open('app.py', 'w') as f:
    f.write(streamlit_code)


# In[150]:


import bcrypt

# Define user roles and their passwords
passwords = {
    "exec_pass": "your_exec_password",
    "finance_pass": "your_finance_password",
    "data_scientist_pass": "your_data_password"
}

# Function to hash passwords
def hash_password(plain_text_password):
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(plain_text_password.encode(), salt)
    return hashed.decode()  # Convert bytes to string for storage

# Hash and store passwords
hashed_passwords = {role: hash_password(password) for role, password in passwords.items()}

# Print hashed passwords for storing in `secrets.toml`
for role, hashed in hashed_passwords.items():
    print(f'{role} = "{hashed}"')


# In[34]:


import toml

with open("/Users/ashleycriswell/secrets.toml", "r") as f:
    secrets = toml.load(f)

print(secrets)  # ✅ See if TOML loads properly


# In[148]:


streamlit_code='''
import streamlit as st
import pandas as pd
import pickle
import plotly.express as px
import numpy as np
import holidays
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from io import BytesIO
from fpdf import FPDF
import datetime

# User authentication (simple for now)
users = {
    "exec_user": "password123",
    "finance_user": "securepass456",
    "data_user": "datapass789"
}

st.title("User Login")

username_input = st.text_input("Username")
password_input = st.text_input("Password", type="password")
login_button = st.button("Login")

if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

if login_button:
    if username_input in users and users[username_input] == password_input:
        st.session_state.authenticated = True
        st.success(f"Welcome, {username_input}!")
        st.rerun()
    else:
        st.error("Invalid credentials!")

# Only show tabs if the user is authenticated
if st.session_state.authenticated:
    st.title("Box Office Revenue Prediction")

    # Define tabs
    tab_names = [
        "Upload Data", "EDA", "Data Cleaning", "Feature Engineering",
        "Model Training", "Predictions", "Performance", "Download Report"
    ]
    tabs = st.tabs(tab_names)

    with tabs[0]:  # Upload Data
        st.header("Upload Your Data")
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.session_state.df = df
            st.success("File uploaded successfully.")
            st.dataframe(df.head())

    if 'df' in st.session_state:
        df = st.session_state.df

        with tabs[1]:  # EDA
            st.header("Exploratory Data Analysis (EDA)")
            st.write("Basic Statistics")
            st.write(df.describe())

            st.write("Correlation Heatmap")
            numeric_df = df.select_dtypes(include=['number'])
            fig = px.imshow(numeric_df.corr(), text_auto=True, color_continuous_scale="Viridis")
            st.plotly_chart(fig)

        with tabs[2]:  # Data Cleaning
            st.header("Data Cleaning")
            st.write("Handling Missing Values")
            df = df.fillna(df.median())
            st.session_state.df = df
            st.success("Missing values handled.")

        with tabs[3]:  # Feature Engineering
            st.header("Feature Engineering")
            if st.checkbox("Extract Release Year and Month"):
                df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
                df['Release Year'] = df['release_date'].dt.year
                df['Release Month'] = df['release_date'].dt.month
                st.session_state.df = df
                st.success("Features added!")

        with tabs[4]:  # Model Training
            st.header("Model Training")
            target = "Domestic Gross (USD)"
            features = [col for col in df.columns if col != target and df[col].dtype in ['int64', 'float64']]
            X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, random_state=42)
            model = RandomForestRegressor()
            model.fit(X_train, y_train)
            st.session_state.model = model
            st.session_state.X_test = X_test
            st.session_state.y_test = y_test
            st.success("Model trained successfully!")

        with tabs[5]:  # Predictions
            st.header("Predictions")
            if 'model' in st.session_state:
                predictions = st.session_state.model.predict(st.session_state.X_test)
                st.session_state.predictions = predictions
                st.write("Predictions:", predictions[:5])

        with tabs[6]:  # Performance Evaluation
            st.header("Performance Evaluation")
            if 'predictions' in st.session_state:
                mae = mean_absolute_error(st.session_state.y_test, st.session_state.predictions)
                r2 = r2_score(st.session_state.y_test, st.session_state.predictions)
                st.session_state.mae = mae
                st.session_state.r2 = r2
                st.write(f"MAE: {mae}")
                st.write(f"R² Score: {r2}")

        with tabs[7]:  # Download Report
            st.header("Download Report")
            buffer = BytesIO()
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            pdf.cell(200, 10, txt="Model Performance Report", ln=True, align='C')
            pdf.cell(200, 10, txt=f"MAE: {st.session_state.mae}", ln=True)
            pdf.cell(200, 10, txt=f"R² Score: {st.session_state.r2}", ln=True)
            pdf.output(buffer)
            buffer.seek(0)
            st.download_button("Download Report", buffer, "report.pdf", "application/pdf")

'''

# Write the code to the app.py file
with open('app.py', 'w') as f:
    f.write(streamlit_code)


# In[165]:


# final submission 
streamlit_code = '''
import streamlit as st
import pandas as pd
import pickle
from xgboost import XGBRegressor
import plotly.express as px
import holidays
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO
from fpdf import FPDF
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import logging
import datetime
import toml
import getpass
import bcrypt
import streamlit_authenticator as stauth

# Store user credentials (for simplicity, using plain passwords here)
users = {
    "exec_user": {
        "password": "password123",
        "name": "Executive User"
    },
    "finance_user": {
        "password": "securepass456",
        "name": "Finance User"
    },
    "data_user": {
        "password": "datapass789",
        "name": "Data User"
    }
}

# Set up the login form
st.title("User Login")

# Get username and password input
username_input = st.text_input("Username")
password_input = st.text_input("Password", type="password")

# Initialize session state for login tracking
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

# Check if login button is pressed
login_button = st.button("Login")

if login_button:
    # Check if the username exists in the users dictionary
    if username_input in users:
        # Check if the password is correct
        if users[username_input]["password"] == password_input:
            st.session_state.authenticated = True  # Set authentication state to True
            st.success(f"Welcome {users[username_input]['name']}!")
            st.rerun()
        else:
            st.error("Incorrect password!")
    else:
        st.error("Invalid credentials")
        
    st.title("User Login")

# Only show tabs if the user is authenticated
if st.session_state.authenticated:
    st.title("Box Office Revenue Prediction")

    # Define tabs
    tab_names = [
        "Upload Data", "EDA", "Data Cleaning", "Feature Engineering",
        "Model Training", "Predictions", "Performance", "Download Report"
    ]
    tabs = st.tabs(tab_names)

    with tabs[0]:  # Upload Data
        st.header("Upload Your Data")
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
            except Exception as e:
                st.error(f"An error occurred while reading the file: {e}")
                st.session_state.df = df
                st.success("File uploaded successfully.")
                st.dataframe(df.head())
                
        # Handle errors for empty files or missing data
        if df.empty:
            st.error("Please upload a valid CSV file.")
        else:
            st.success("File uploaded successfully.")

        if 'df' in st.session_state:
            df = st.session_state.df

    # Load the trained model
    try:
        with open('xgboost_model.pkl', 'rb') as model_file:
            model = pickle.load(model_file)
    except FileNotFoundError:
        st.error("Model file not found!")

    st.write("Data Preview")
    st.dataframe(df.head())  # Use st.dataframe for better readability
    
    
    with tabs[1]:  # EDA
        st.header("Exploratory Data Analysis (EDA)")
        st.write("Basic Statistics")
        st.write(df.describe())

        st.write("Correlation Heatmap")
        # Select only numeric columns for correlation heatmap
        numeric_df = df.select_dtypes(include=['number'])
        corr = numeric_df.corr()
        fig = px.imshow(corr, text_auto=True, color_continuous_scale="Viridis")
        st.plotly_chart(fig)

        # Data Exploration: Additional Features
        st.header('Data Exploration')

        # Option for users to choose the type of plot for numeric columns
        plot_type = st.radio("Choose a plot type", ['Histogram', 'Box Plot'])

        # Numeric feature selection (Budget, Revenue, etc.)
        numeric_feature = st.selectbox("Choose a numeric feature to explore", 
                                        ['Production Budget (USD)', 'Domestic Gross (USD)', 'Opening Weekend (USD)'])

        # Display Histogram or Box Plot based on user selection
        if plot_type == 'Histogram':
            st.write(f"Histogram of {numeric_feature}:")
            fig = px.histogram(df, x=numeric_feature, nbins=50, title=f"Histogram of {numeric_feature}")
            st.plotly_chart(fig)
        else:
            st.write(f"Box Plot of {numeric_feature}:")
            fig = px.box(df, y=numeric_feature, title=f"Box Plot of {numeric_feature}")
            st.plotly_chart(fig)

        # Categorical feature for Bar Chart (e.g., Genre)
        st.header("Bar Chart for Categorical Features")

        categorical_feature = st.selectbox("Choose a categorical feature to explore", 
                                    ['Genre', 'Certificates', 'Source'])  # Adjust these based on your data

        st.write(f"Bar chart of {categorical_feature}:")
        bar_fig = px.bar(df, x=categorical_feature, title=f"Bar Chart of {categorical_feature}", 
                    category_orders={categorical_feature: df[categorical_feature].value_counts().index.tolist()})
        st.plotly_chart(bar_fig)

        # Scatter Plot to visualize relationships between two numeric variables (e.g., Budget vs. Box Office Revenue)
        st.header("Scatter Plot to Visualize Relationships")

        x_feature = st.selectbox("Choose the X-axis feature", 
                            ['Production Budget (USD)', 'MetaScore', 'IMDb Rating', 'Opening Weekend (USD)'])

        y_feature = st.selectbox("Choose the Y-axis feature", 
                            ['Domestic Gross (USD)', 'Production Budget (USD)', 'MetaScore', 'IMDb Rating', 'Opening Weekend (USD)'])

        st.write(f"Scatter plot between {x_feature} and {y_feature}:")
        scatter_fig = px.scatter(df, x=x_feature, y=y_feature, title=f"Scatter Plot: {x_feature} vs {y_feature}")
        st.plotly_chart(scatter_fig)
        
    with tabs[2]:  # Data Cleaning
        st.header("Data Cleaning")
        st.write("Raw Data Preview:")
        st.dataframe(df.head())

        ### Step 1: Remove Invalid Data Points
        with st.expander("Data Cleaning: Removing Invalid Data Points"):
            df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
            df['release_year'] = df['release_date'].dt.year

            # Explanation for removing 2020 movies with $0 Domestic Gross
            st.write("Movies released in 2020 with a Domestic Gross of $0 are removed due to potential COVID-19 impacts on box office performance, leading to unreliable revenue data.")

            invalid_movies = df[(df['Domestic Gross (USD)'] == 0) & (df['release_year'] == 2020)]
            if not invalid_movies.empty:
                st.write("Titles being removed:", invalid_movies['Title'].tolist())

            df = df[~df.index.isin(invalid_movies.index)]
            df.drop(columns=['release_year'], inplace=True)

            st.success("Movies with $0 Domestic Gross from 2020 have been successfully removed.")

        ### Step 2: Handle Missing Values
        with st.expander("Handle Missing Values"):
            numeric_fill = st.radio("Numeric columns fill method", ['Median', 'Mean'])
            categorical_fill = st.radio("Categorical columns fill method", ['Mode', 'Custom'])

            numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
            if numeric_fill == 'Median':
                df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
            else:
                df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

            categorical_cols = df.select_dtypes(include=['object']).columns
            if categorical_fill == 'Mode':
                df[categorical_cols] = df[categorical_cols].fillna(df[categorical_cols].mode().iloc[0])
            else:
                custom_fill_value = st.text_input("Enter custom value for missing categorical data")
                if custom_fill_value:
                    df[categorical_cols] = df[categorical_cols].fillna(custom_fill_value)

        ### Step 3: Feature Engineering
        with st.expander("Feature Engineering"):
            if st.checkbox("Extract Release Year and Month"):
                df['Release Year'] = df['release_date'].dt.year
                df['Release Month'] = df['release_date'].dt.month

            if st.checkbox("Add Holiday Release Feature"):
                us_holidays = holidays.US()
                df['Holiday_Release'] = df['release_date'].apply(lambda x: 1 if x in us_holidays else 0)

            if st.checkbox("Add Week of Year Feature"):
                df['Week_of_Year'] = df['release_date'].dt.isocalendar().week

        ### Step 4: Encoding
        with st.expander("Encoding"):
            if st.checkbox("Enable Label Encoding for Genre and Director"):
                label_enc_cols = ['Genre', 'Director']
                for col in label_enc_cols:
                    encoder = LabelEncoder()
                    df[col] = encoder.fit_transform(df[col])

            if st.checkbox("Enable One-Hot Encoding for 'Certificates', 'Language', and 'Source'"):
                df = pd.get_dummies(df, columns=['Certificates', 'original_language', 'Source'])

        ### Step 5: Log Transformation (Optional)
        with st.expander("Log Transformation (Optional)"):
            apply_log_transform = st.checkbox("Apply Log Transform to Skewed Columns")
            if apply_log_transform:
                skewed_cols = df.select_dtypes(include=['float64', 'int64']).apply(lambda x: x.skew()).abs()
                high_skew = skewed_cols[skewed_cols > 0.75].index
                df[high_skew] = df[high_skew].apply(lambda x: np.log1p(x))

        st.subheader("Final Processed Data")
        st.dataframe(df.head())

        # Download processed data
        st.subheader("Download Processed Data")
        st.download_button("Download Processed CSV", df.to_csv(index=False), "processed_data.csv")      
'''

# Write the code to the app.py file
with open('app.py', 'w') as f:
    f.write(streamlit_code)


# In[238]:


# final submission 
streamlit_code = '''
import streamlit as st
import pandas as pd
import pickle
from xgboost import XGBRegressor
import plotly.express as px
import holidays
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO
from fpdf import FPDF
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import logging
import datetime
import toml
import getpass
import bcrypt
import streamlit_authenticator as stauth
import pyotp
import time
import qrcode
from cryptography.fernet import Fernet
import sqlite3
        
# Initialize session state variables if they don't exist
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "last_activity" not in st.session_state:
    st.session_state.last_activity = time.time()
if "username" not in st.session_state:
    st.session_state.username = None
if "mfa_secret" not in st.session_state:
    st.session_state.mfa_secret = None
if "role" not in st.session_state:
    st.session_state.role = None

# Load cipher key from Streamlit Secrets
cipher_key = st.secrets["SECRET_KEY"].encode()

cipher = Fernet(cipher_key)

# Database Connection & Fetch User Details
def get_user(username):
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
    
    cursor.execute("SELECT name, password_hash, mfa_secret, role FROM users WHERE username=?", (username,))
    user = cursor.fetchone()
    
    conn.close()
    
    if user:
        return {
            "name": user[0],
            "password_hash": user[1],
            "mfa_secret": cipher.decrypt(user[2].encode()).decode() if user[2] else None,  # Decrypt MFA secret
            "role": user[3]
        }
    return None

# Check inactivity timeout (15 minutes)
def check_timeout():
    if st.session_state.authenticated:
        current_time = time.time()
        if current_time - st.session_state.last_activity > 900:  # 900 seconds = 15 minutes
            st.warning("Session timed out due to inactivity. Please log in again.")
            st.session_state.authenticated = False
            st.session_state.username = None
            st.session_state.role = None
            st.rerun()

# 🚨 **Ensure NOTHING renders unless the user is authenticated**
if not st.session_state.authenticated:

    # Render login form
    st.title("🔒 Secure User Login")

    username_input = st.text_input("Username")
    password_input = st.text_input("Password", type="password")
    login_button = st.button("Login")

    if login_button:
        user_data = get_user(username_input)  # Fetch from database

        if user_data:
            hashed_pw = user_data["password_hash"]

            if bcrypt.checkpw(password_input.encode(), hashed_pw.encode("utf-8")):
                st.session_state.username = username_input
                st.session_state.authenticated = False  # MFA required
                st.session_state.role = user_data["role"]

                # Generate an MFA secret if the user has never set one up
                if user_data["mfa_secret"] is None:
                    user_data["mfa_secret"] = pyotp.random_base32()
                
                st.session_state.mfa_secret = user_data["mfa_secret"]
                st.session_state.last_activity = time.time()

                st.success("✅ Password correct. Please scan the QR code to set up MFA.")
            else:
                st.error("❌ Incorrect password!")
        else:
            st.error("❌ Invalid credentials!")

    # MFA Step (only after password is verified)
    if st.session_state.username and not st.session_state.authenticated:
        st.subheader("🔑 Set Up or Enter Your MFA Code")
        totp = pyotp.TOTP(st.session_state.mfa_secret)

        # Generate a QR Code for first-time MFA setup
        otp_url = totp.provisioning_uri(st.session_state.username, issuer_name="Box Office Prediction App")
        qr = qrcode.make(otp_url)
        buf = BytesIO()
        qr.save(buf, format="PNG")
        st.image(buf.getvalue(), caption="📷 Scan this QR Code with Google Authenticator")

        mfa_input = st.text_input("Enter MFA Code", type="password")

        if st.button("Verify MFA"):
            if totp.verify(mfa_input):
                st.session_state.authenticated = True
                st.session_state.last_activity = time.time()

                # Store username in session to fetch user details after rerun
                st.session_state.username = username_input  

                st.success("🎉 MFA Verified! Logging you in...")
                st.rerun()

            else:
                st.error("❌ Invalid MFA Code!")

    st.stop()  # 🚨 **Prevents anything below from rendering unless authenticated**
    
# 🔓 **User is authenticated, enforce role-based access**
check_timeout()  # Ensure inactivity timeout is enforced

if st.session_state.authenticated and st.session_state.username:
    user_data = get_user(st.session_state.username)  # Re-fetch user data
    if user_data:
        st.session_state.role = user_data["role"]
    else:
        st.error("⚠️ User not found. Please log in again.")
        st.session_state.authenticated = False
        st.session_state.username = None
        st.stop()

# Ensure role exists before rendering content
if not st.session_state.role:
    st.error("🚫 Unauthorized access. Please contact the admin.")
    st.stop()

# Role-Based Access Control
if st.session_state.role == "executive":
    st.subheader("📊 Executive Dashboard")
    st.write("You can view reports generated by the data science team.")
    # Load and display reports from a database or stored files
    
elif st.session_state.role == "finance":
    st.subheader("📈 Finance Analyst Workspace")
    st.write("You can view reports and run predictive models.")
    # Allow model execution, but restrict raw data access
    
elif st.session_state.role == "data_science":
    st.subheader("🔬 Data Science Team Dashboard")
    st.write("You have full access to reports, model execution, and raw data.")
    # Show full access to all tools, raw data, and model outputs

else:
    st.error("🚫 Unauthorized access. Please contact the admin.")
    st.stop()

# Log Out Button (Place this AFTER user authentication check)
if st.session_state.authenticated:
    if st.sidebar.button("🔒 Log Out"):
        st.session_state.logged_out = True  # Set a flag
        st.session_state.clear()  # Clears session variables
        st.success("✅ Logged out successfully. Redirecting...")
        time.sleep(2)
        st.rerun()  # Now it will trigger correctly

# Main App Content After Authentication
st.title("🎬 Box Office Revenue Prediction Dashboard")
st.write("✅ You are securely logged in.")

# Update last activity time on user interaction
if st.button("Refresh Session"):
    st.session_state.last_activity = time.time()
    st.success("🔄 Session refreshed!")

# Only show tabs if the user is authenticated
if st.session_state.authenticated:
    st.title("Box Office Revenue Prediction")

# Define tabs
tab_names = [
    "Upload Data", "EDA", "Data Cleaning", "Feature Engineering",
    "Model Training", "Predictions", "Performance", "Download Report"
]
tabs = st.tabs(tab_names)

with tabs[0]:  # Upload Data
    st.header("Upload Your Data")

    # Ensure df exists in session state
    if "df" not in st.session_state:
        st.session_state.df = None

# Always show file upload options
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    url_input = st.text_input("Or enter a URL to fetch the data")

# Reset data if user removes file
    if uploaded_file is None and not url_input:
        st.session_state.df = None
        st.session_state.cleaned_df = None
        st.session_state.processed_df = None  # Clear other stored versions if needed

    # Handle File Upload
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.df = df  # Save to session state
            st.success("File uploaded successfully.")
            st.dataframe(df.head())
        except Exception as e:
            st.error(f"An error occurred while reading the file: {e}")
            st.session_state.df = None  # Ensure df is reset

    # Handle URL Input
    elif url_input:  
        try:
            df = pd.read_csv(url_input)
            st.session_state.df = df  # Save to session state
            st.success("Data loaded successfully from URL.")
            st.dataframe(df.head())
        except Exception as e:
            st.error(f"An error occurred while reading the URL: {e}")
            st.session_state.df = None

    # Retrieve df from session state
    if st.session_state.df is not None:
        df = st.session_state.df

        if df.empty:
            st.error("The uploaded dataset is empty. Please check your file or URL.")
    else:
        st.info("No file uploaded or URL entered yet.")  # ✅ Keep this message but remove global checks

    # Load the trained model
    try:
        with open('xgboost_model.pkl', 'rb') as model_file:
            model = pickle.load(model_file)
    except FileNotFoundError:
        st.error("Model file not found!")

with tabs[1]:  # EDA
    if st.session_state.role in ["data_science", "finance"]:
        st.header("Exploratory Data Analysis (EDA)")
        # (Existing EDA code)

        # Ensure df exists before accessing it
        if "df" not in st.session_state or st.session_state.df is None:
            st.warning("No data uploaded yet. Please upload a CSV file or URL in the 'Upload Data' tab.")
            st.stop()  # 🚀 This prevents further execution when df is missing

        df = st.session_state.df  # Now it's safe to use df

        # Display Basic Statistics
        st.subheader("Basic Statistics")
        st.write(df.describe())

        # Correlation Heatmap
        st.subheader("Correlation Heatmap")
        numeric_df = df.select_dtypes(include=['number'])
        if not numeric_df.empty:  # Ensure numeric columns exist
            corr = numeric_df.corr()
            fig = px.imshow(corr, text_auto=True, color_continuous_scale="Viridis")
            st.plotly_chart(fig)
        else:
            st.warning("No numeric columns available for correlation heatmap.")

        # Data Exploration: Additional Features
        st.subheader("Data Exploration")

        # Option for users to choose the type of plot for numeric columns
        plot_type = st.radio("Choose a plot type", ['Histogram', 'Box Plot'])

        # Numeric feature selection
        numeric_feature = st.selectbox("Choose a numeric feature to explore", 
                                        ['Production Budget (USD)', 'Domestic Gross (USD)', 'Opening Weekend (USD)'])

        # Display Histogram or Box Plot based on user selection
        if plot_type == 'Histogram':
            fig = px.histogram(df, x=numeric_feature, nbins=50, title=f"Histogram of {numeric_feature}")
            st.plotly_chart(fig)
        else:
            fig = px.box(df, y=numeric_feature, title=f"Box Plot of {numeric_feature}")
            st.plotly_chart(fig)

        # Categorical feature for Bar Chart
        st.subheader("Bar Chart for Categorical Features")

        categorical_feature = st.selectbox("Choose a categorical feature to explore", 
                                           ['Genre', 'Certificates', 'Source'])

        if categorical_feature in df.columns:
            bar_fig = px.bar(df, x=categorical_feature, title=f"Bar Chart of {categorical_feature}", 
                            category_orders={categorical_feature: df[categorical_feature].value_counts().index.tolist()})
            st.plotly_chart(bar_fig)
        else:
            st.warning(f"{categorical_feature} column not found in the dataset.")

        # Scatter Plot to visualize relationships between two numeric variables
        st.subheader("Scatter Plot to Visualize Relationships")

        x_feature = st.selectbox("Choose the X-axis feature", 
                                 ['Production Budget (USD)', 'MetaScore', 'IMDb Rating', 'Opening Weekend (USD)'])

        y_feature = st.selectbox("Choose the Y-axis feature", 
                                 ['Domestic Gross (USD)', 'Production Budget (USD)', 'MetaScore', 'IMDb Rating', 'Opening Weekend (USD)'])

        scatter_fig = px.scatter(df, x=x_feature, y=y_feature, title=f"Scatter Plot: {x_feature} vs {y_feature}")
        st.plotly_chart(scatter_fig)
    else:
        st.warning("🚫 You do not have permission to access EDA.")
            
with tabs[2]:  # Data Cleaning
    if st.session_state.role == "data_science":
        st.header("Data Cleaning")
        st.write("You can clean and preprocess data here.")
        
        if "df" not in st.session_state or st.session_state.df is None or st.session_state.df.empty:
            st.warning("No data uploaded yet. Please upload a CSV file or URL in the 'Upload Data' tab.")
            st.session_state.cleaned_df = None  # ✅ Reset cleaned_df
            st.stop()

        # Retrieve or initialize cleaned_df
        if "cleaned_df" not in st.session_state or st.session_state.cleaned_df is None:
            st.session_state.cleaned_df = st.session_state.df.copy()

        cleaned_df = st.session_state.cleaned_df  # ✅ Now it's guaranteed to exist

        # ✅ Only show the preview if cleaned_df is not empty
        if cleaned_df.empty:
            st.warning("⚠️ No data available for preview.")
        else:
            st.subheader("Raw Data Preview:")
            st.dataframe(cleaned_df.head())

            df = st.session_state.df.copy()  # Copy only when df exists

            # Load data into session state if not already present
            if "cleaned_df" not in st.session_state:
                st.session_state.cleaned_df = df.copy()

            # Work on a copy of the session state dataframe
            cleaned_df = st.session_state.cleaned_df

            st.subheader("Raw Data Preview:")
            st.dataframe(cleaned_df.head())

            ### Step 1: Remove Invalid Data Points
            with st.expander("Data Cleaning: Removing Invalid Data Points"):
                cleaned_df['release_date'] = pd.to_datetime(cleaned_df['release_date'], errors='coerce')
                cleaned_df['release_year'] = cleaned_df['release_date'].dt.year

                st.write("Movies released in 2020 with a Domestic Gross of $0 are removed due to potential COVID-19 impacts.")

                invalid_movies = cleaned_df[(cleaned_df['Domestic Gross (USD)'] == 0) & (cleaned_df['release_year'] == 2020)]
                if not invalid_movies.empty:
                    st.write("Titles being removed:", invalid_movies['Title'].tolist())

                cleaned_df = cleaned_df[~cleaned_df.index.isin(invalid_movies.index)]
                cleaned_df.drop(columns=['release_year'], inplace=True)

                st.session_state.cleaned_df = cleaned_df
                st.success("✅ Movies with $0 Domestic Gross from 2020 have been successfully removed.")

            ### Step 2: Handle Missing Values
            with st.expander("Handle Missing Values"):
                numeric_fill = st.radio("Numeric columns fill method", ['Median', 'Mean'])
                categorical_fill = st.radio("Categorical columns fill method", ['Mode', 'Custom'])

                numeric_cols = cleaned_df.select_dtypes(include=['float64', 'int64']).columns
                if numeric_fill == 'Median':
                    cleaned_df[numeric_cols] = cleaned_df[numeric_cols].fillna(cleaned_df[numeric_cols].median())
                else:
                    cleaned_df[numeric_cols] = cleaned_df[numeric_cols].fillna(cleaned_df[numeric_cols].mean())

                categorical_cols = cleaned_df.select_dtypes(include=['object']).columns
                if categorical_fill == 'Mode':
                    cleaned_df[categorical_cols] = cleaned_df[categorical_cols].fillna(cleaned_df[categorical_cols].mode().iloc[0])
                else:
                    custom_fill_value = st.text_input("Enter custom value for missing categorical data")
                    if custom_fill_value:
                        cleaned_df[categorical_cols] = cleaned_df[categorical_cols].fillna(custom_fill_value)

                st.session_state.cleaned_df = cleaned_df

            ### Step 3: Feature Engineering
            with st.expander("Feature Engineering"):
                if st.checkbox("Extract Release Year and Month"):
                    cleaned_df['Release Year'] = cleaned_df['release_date'].dt.year
                    cleaned_df['Release Month'] = cleaned_df['release_date'].dt.month

                if st.checkbox("Add Holiday Release Feature"):
                    us_holidays = holidays.US()
                    cleaned_df['Holiday_Release'] = cleaned_df['release_date'].apply(lambda x: 1 if x in us_holidays else 0)

                if st.checkbox("Add Week of Year Feature"):
                    cleaned_df['Week_of_Year'] = cleaned_df['release_date'].dt.isocalendar().week

                st.session_state.cleaned_df = cleaned_df

            ### Step 4: Encoding
            with st.expander("Encoding"):
                if st.checkbox("Enable Label Encoding for Genre and Director"):
                    label_enc_cols = ['Genre', 'Director']
                    for col in label_enc_cols:
                        if col in cleaned_df.columns:
                            cleaned_df[col] = cleaned_df[col].astype(str)  # Ensure values are strings
                            encoder = LabelEncoder()
                            cleaned_df[col] = encoder.fit_transform(cleaned_df[col])

                if st.checkbox("Enable One-Hot Encoding for 'Certificates', 'Language', and 'Source'"):
                    one_hot_cols = ['Certificates', 'original_language', 'Source']
                    cleaned_df = pd.get_dummies(cleaned_df, columns=[col for col in one_hot_cols if col in cleaned_df.columns])

                st.session_state.cleaned_df = cleaned_df

            ### Step 5: Log Transformation (Optional)
            with st.expander("Log Transformation (Optional)"):
                apply_log_transform = st.checkbox("Apply Log Transform to Skewed Columns")
                if apply_log_transform:
                    skewed_cols = cleaned_df.select_dtypes(include=['float64', 'int64']).apply(lambda x: x.skew()).abs()
                    high_skew = skewed_cols[skewed_cols > 0.75].index
                    cleaned_df[high_skew] = cleaned_df[high_skew].fillna(0)  # Fill NaNs before log transform
                    cleaned_df[high_skew] = cleaned_df[high_skew].apply(lambda x: np.log1p(x))  # Apply log1p(x)

                st.session_state.cleaned_df = cleaned_df

            ### Display Processed Data
            st.subheader("Final Processed Data")
            st.dataframe(cleaned_df.head())

            ### Download Processed Data
            st.subheader("Download Processed Data")
            st.download_button("Download Processed CSV", cleaned_df.to_csv(index=False), "processed_data.csv")

    else:
        st.warning("🚫 You do not have permission to access data cleaning.")
'''

# Write the code to the app.py file
with open('app.py', 'w') as f:
    f.write(streamlit_code)


# In[193]:


import sqlite3
from cryptography.fernet import Fernet
import bcrypt

# Generate an encryption key (only run once, then store it securely)
encryption_key = Fernet.generate_key()
cipher = Fernet(encryption_key)

# Initialize the database
def init_db():
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
    
    # Create users table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            name TEXT,
            password_hash TEXT,
            mfa_secret TEXT,
            role TEXT
        )
    """)
    
    conn.commit()
    conn.close()

init_db()


# In[196]:


import pyotp
def add_user(username, name, password, role):
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
    
    # Hash password
    password_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    
    # Encrypt MFA secret
    mfa_secret = pyotp.random_base32()
    encrypted_mfa = cipher.encrypt(mfa_secret.encode()).decode()
    
    # Insert user
    cursor.execute("INSERT INTO users (username, name, password_hash, mfa_secret, role) VALUES (?, ?, ?, ?, ?)", 
                   (username, name, password_hash, encrypted_mfa, role))
    
    conn.commit()
    conn.close()

# Example users
add_user("exec_user", "Executive User", "password123", "executive")
add_user("finance_user", "Finance User", "securepass456", "finance")
add_user("data_user", "Data User", "datapass789", "data_science")


# In[199]:


from cryptography.fernet import Fernet

# Generate a new key
cipher_key = Fernet.generate_key()

# Save it to a file (Do this once and keep the file secure)
with open("secret.key", "wb") as key_file:
    key_file.write(cipher_key)

print("New cipher key generated and saved.")


# In[204]:


with open("secret.key", "rb") as key_file:
    stored_key = key_file.read()

print("Cipher Key:", stored_key)


# In[205]:


import sqlite3
import pyotp
from cryptography.fernet import Fernet

# Load correct cipher key
with open("secret.key", "rb") as key_file:
    cipher_key = key_file.read()

cipher = Fernet(cipher_key)

conn = sqlite3.connect("users.db")
cursor = conn.cursor()

cursor.execute("SELECT username FROM users")
users = cursor.fetchall()

for (username,) in users:
    new_mfa_secret = pyotp.random_base32()
    encrypted_mfa = cipher.encrypt(new_mfa_secret.encode()).decode()
    
    cursor.execute("UPDATE users SET mfa_secret = ? WHERE username = ?", (encrypted_mfa, username))
    print(f"Updated MFA secret for {username}")

conn.commit()
conn.close()


# In[235]:


import os
print(os.path.exists("secret.key"))


# In[236]:


import sqlite3
from cryptography.fernet import Fernet

# Load the stored key
with open("secret.key", "rb") as key_file:
    stored_key = key_file.read()

cipher = Fernet(stored_key)

conn = sqlite3.connect("users.db")
cursor = conn.cursor()

cursor.execute("SELECT mfa_secret FROM users")
encrypted_secret = cursor.fetchone()[0]

try:
    decrypted_secret = cipher.decrypt(encrypted_secret.encode()).decode()
    print("Decryption successful:", decrypted_secret)
except Exception as e:
    print("Decryption failed:", e)


# In[ ]:




