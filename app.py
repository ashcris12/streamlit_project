#!/usr/bin/env python
# coding: utf-8

# In[251]:


import sqlite3
import pyotp
from cryptography.fernet import Fernet
import bcrypt
import streamlit as st

# Initialize Database
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

# Add User (with encrypted MFA secret)
def add_user(username, name, password, role):
    # Load encryption key
    cipher_key = st.secrets["SECRET_KEY"].encode()  # Streamlit secrets management
    cipher = Fernet(cipher_key)

    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()

    # Check the password before hashing
    print(f"Hashing password for {username}: {password}")
    
    # Hash password
    password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt(12)).decode('utf-8')
    
    # Encrypt MFA secret
    mfa_secret = pyotp.random_base32()
    encrypted_mfa = cipher.encrypt(mfa_secret.encode()).decode()

    # Print SQL query and values
    print(f"Inserting user {username} into the database with role {role}")

    try:
        # Insert user into the database
        cursor.execute("""
            INSERT INTO users (username, name, password_hash, mfa_secret, role) 
            VALUES (?, ?, ?, ?, ?)
        """, (username, name, password_hash, encrypted_mfa, role))
        conn.commit()
    except sqlite3.IntegrityError as e:
        print(f"Error inserting user: {e}")
    
    conn.close()
    
# Decrypt MFA Secret 
def decrypt_mfa_secret(username):
    # Load encryption key
    cipher_key = st.secrets["SECRET_KEY"].encode()
    cipher = Fernet(cipher_key)

    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()

    cursor.execute("SELECT mfa_secret FROM users WHERE username = ?", (username,))
    encrypted_secret = cursor.fetchone()

    if encrypted_secret:
        try:
            # Decrypt MFA secret
            decrypted_secret = cipher.decrypt(encrypted_secret[0].encode()).decode()
            print(f"Decryption successful for {username}: {decrypted_secret}")
        except Exception as e:
            print(f"Decryption failed for {username}: {e}")
    else:
        print(f"No MFA secret found for {username}")

    conn.close()

# Example usage:
init_db()  # Initialize the DB 

# Add example users based on role
add_user("exec_user", "Executive User", "password123", "executive")
add_user("finance_user", "Finance User", "securepass456", "finance")
add_user("data_user", "Data User", "datapass789", "data_science")


# In[252]:


# final submission 
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

# Ensure NOTHING from the app renders unless the user is authenticated
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

    # MFA Step (only AFTER password is verified)
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

                st.success("MFA Verified! Logging you in...")
                st.rerun()

            else:
                st.error("❌ Invalid MFA Code!")

    st.stop()  # Prevents anything below from rendering unless authenticated
    
# User is authenticated, enforce RBAC
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

# Log Out Button
if st.session_state.authenticated:
    if st.sidebar.button("🔒 Log Out"):
        st.session_state.logged_out = True  # Set a flag
        st.session_state.clear()  # Clears session variables
        st.success("✅ Logged out successfully. Redirecting...")
        time.sleep(2)
        st.rerun()  

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
    "Model Training", "Predictions & Performance", "Download Report"
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
        st.info("No file uploaded or URL entered yet.")  

with tabs[1]:  # EDA
    if st.session_state.role in ["data_science", "finance"]:
        st.header("Exploratory Data Analysis (EDA)")

        # Ensure df exists before accessing it
        if "df" not in st.session_state or st.session_state.df is None:
            st.warning("No data uploaded yet. Please upload a CSV file or URL in the 'Upload Data' tab.")
            st.stop()  # 🚀 This prevents further execution when df is missing

        df = st.session_state.df  

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

        cleaned_df = st.session_state.cleaned_df  

        # Only show the preview if cleaned_df is not empty
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

with tabs[3]:  # Upload Data
    if st.session_state.role in ["data_science", "finance"]:
    st.header("Exploratory Data Analysis (EDA)")

    # Ensure df exists before accessing it
    if "df" not in st.session_state or st.session_state.df is None:
        st.warning("No data uploaded yet. Please upload a CSV file or URL in the 'Upload Data' tab.")
        st.stop()
    
    df = st.session_state.df
    
    # Section: Data Overview
    st.header("Data Overview")
    if st.checkbox('Show Data Overview'):
        st.write(df.head())
        st.write(df.describe())
        st.write(df.info())
    
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
        options=all_features,
        default=features
    )
    
    if not selected_features:
        st.warning("Please select at least one feature.")
        selected_features = features
    
    # Define X and y after selection
    X = df[selected_features]
    y = df[target]
    
    # Feature Selection using RandomForest
    if st.checkbox('Perform Feature Selection'):
        st.write("Feature importance based on RandomForest")
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        feature_importances = pd.Series(model.feature_importances_, index=selected_features)
        st.bar_chart(feature_importances.sort_values(ascending=False))
        
        # Allow user to select important features only
        important_features = feature_importances[feature_importances > feature_importances.mean()].index.tolist()
        selected_features = st.multiselect("Refine selected features based on importance:", options=selected_features, default=important_features)
    
    # Feature Interaction
    st.header("Feature Interaction")
    interaction_type = st.radio("Select interaction type:", ("Polynomial Features", "Feature Crosses"))
    
    if interaction_type == "Polynomial Features":
        poly_degree = st.slider("Select polynomial degree:", 2, 3, 2)
        X_poly = X.copy()
        for col in selected_features:
            X_poly[f"{col}^2"] = X[col] ** 2
            if poly_degree == 3:
                X_poly[f"{col}^3"] = X[col] ** 3
        st.write("Polynomial features added:")
        st.write(X_poly.head())
    
    elif interaction_type == "Feature Crosses":
        X_cross = X.copy()
        for f1, f2 in combinations(selected_features, 2):
            X_cross[f"{f1}*{f2}"] = X[f1] * X[f2]
        st.write("Feature crosses added:")
        st.write(X_cross.head())
    
    # Split the data into training and testing sets (80/20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Section: Correlation Analysis
    st.header("Correlation Analysis")
    if st.checkbox('Show Correlation Heatmap'):
        corr = df[selected_features + [target]].corr()
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(corr, annot=True, cmap='seismic', ax=ax)
        st.pyplot(fig)
        
# In[ ]:




