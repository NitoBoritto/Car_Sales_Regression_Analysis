import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy.special import inv_boxcox
import os

# ================== PAGE CONFIG ==================
st.set_page_config(page_title="Price Predictor", layout="wide")

# ================== NEON GLOBAL STYLE ==================
st.markdown("""
<style>
    .stApp {
        background-image: linear-gradient(rgba(5, 7, 10, 0.8), rgba(5, 7, 10, 0.8)), 
                          url("https://images.unsplash.com/photo-1503376780353-7e6692767b70?ixlib=rb-4.0.3&auto=format&fit=crop&w=1920&q=80");
        background-attachment: fixed;
        background-size: cover;
    }
    html, body, [class*="css"] { color: #e5e7eb; }
    h1, h2, h3 { 
        color: #00f2ff !important; 
        text-shadow: 0 0 10px #00f2ff; 
    }
    [data-testid="stSidebar"] {
        background-color: rgba(5, 7, 10, 0.9) !important;
    }
    
    .kpi-card {
        background: rgba(15, 23, 42, 0.7);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(0, 242, 255, 0.3);
        border-radius: 15px;
        padding: 22px;
        text-align: center;
        transition: all 0.4s ease;
    }
    
    .kpi-card:hover {
        border: 1px solid #00f2ff;
        box-shadow: 0 0 20px rgba(0, 242, 255, 0.6);
        transform: translateY(-8px);
    }
    
    .prediction-card {
        background: linear-gradient(135deg, rgba(0, 242, 255, 0.2), rgba(112, 0, 255, 0.2));
        backdrop-filter: blur(15px);
        border: 2px solid #00f2ff;
        border-radius: 20px;
        padding: 30px;
        text-align: center;
        box-shadow: 0 0 30px rgba(0, 242, 255, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# ================== LOAD DATA ==================
@st.cache_data
def load_data():
    file_path = "car_data_cleaned.csv"
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
    else:
        st.error("Data file not found!")
        return None
    
    if "brand" not in df.columns and "full_name" in df.columns:
        df["brand"] = df["full_name"].str.split(' ').str[1]
    
    return df

@st.cache_resource
def train_model(_df):
    """Train Random Forest model on the data - matching notebook approach"""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from scipy.stats import boxcox
    
    df = _df.copy()
    
    # Encode transmission type
    le = LabelEncoder()
    df['transmission_type_encoded'] = le.fit_transform(df['transmission_type'])
    
    # Create dummy variables (matching notebook)
    df = pd.get_dummies(df, columns=['insurance', 'owner_type', 'fuel_type', 'body_type'],
                        drop_first=True, dtype=int)
    
    # Define features (matching your notebook's non_zero_features from ElasticNet)
    base_features = ['registered_year', 'engine_capacity', 'kms_driven', 'seats', 
                     'max_power_clean_bhp', 'mileage_number', 'transmission_type_encoded']
    
    # Get all dummy columns
    dummy_cols = [col for col in df.columns if any(prefix in col for prefix in 
                  ['insurance_', 'owner_type_', 'fuel_type_', 'body_type_'])]
    
    feature_names = base_features + dummy_cols
    feature_names = [f for f in feature_names if f in df.columns]
    
    # Prepare X and y
    X = df[feature_names].copy()
    y = df['boxcox_resale_price'].copy()
    
    # Handle missing values
    X = X.fillna(X.median())
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train Random Forest (matching notebook parameters)
    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_scaled, y)
    
    # Calculate the actual lambda from the original resale_price
    # We need to reverse-engineer lambda from the boxcox values
    original_prices = _df['resale_price'].values
    boxcox_prices = _df['boxcox_resale_price'].values
    
    # Estimate lambda by fitting boxcox on original prices
    positive_prices = original_prices[original_prices > 0]
    _, fitted_lambda = boxcox(positive_prices)
    
    return rf, scaler, feature_names, fitted_lambda, df

# Load data and model
raw_df = load_data()

if raw_df is not None:
    rf_model, scaler, feature_names, lambda_val, processed_df = train_model(raw_df)
    
    # ================== MAIN UI ==================
    st.title("🤖 AI Price Predictor")
    st.markdown("### Powered by Random Forest Regression")
    st.caption(f"Model trained with BoxCox λ = {lambda_val:.6f}")
    
    st.markdown("---")
    
    # Create input form
    st.subheader("📝 Enter Car Details")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        registered_year = st.slider("📅 Registration Year", 
                                     min_value=2004, max_value=2024, value=2018)
        
        engine_capacity = st.number_input("🔧 Engine Capacity (cc)", 
                                           min_value=500, max_value=5000, value=1500, step=100)
        
        kms_driven = st.number_input("🛣️ Kilometers Driven", 
                                      min_value=0, max_value=200000, value=50000, step=5000)
    
    with col2:
        seats = st.selectbox("💺 Number of Seats", options=[4, 5, 6, 7, 8, 9], index=1)
        
        max_power = st.number_input("⚡ Max Power (BHP)", 
                                     min_value=50, max_value=200, value=100, step=5)
        
        mileage = st.number_input("⛽ Mileage (km/l)", 
                                   min_value=10.0, max_value=30.0, value=18.0, step=0.5)
    
    with col3:
        transmission = st.selectbox("⚙️ Transmission", options=["Manual", "Automatic"])
        
        fuel_type = st.selectbox("🔥 Fuel Type", 
                                  options=["Petrol", "Diesel", "CNG", "LPG", "Electric"])
        
        body_type = st.selectbox("🚗 Body Type", 
                                  options=["Hatchback", "Sedan", "SUV", "MUV", "Minivans", "Pickup", "Coupe"])
    
    col4, col5 = st.columns(2)
    with col4:
        owner_type = st.selectbox("👤 Owner Type", 
                                   options=["First Owner", "Second Owner", "Third Owner", "Fourth Owner", "Fifth Owner"])
    with col5:
        insurance = st.selectbox("🛡️ Insurance", 
                                  options=["Comprehensive", "Third Party", "Zero Dep", "No Insurance"])
    
    # Predict button
    st.markdown("---")
    
    if st.button("🔮 Predict Price", use_container_width=True):
        # Build input dictionary matching the feature names
        input_data = {}
        
        # Base numerical features
        input_data['registered_year'] = float(registered_year)
        input_data['engine_capacity'] = float(engine_capacity)
        input_data['kms_driven'] = float(kms_driven)
        input_data['seats'] = float(seats)
        input_data['max_power_clean_bhp'] = float(max_power)
        input_data['mileage_number'] = float(mileage)
        input_data['transmission_type_encoded'] = 0 if transmission == "Manual" else 1
        
        # Insurance dummies (drop_first=True means Comprehensive is baseline)
        input_data['insurance_No Insurance'] = 1 if insurance == "No Insurance" else 0
        input_data['insurance_Third Party'] = 1 if insurance == "Third Party" else 0
        input_data['insurance_Zero Dep'] = 1 if insurance == "Zero Dep" else 0
        
        # Owner type dummies (drop_first=True means Fifth Owner is baseline)
        input_data['owner_type_First Owner'] = 1 if owner_type == "First Owner" else 0
        input_data['owner_type_Fourth Owner'] = 1 if owner_type == "Fourth Owner" else 0
        input_data['owner_type_Second Owner'] = 1 if owner_type == "Second Owner" else 0
        input_data['owner_type_Third Owner'] = 1 if owner_type == "Third Owner" else 0
        
        # Fuel type dummies (drop_first=True means CNG is baseline)
        input_data['fuel_type_Diesel'] = 1 if fuel_type == "Diesel" else 0
        input_data['fuel_type_Electric'] = 1 if fuel_type == "Electric" else 0
        input_data['fuel_type_LPG'] = 1 if fuel_type == "LPG" else 0
        input_data['fuel_type_Petrol'] = 1 if fuel_type == "Petrol" else 0
        
        # Body type dummies (drop_first=True means Coupe is baseline)
        input_data['body_type_Hatchback'] = 1 if body_type == "Hatchback" else 0
        input_data['body_type_MUV'] = 1 if body_type == "MUV" else 0
        input_data['body_type_Minivans'] = 1 if body_type == "Minivans" else 0
        input_data['body_type_Pickup'] = 1 if body_type == "Pickup" else 0
        input_data['body_type_SUV'] = 1 if body_type == "SUV" else 0
        input_data['body_type_Sedan'] = 1 if body_type == "Sedan" else 0
        
        # Create DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Ensure all required features are present with correct order
        for col in feature_names:
            if col not in input_df.columns:
                input_df[col] = 0
        
        input_df = input_df[feature_names]
        
        # Scale and predict
        input_scaled = scaler.transform(input_df)
        prediction_boxcox = rf_model.predict(input_scaled)[0]
        
        # Inverse BoxCox transformation to get actual price
        # BoxCox inverse: if lambda != 0: y = (y_transformed * lambda + 1)^(1/lambda)
        if lambda_val != 0:
            predicted_price = inv_boxcox(prediction_boxcox, lambda_val)
        else:
            predicted_price = np.exp(prediction_boxcox)
        
        # Display prediction
        st.markdown("---")
        
        col_left, col_center, col_right = st.columns([1, 2, 1])
        
        with col_center:
            st.markdown(f"""
            <div class="prediction-card">
                <div style="color: #888; font-size: 1rem; margin-bottom: 10px;">PREDICTED RESALE PRICE</div>
                <div style="color: #00f2ff; font-size: 3rem; font-weight: bold; text-shadow: 0 0 20px #00f2ff;">
                    ${predicted_price:,.0f}
                </div>
                <div style="color: #7000ff; font-size: 0.9rem; margin-top: 10px;">
                    BoxCox Value: {prediction_boxcox:.4f} | λ: {lambda_val:.4f}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Show price range comparison
        st.markdown("---")
        st.subheader("📊 Price Comparison with Similar Cars")
        
        # Filter similar cars
        similar_cars = raw_df[
            (raw_df['registered_year'] >= registered_year - 2) &
            (raw_df['registered_year'] <= registered_year + 2) &
            (raw_df['body_type'] == body_type) &
            (raw_df['fuel_type'] == fuel_type)
        ]
        
        if len(similar_cars) >= 5:
            col1, col2 = st.columns(2)
            
            with col1:
                # Price distribution of similar cars
                fig = px.histogram(similar_cars, x='resale_price', nbins=30,
                                   color_discrete_sequence=['#00f2ff'],
                                   title=f'Price Distribution: {body_type}s ({fuel_type}, {registered_year-2}-{registered_year+2})')
                
                # Add predicted price line
                fig.add_vline(x=predicted_price, line_dash="dash", line_color="#ff00ff",
                              annotation_text=f"Prediction: ${predicted_price:,.0f}")
                
                fig.update_layout(template="plotly_dark", 
                                  paper_bgcolor='rgba(0,0,0,0)', 
                                  plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # KPI comparison
                avg_similar = similar_cars['resale_price'].mean()
                min_similar = similar_cars['resale_price'].min()
                max_similar = similar_cars['resale_price'].max()
                median_similar = similar_cars['resale_price'].median()
                
                comparison_metrics = [
                    ("🔮 Your Prediction", f"${predicted_price:,.0f}", "#00f2ff"),
                    ("📊 Market Average", f"${avg_similar:,.0f}", "#64b5f6"),
                    ("📉 Market Median", f"${median_similar:,.0f}", "#90caf9"),
                    ("⬇️ Market Min", f"${min_similar:,.0f}", "#b3e5fc"),
                    ("⬆️ Market Max", f"${max_similar:,.0f}", "#e1f5fe")
                ]
                
                for label, val, color in comparison_metrics:
                    st.markdown(f"""<div class="kpi-card" style="margin-bottom: 10px;">
                        <div style="color: #888; font-size: 0.8rem;">{label.upper()}</div>
                        <div style="color: {color}; font-size: 1.5rem; font-weight: bold;">{val}</div>
                    </div>""", unsafe_allow_html=True)
                
                # Price position indicator
                if predicted_price < avg_similar:
                    diff_pct = ((avg_similar - predicted_price) / avg_similar) * 100
                    st.success(f"✅ Your prediction is {diff_pct:.1f}% below market average - Good deal!")
                else:
                    diff_pct = ((predicted_price - avg_similar) / avg_similar) * 100
                    st.warning(f"⚠️ Your prediction is {diff_pct:.1f}% above market average")
        else:
            st.info(f"ℹ️ Only {len(similar_cars)} similar cars found. Showing overall market comparison.")
            
            avg_all = raw_df['resale_price'].mean()
            st.markdown(f"""<div class="kpi-card">
                <div style="color: #888; font-size: 0.8rem;">OVERALL MARKET AVERAGE</div>
                <div style="color: #00f2ff; font-size: 1.5rem; font-weight: bold;">${avg_all:,.0f}</div>
            </div>""", unsafe_allow_html=True)
    
    # ================== FEATURE IMPORTANCE ==================
    st.markdown("---")
    st.subheader("🎯 What Influences Price Most?")
    
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': rf_model.feature_importances_
    }).sort_values('Importance', ascending=True).tail(15)
    
    # Clean up feature names for display
    importance_df['Feature_Clean'] = importance_df['Feature'].str.replace('_', ' ').str.title()
    
    fig_imp = px.bar(importance_df, x='Importance', y='Feature_Clean', orientation='h',
                      color='Importance', 
                      color_continuous_scale=[[0, '#B3E5FC'], [0.5, '#42A5F5'], [1, '#0D47A1']])
    fig_imp.update_layout(
        template="plotly_dark", 
        paper_bgcolor='rgba(0,0,0,0)', 
        plot_bgcolor='rgba(0,0,0,0)',
        title="Top 15 Features Influencing Price",
        yaxis_title="",
        xaxis_title="Importance Score"
    )
    st.plotly_chart(fig_imp, use_container_width=True)
    
    # Model info
    with st.expander("ℹ️ About the Model"):
        st.markdown(f"""
        **Model Type:** Random Forest Regressor  
        **Trees:** 200  
        **Features Used:** {len(feature_names)}  
        **Training Samples:** {len(raw_df):,}  
        **BoxCox Lambda:** {lambda_val:.6f}  
        
        The model predicts the BoxCox-transformed resale price, which is then 
        converted back to USD using the inverse BoxCox transformation.
        """)

else:
    st.error("❌ Failed to load data. Please check the data file exists.")