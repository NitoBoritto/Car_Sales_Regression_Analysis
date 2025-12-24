import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os

# ================== PAGE CONFIG ==================
st.set_page_config(
    page_title="🚗 Neon Car Intel",
    page_icon="🚗",
    layout="wide"
)

# ================== NEON GLOBAL STYLE & BACKGROUND ==================
st.markdown("""
<style>
    /* Full Page Background */
    .stApp {
        background-image: linear-gradient(rgba(5, 7, 10, 0.8), rgba(5, 7, 10, 0.8)), 
                          url("https://images.unsplash.com/photo-1503376780353-7e6692767b70?ixlib=rb-4.0.3&auto=format&fit=crop&w=1920&q=80");
        background-attachment: fixed;
        background-size: cover;
    }

    html, body, [class*="css"] {
        color: #e5e7eb;
    }
    
    h1, h2, h3 { 
        color: #00f2ff !important; 
        text-shadow: 0 0 10px #00f2ff, 0 0 20px #00f2ff;
    }

    /* Glassmorphism KPI Cards */
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

    .glass-nav {
        background: rgba(30, 58, 138, 0.3);
        backdrop-filter: blur(12px);
        border-radius: 12px;
        padding: 15px;
        border: 1px solid rgba(96, 165, 250, 0.2);
        margin-bottom: 20px;
    }

    [data-testid="stSidebar"] {
        background-color: rgba(5, 7, 10, 0.9) !important;
    }
</style>
""", unsafe_allow_html=True)

# ================== DATA LOAD & CLEANING ==================
@st.cache_data
def load_data():
    file_path = "car_data_cleaned.csv"
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
    else:
        url = "https://raw.githubusercontent.com/NitoBoritto/Car_Sales_Regression_Analysis/master/car_resale_prices.csv"
        df = pd.read_csv(url)

    df["resale_price"] = pd.to_numeric(df.get("resale_price", 0), errors="coerce").fillna(0)
    df["kms_driven"] = pd.to_numeric(df.get("kms_driven", 0), errors="coerce").fillna(0)
    
    # Extract brand from second word
    if "brand" not in df.columns and "full_name" in df.columns:
        df["brand"] = df["full_name"].str.split(' ').str[1]
    
    if "registered_year" not in df.columns and "year" in df.columns:
        df["registered_year"] = df["year"]
        
    return df

raw_df = load_data()

# ================== SIDEBAR FILTERS (DROPDOWN STYLE) ==================
st.sidebar.title("🧭 Navigation & Filters")

# Body Type Filter - Dropdown with expander
with st.sidebar.expander("🚗 Body Type", expanded=False):
    body_types = []
    if "body_type" in raw_df.columns:
        body_types = sorted(raw_df["body_type"].dropna().unique())
    
    select_all_body = st.checkbox("Select All Body Types", value=True, key="all_body")
    if select_all_body:
        selected_body = body_types
    else:
        selected_body = st.multiselect("Choose Body Types:", options=body_types, default=[], key="body_select")

# Insurance Filter - Dropdown with expander
with st.sidebar.expander("🛡️ Insurance Type", expanded=False):
    ins_types = []
    if "insurance" in raw_df.columns:
        ins_types = sorted(raw_df["insurance"].dropna().unique())
    
    select_all_ins = st.checkbox("Select All Insurance Types", value=True, key="all_ins")
    if select_all_ins:
        selected_ins = ins_types
    else:
        selected_ins = st.multiselect("Choose Insurance Types:", options=ins_types, default=[], key="ins_select")

# Owner Type Filter - Dropdown with expander
with st.sidebar.expander("👤 Owner Type", expanded=False):
    owner_types = []
    if "owner_type" in raw_df.columns:
        owner_types = sorted(raw_df["owner_type"].dropna().unique())
    
    select_all_owner = st.checkbox("Select All Owner Types", value=True, key="all_owner")
    if select_all_owner:
        selected_owner = owner_types
    else:
        selected_owner = st.multiselect("Choose Owner Types:", options=owner_types, default=[], key="owner_select")

# Apply Filters
df = raw_df.copy()
if selected_body and "body_type" in df.columns:
    df = df[df["body_type"].isin(selected_body)]
if selected_ins and "insurance" in df.columns:
    df = df[df["insurance"].isin(selected_ins)]
if selected_owner and "owner_type" in df.columns:
    df = df[df["owner_type"].isin(selected_owner)]

# ================== MAIN UI ==================
st.title("⚡ Car Sales Intelligence")
st.markdown("<div class='glass-nav'>🧭 🏡 Home Page</div>", unsafe_allow_html=True)

# KPI Cards
cols = st.columns(4)
avg_price_per_km = df[df["kms_driven"] > 0]["resale_price"].sum() / df[df["kms_driven"] > 0]["kms_driven"].sum() if len(df[df["kms_driven"] > 0]) > 0 else 0

metrics = [
    ("Avg Price", f"${df['resale_price'].mean():,.0f}" if not df.empty else "$0"),
    ("Max Price", f"${df['resale_price'].max():,.0f}" if not df.empty else "$0"),
    ("Avg Price/KM", f"${avg_price_per_km:.2f}"),
    ("Total Units", f"{len(df):,}")
]

for i, (label, val) in enumerate(metrics):
    with cols[i]:
        st.markdown(f"""<div class="kpi-card">
            <div style="color: #888; font-size: 0.8rem;">{label.upper()}</div>
            <div style="color: #00f2ff; font-size: 1.7rem; font-weight: bold;">{val}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("---")
l, r = st.columns(2)

with l:
    st.subheader("🌐 Market Share by Fuel")
    if "fuel_type" in df.columns and not df.empty:
        fuel_data = df.groupby("fuel_type")["resale_price"].sum().reset_index()
        
        # Bright blueish-grey color palette
        fuel_colors = ['#00f2ff', '#64b5f6', '#90caf9', '#b3e5fc', '#e1f5fe']
        
        fig = px.pie(fuel_data, names="fuel_type", values="resale_price", hole=.6,
                     color_discrete_sequence=fuel_colors)
        fig.update_traces(textfont_size=14, textfont_color='white', textposition='outside')
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="#e0e0e0")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No fuel type data available or no data matches filters")

with r:
    st.subheader("⚡ Top 5 Brands per Year")
    if "brand" in df.columns and "registered_year" in df.columns and not df.empty:
        brand_year_df = df.groupby(['registered_year', 'brand']).size().reset_index(name='sales_count')
        top_5 = brand_year_df.sort_values(['registered_year', 'sales_count'], ascending=[True, False]).groupby('registered_year').head(5)
        available_years = sorted(df['registered_year'].unique())[-5:]
        plot_df = top_5[top_5['registered_year'].isin(available_years)]

        fig2 = px.bar(plot_df, x="registered_year", y="sales_count", color="brand", 
                      barmode="group", color_discrete_sequence=px.colors.sequential.Plotly3)
        fig2.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="#e0e0e0", legend_title="Brand Name")
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.warning("No brand/year data available or no data matches filters")

st.subheader("⏱️ Price Trends Over Time")
if "registered_year" in df.columns and not df.empty:
    time_df = df.groupby("registered_year")["resale_price"].mean().reset_index()
    fig3 = px.line(time_df, x="registered_year", y="resale_price", markers=True)
    fig3.update_traces(line_color='#00f2ff')
    fig3.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="#e0e0e0")
    st.plotly_chart(fig3, use_container_width=True)
else:
    st.warning("No year data available or no data matches filters")