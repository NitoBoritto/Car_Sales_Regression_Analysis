import streamlit as st
import pandas as pd
import plotly.express as px
import os

# ================== PAGE CONFIG ==================
st.set_page_config(page_title="Brand Deep-Dive", layout="wide")

# ================== NEON GLOBAL STYLE & BACKGROUND ==================
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
    
    /* Glassmorphism KPI Cards - Same as Home Page */
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
</style>
""", unsafe_allow_html=True)

# ================== DATA LOAD ==================
@st.cache_data
def load_data():
    file_path = "car_data_cleaned.csv"
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
    else:
        url = "https://raw.githubusercontent.com/NitoBoritto/Car_Sales_Regression_Analysis/master/car_resale_prices.csv"
        df = pd.read_csv(url)
    
    if "brand" not in df.columns and "full_name" in df.columns:
        df["brand"] = df["full_name"].str.split(' ').str[1]
    
    return df

raw_df = load_data()

# ================== SIDEBAR FILTERS (DROPDOWN STYLE) ==================
st.sidebar.title("🧭 Navigation & Filters")

# Body Type Filter - Dropdown with expander
with st.sidebar.expander("🚗 Body Type", expanded=False):
    body_types = []
    if "body_type" in raw_df.columns:
        body_types = sorted(raw_df["body_type"].dropna().unique())
    
    select_all_body = st.checkbox("Select All Body Types", value=True, key="brand_all_body")
    if select_all_body:
        selected_body = body_types
    else:
        selected_body = st.multiselect("Choose Body Types:", options=body_types, default=[], key="brand_body_select")

# Insurance Filter - Dropdown with expander
with st.sidebar.expander("🛡️ Insurance Type", expanded=False):
    ins_types = []
    if "insurance" in raw_df.columns:
        ins_types = sorted(raw_df["insurance"].dropna().unique())
    
    select_all_ins = st.checkbox("Select All Insurance Types", value=True, key="brand_all_ins")
    if select_all_ins:
        selected_ins = ins_types
    else:
        selected_ins = st.multiselect("Choose Insurance Types:", options=ins_types, default=[], key="brand_ins_select")

# Owner Type Filter - Dropdown with expander
with st.sidebar.expander("👤 Owner Type", expanded=False):
    owner_types = []
    if "owner_type" in raw_df.columns:
        owner_types = sorted(raw_df["owner_type"].dropna().unique())
    
    select_all_owner = st.checkbox("Select All Owner Types", value=True, key="brand_all_owner")
    if select_all_owner:
        selected_owner = owner_types
    else:
        selected_owner = st.multiselect("Choose Owner Types:", options=owner_types, default=[], key="brand_owner_select")

# Apply Filters
df = raw_df.copy()
if selected_body and "body_type" in df.columns:
    df = df[df["body_type"].isin(selected_body)]
if selected_ins and "insurance" in df.columns:
    df = df[df["insurance"].isin(selected_ins)]
if selected_owner and "owner_type" in df.columns:
    df = df[df["owner_type"].isin(selected_owner)]

# ================== MAIN CONTENT ==================
st.title("📊 Brand Deep-Dive Analysis")

# 1. Resale Price Distribution Histogram
st.subheader("💵 Resale Price Distribution")

price_view = st.radio(
    "Select Distribution View:",
    ["Original Prices (USD)", "BoxCox Transformed (Normalized)"],
    horizontal=True
)

if not df.empty:
    if price_view == "Original Prices (USD)" and "resale_price" in df.columns:
        fig_hist = px.histogram(
            df, x='resale_price', nbins=50,
            color_discrete_sequence=['#00f2ff'],
            labels={'resale_price': 'Resale Price (USD)'},
            title='Distribution of Car Resale Prices (USD)'
        )
        fig_hist.update_layout(
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            bargap=0.1,
            showlegend=False
        )
        fig_hist.update_traces(marker_line_width=1, marker_line_color='#7000ff')
        st.plotly_chart(fig_hist, use_container_width=True)

        # KPI Cards - Same style as Home Page
        cols = st.columns(4)
        metrics = [
            ("📊 Average Price", f"${df['resale_price'].mean():,.0f}"),
            ("📉 Median Price", f"${df['resale_price'].median():,.0f}"),
            ("🔻 Min Price", f"${df['resale_price'].min():,.0f}"),
            ("🔺 Max Price", f"${df['resale_price'].max():,.0f}")
        ]
        for i, (label, val) in enumerate(metrics):
            with cols[i]:
                st.markdown(f"""<div class="kpi-card">
                    <div style="color: #888; font-size: 0.8rem;">{label.upper()}</div>
                    <div style="color: #00f2ff; font-size: 1.7rem; font-weight: bold;">{val}</div>
                </div>""", unsafe_allow_html=True)

    elif price_view == "BoxCox Transformed (Normalized)" and "boxcox_resale_price" in df.columns:
        fig_hist = px.histogram(
            df, x='boxcox_resale_price', nbins=50,
            color_discrete_sequence=['#7000ff'],
            labels={'boxcox_resale_price': 'BoxCox-Transformed Price (Normalized)'},
            title='Distribution of Car Resale Prices (BoxCox Normalized)'
        )
        fig_hist.update_layout(
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            bargap=0.1,
            showlegend=False
        )
        fig_hist.update_traces(marker_line_width=1, marker_line_color='#00f2ff')
        st.plotly_chart(fig_hist, use_container_width=True)

        # KPI Cards - Same style as Home Page (Rounded to 2 decimals)
        cols = st.columns(4)
        metrics = [
            ("📊 Average (BoxCox)", f"{df['boxcox_resale_price'].mean():.2f}"),
            ("📉 Median (BoxCox)", f"{df['boxcox_resale_price'].median():.2f}"),
            ("🔻 Min (BoxCox)", f"{df['boxcox_resale_price'].min():.2f}"),
            ("🔺 Max (BoxCox)", f"{df['boxcox_resale_price'].max():.2f}")
        ]
        for i, (label, val) in enumerate(metrics):
            with cols[i]:
                st.markdown(f"""<div class="kpi-card">
                    <div style="color: #888; font-size: 0.8rem;">{label.upper()}</div>
                    <div style="color: #00f2ff; font-size: 1.7rem; font-weight: bold;">{val}</div>
                </div>""", unsafe_allow_html=True)

        st.info("ℹ️ **BoxCox transformation** normalizes the USD price distribution for better statistical analysis. Original prices are shown in other charts.")
    else:
        st.warning("Selected price data not available")
else:
    st.warning("No data available")

st.markdown("---")

# 2. Sum of Sales by Brand
st.subheader("💰 Total Sales Value by Brand")
if "brand" in df.columns and not df.empty:
    brand_sales = df.groupby("brand")["resale_price"].sum().sort_values(ascending=False).head(15).reset_index()
    fig_brand = px.bar(brand_sales, x="brand", y="resale_price", color="resale_price", 
                       color_continuous_scale=[[0, '#B3E5FC'], [0.5, '#42A5F5'], [1, '#0D47A1']])
    fig_brand.update_layout(template="plotly_dark", xaxis_tickangle=-45, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_brand, use_container_width=True)
else:
    st.warning("No brand data available")

col1, col2 = st.columns(2)

with col1:
    st.subheader("📅 Top 10 Resale Years")
    if "registered_year" in df.columns and not df.empty:
        year_counts = df['registered_year'].value_counts().head(10).reset_index()
        year_counts.columns = ['year', 'count']
        fig_year = px.bar(year_counts, x='year', y='count', text_auto=True)
        fig_year.update_traces(marker_color='#00f2ff')
        fig_year.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_year, use_container_width=True)
    else:
        st.warning("No year data available")

with col2:
    st.subheader("🏆 Top 10 Brands by Count")
    if "brand" in df.columns and not df.empty:
        brand_counts = df["brand"].value_counts().head(10).reset_index()
        brand_counts.columns = ['brand', 'count']
        fig_model = px.bar(brand_counts, y='brand', x='count', orientation='h', color='count', color_continuous_scale='Blues')
        fig_model.update_layout(template="plotly_dark", yaxis={'categoryorder':'total ascending'}, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_model, use_container_width=True)
    else:
        st.warning("No brand data available")