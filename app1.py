# =====================================
# Customer Segmentation RFM Analysis - Complete Streamlit App
# =====================================

# import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.stats import skew
import streamlit as st
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# =====================================
# Page Configuration
# =====================================

st.set_page_config(
    page_title="Customer Segmentation - Store X",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling with improved contrast (dark theme)
st.markdown("""
<style>
    body {
        background-color: #121212;
        color: #E0E0E0;
    }
    .main-header {
        font-size: 2.5rem;
        color: #FFFFFF;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        color: #00BFFF;
        border-bottom: 2px solid #00BFFF;
        padding-bottom: 0.5rem;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #252525;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
        margin: 0.5rem;
        color: #E0E0E0;
    }
    .insight-box {
        background-color: #1E1E1E;
        padding: 1rem;
        border-left: 4px solid #00BFFF;
        margin: 1rem 0;
        color: #E0E0E0;
    }
    .stDataFrame {
        color: #E0E0E0;
        background-color: #252525;
    }
</style>
""", unsafe_allow_html=True)

# =====================================
# Utility Functions
# =====================================

@st.cache_data
def fn_rename(df: pd.DataFrame, ori_cols: list, new_cols: list) -> pd.DataFrame:
    return df.rename(columns=dict(zip(ori_cols, new_cols)))

@st.cache_data
def fn_null_nan_count(df: pd.DataFrame) -> pd.DataFrame:
    results = []
    total_rows = len(df)
    for col in df.columns:
        null_cnt = df[col].isnull().sum()
        null_pct = round((null_cnt / total_rows) * 100, 1) if total_rows > 0 else 0.0
        results.append({
            "Column": col, 
            "Null Count": null_cnt, 
            "Null %": null_pct,
            "Data Type": str(df[col].dtype)
        })
    return pd.DataFrame(results)

@st.cache_data
def fn_date_format(df: pd.DataFrame, cols: list, date_pattern: str = "%d-%m-%Y") -> pd.DataFrame:
    df2 = df.copy()
    for col in cols:
        df2[col] = pd.to_datetime(df2[col], format=date_pattern, errors="coerce")
    return df2

def create_rfm_map_visual():
    """Create RFM segmentation map visualization"""
    fig, ax = plt.subplots(figsize=(8, 6))
    segments = [
        (1, 1, 1, 3, "Lost", "#FFB266"),
        (1, 4, 2, 1, "Can't\nLose\nThem", "#FF6666"),
        (2, 1, 1, 3, "Needs\nAttention", "#6666FF"),
        (3, 3, 2, 2, "Loyal", "#66E0E0"),
        (3, 1, 2, 2, "Potential", "#B0C4DE"),
        (4, 1, 1, 1, "New", "#3399FF"),
        (4, 4, 1, 1, "Champions", "#008080"),
    ]
    
    for x, y, w, h, label, color in segments:
        rect = patches.Rectangle((x - 0.5, y - 0.5), w, h, 
                               linewidth=2, edgecolor="black", facecolor=color, alpha=0.8)
        ax.add_patch(rect)
        ax.text(x - 0.5 + w / 2, y - 0.5 + h / 2, label, 
               ha="center", va="center", fontsize=10, weight="bold")
    
    ax.set_xlim(0.5, 4.5)
    ax.set_ylim(0.5, 4.5)
    ax.set_xticks([1, 2, 3, 4])
    ax.set_yticks([1, 2, 3, 4])
    ax.set_xlabel("R (Recency Score)", fontsize=12, fontweight='bold')
    ax.set_ylabel("(F + M) / 2 (Frequency + Monetary Average)", fontsize=12, fontweight='bold')
    ax.set_title("RFM Customer Segmentation Map", fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)
    return fig

@st.cache_data
def clean_and_process_data(df_products, df_transactions, report_from_dt, report_to_dt):
    """Clean and preprocess the data"""
    # Rename columns
    products_cols_ori = list(df_products.columns)
    transactions_cols_ori = list(df_transactions.columns)
    
    df_products = fn_rename(df_products, products_cols_ori, 
                           ["product_id", "product_name", "product_price", "product_category"])
    df_transactions = fn_rename(df_transactions, transactions_cols_ori, 
                               ["customer_id", "Date", "product_id", "quantity"])
    
    # Convert date format
    df_transactions = fn_date_format(df_transactions, cols=["Date"], date_pattern="%d-%m-%Y")
    
    # Merge data
    df_trans_pre = df_transactions.merge(df_products, on="product_id", how="left")
    df_trans_pre["amount"] = df_trans_pre["quantity"] * df_trans_pre["product_price"]
    
    # Find invalid records
    invalid_amount = len(df_trans_pre[df_trans_pre["amount"] <= 0])
    invalid_date = len(df_trans_pre[(df_trans_pre["Date"] < report_from_dt) | (df_trans_pre["Date"] > report_to_dt)])
    invalid_name = len(df_trans_pre[df_trans_pre["product_name"].isna()])
    
    # Remove invalid records and filter by date range
    df_clean = df_trans_pre[
        (df_trans_pre["amount"] > 0) & 
        (df_trans_pre["Date"] >= report_from_dt) &
        (df_trans_pre["Date"] <= report_to_dt) & 
        (df_trans_pre["product_name"].notna())
    ].copy()
    
    return df_clean, invalid_amount, invalid_date, invalid_name

@st.cache_data
def create_rfm_table(df_trans_clean, report_to_dt):
    """Create RFM table from transaction data"""
    max_date = df_trans_clean["Date"].max().date()
    
    def recency_calc(x):
        return (report_to_dt.date() - x.max().date()).days
    
    def frequency_calc(x):
        return len(x.unique())
    
    def monetary_calc(x):
        return round(x.sum(), 2)
    
    df_rfm = df_trans_clean.groupby("customer_id").agg(
        recency=("Date", recency_calc),
        frequency=("Date", frequency_calc),
        monetary=("amount", monetary_calc)
    ).reset_index()
    
    return df_rfm, max_date

def rfm_quartile_scoring(df_rfm):
    """Calculate RFM quartile scores"""
    r_labels = range(4, 0, -1)
    f_labels = range(1, 5)
    m_labels = range(1, 5)
    
    df_scored = df_rfm.copy()
    df_scored['r'] = pd.qcut(df_rfm["recency"].rank(method="first"), 4, labels=r_labels).astype(int)
    df_scored['f'] = pd.qcut(df_rfm["frequency"].rank(method="first"), 4, labels=f_labels).astype(int)
    df_scored['m'] = pd.qcut(df_rfm["monetary"].rank(method="first"), 4, labels=m_labels).astype(int)
    
    # Create RFM segment string
    df_scored['rfm_segment'] = df_scored['r'].astype(str) + df_scored['f'].astype(str) + df_scored['m'].astype(str)
    df_scored['rfm_score'] = df_scored['r'] + df_scored['f'] + df_scored['m']
    
    return df_scored

# RFM Segment Rules
RFM_SEGMENT_RULES = {
    "Champions": {"rank": 1, "color": "#008080", "R_low": 3, "R_high": 4, "FM_low": 3, "FM_high": 4},
    "Loyal": {"rank": 2, "color": "#66E0E0", "R_low": 2, "R_high": 4, "FM_low": 2, "FM_high": 4},
    "Potential": {"rank": 3, "color": "#B0C4DE", "R_low": 2, "R_high": 4, "FM_low": 0, "FM_high": 2},
    "New": {"rank": 4, "color": "#3399FF", "R_low": 3, "R_high": 4, "FM_low": 0, "FM_high": 1},
    "Can't Lose Them": {"rank": 5, "color": "#FF6666", "R_low": 0, "R_high": 2, "FM_low": 3, "FM_high": 4},
    "Needs Attention": {"rank": 6, "color": "#6666FF", "R_low": 1, "R_high": 2, "FM_low": 0, "FM_high": 3},
    "Lost": {"rank": 7, "color": "#FFB266", "R_low": 0, "R_high": 1, "FM_low": 0, "FM_high": 3},
}

def label_rfm_segments(df_rfm_scored, rule_dict=RFM_SEGMENT_RULES):
    """Apply rule-based RFM labeling"""
    df_labeled = df_rfm_scored.copy()
    df_labeled["fm_score"] = (df_labeled["f"] + df_labeled["m"]) / 2
    
    def assign_segment(row):
        r, fm = row["r"], row["fm_score"]
        for segment, rule in sorted(rule_dict.items(), key=lambda x: x[1]["rank"]):
            if (rule["R_low"] < r <= rule["R_high"]) and (rule["FM_low"] < fm <= rule["FM_high"]):
                return segment
        return "Uncategorized"
    
    df_labeled["segment_label"] = df_labeled.apply(assign_segment, axis=1)
    return df_labeled

def feature_engineering_rfm(df_rfm):
    """Apply feature engineering for clustering"""
    df_eng = df_rfm.copy()
    
    # Log transformation
    df_eng["recency_log"] = np.log1p(df_eng["recency"])
    df_eng["monetary_log"] = np.log1p(df_eng["monetary"])
    
    # Robust scaling
    scaler = RobustScaler()
    scaled_features = scaler.fit_transform(
        df_eng[['recency_log', 'frequency', 'monetary_log']]
    )
    
    df_eng[['recency_scaled', 'frequency_scaled', 'monetary_scaled']] = scaled_features
    return df_eng, scaler

def kmeans_clustering(df_scaled_features, n_clusters=4, random_state=42):
    """Perform K-means clustering"""
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10, init='k-means++')
    clusters = kmeans.fit_predict(df_scaled_features)
    return clusters, kmeans

def calculate_silhouette_scores(df_scaled, cluster_range=range(2, 11)):
    """Calculate silhouette scores for different cluster numbers"""
    scores = []
    inertias = []
    
    for k in cluster_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(df_scaled)
        score = silhouette_score(df_scaled, cluster_labels)
        scores.append(score)
        inertias.append(kmeans.inertia_)
    
    return scores, inertias

def create_segment_analysis(df, segment_col):
    """Create comprehensive segment analysis"""
    analysis = df.groupby(segment_col).agg({
        'customer_id': 'count',
        'recency': ['mean', 'median'],
        'frequency': ['mean', 'median'], 
        'monetary': ['mean', 'median', 'sum']
    }).round(2)
    
    analysis.columns = ['customer_count', 'recency_mean', 'recency_median',
                       'frequency_mean', 'frequency_median', 
                       'monetary_mean', 'monetary_median', 'total_revenue']
    
    analysis = analysis.reset_index()
    analysis['customer_pct'] = (analysis['customer_count'] / analysis['customer_count'].sum() * 100).round(1)
    analysis['revenue_pct'] = (analysis['total_revenue'] / analysis['total_revenue'].sum() * 100).round(1)
    
    return analysis

# =====================================
# Main Application
# =====================================

def main():
    st.markdown('<h1 class="main-header">📊 Customer Segmentation Project - Store X</h1>', 
                unsafe_allow_html=True)
    
    # Sidebar configuration
    st.sidebar.markdown("## ⚙️ Project Configuration")
    
    # Clustering parameters
    st.sidebar.markdown("### 🎯 Clustering Parameters")
    k_kmeans = st.sidebar.slider("Number of Clusters K-Means", 2, 10, 4)
    
    # Main content area
    st.markdown("---")
    
    # Step 1: Business Understanding
    st.markdown('<h2 class="section-header">1. Business Understanding</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div class="insight-box">
        <h4>🎯 Business Problem</h4>
        <p>Store X primarily sells essential items (vegetables, fruits, meat, fish, eggs, milk, beverages...) 
        to retail customers. The store owner aims to:</p>
        <ul>
            <li>🚀 Sell more goods</li>
            <li>🎯 Introduce products to the right customer segments</li>
            <li>💝 Care for and satisfy customers</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="insight-box">
        <h4>🎯 Project Goals</h4>
        <p>Build a customer segmentation system using RFM analysis to:</p>
        <ul>
            <li>📊 Identify different customer groups</li>
            <li>💡 Develop appropriate business strategies</li>
            <li>🎯 Optimize customer care</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Step 2: Data Understanding & Acquisition
    st.markdown('<h2 class="section-header">2. Data Understanding & Acquisition</h2>', unsafe_allow_html=True)
    
    st.markdown("### 📁 Data Upload")
    st.info("Please upload 2 CSV files: **Products_with_Categories.csv** and **Transactions.csv**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        products_file = st.file_uploader(
            "📦 Products File (CSV)",
            type=["csv"],
            help="File containing product information: ID, name, price, category"
        )
    
    with col2:
        transactions_file = st.file_uploader(
            "🛒 Transactions File (CSV)", 
            type=["csv"],
            help="File containing transaction information: customer, date, product, quantity"
        )
    
    if products_file and transactions_file:
        # Load data
        df_products_raw = pd.read_csv(products_file)
        df_transactions_raw = pd.read_csv(transactions_file)
        
        st.success("✅ Data loaded successfully!")
        
        # Format the date column
        df_trans_temp = fn_date_format(df_transactions_raw.copy(), cols=["Date"], date_pattern="%d-%m-%Y")
        
        # Determine min and max dates from transactions
        min_date = df_trans_temp["Date"].min()
        max_date = df_trans_temp["Date"].max()
        
        # Sidebar date inputs with min/max from data
        st.sidebar.markdown("### 📅 Report Date Range")
        report_from = st.sidebar.date_input(
            "📅 Report Date From",
            value=min_date.date(),
            min_value=min_date.date(),
            max_value=max_date.date(),
            help="Minimum date from the dataset"
        )
        report_to = st.sidebar.date_input(
            "📅 Report Date To",
            value=max_date.date(),
            min_value=report_from,
            max_value=max_date.date(),
            help="Maximum date from the dataset"
        )
        
        # Convert to datetime for processing
        report_from_dt = pd.to_datetime(report_from)
        report_to_dt = pd.to_datetime(report_to)
        
        # Data preview
        st.markdown("### 👀 Data Preview")
        
        tab1, tab2 = st.tabs(["📦 Products Data", "🛒 Transactions Data"])
        
        with tab1:
            st.write(f"**Shape:** {df_products_raw.shape[0]} rows × {df_products_raw.shape[1]} columns")
            st.dataframe(df_products_raw.head())
            
            # Data quality check
            st.write("**Data Quality Check:**")
            st.dataframe(fn_null_nan_count(df_products_raw))
        
        with tab2:
            st.write(f"**Shape:** {df_transactions_raw.shape[0]} rows × {df_transactions_raw.shape[1]} columns")
            st.dataframe(df_transactions_raw.head())
            
            # Data quality check
            st.write("**Data Quality Check:**")
            st.dataframe(fn_null_nan_count(df_transactions_raw))
        
        # Step 3: Data Preparation
        st.markdown('<h2 class="section-header">3. Data Preparation</h2>', unsafe_allow_html=True)
        
        with st.spinner("Processing and cleaning data..."):
            df_clean, invalid_amount, invalid_date, invalid_name = clean_and_process_data(
                df_products_raw, df_transactions_raw, report_from_dt, report_to_dt
            )
        
        # Data cleaning summary
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("❌ Amount ≤ 0", invalid_amount)
        with col2:
            st.metric("📅 Invalid Date", invalid_date)  
        with col3:
            st.metric("🏷️ Missing Product", invalid_name)
        with col4:
            total_clean = len(df_clean)
            st.metric("✅ Clean Records", f"{total_clean:,}")
        
        st.success(f"Data has been cleaned! Removed {invalid_amount + invalid_date + invalid_name:,} invalid records.")
        
        # Create RFM table
        df_rfm, max_date = create_rfm_table(df_clean, report_to_dt)
        
        # RFM overview
        st.markdown("### 📊 RFM Analysis Overview")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("👥 Total Customers", f"{len(df_rfm):,}")
        with col2:
            st.metric("📅 Max Transaction Date", max_date.strftime("%d/%m/%Y"))
        with col3:
            st.metric("💰 Total Revenue", f"${df_rfm['monetary'].sum():,.0f}")
        
        # RFM distributions
        st.markdown("### 📈 RFM Distributions")
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        rfm_cols = ['recency', 'frequency', 'monetary']
        
        for i, col in enumerate(rfm_cols):
            axes[i].hist(df_rfm[col], bins=30, alpha=0.7, color=sns.color_palette()[i])
            axes[i].set_title(f'{col.capitalize()} Distribution')
            axes[i].set_xlabel(col.capitalize())
            axes[i].set_ylabel('Frequency')
            axes[i].grid(True, alpha=0.3)
            
            # Add skewness info
            skewness = skew(df_rfm[col])
            axes[i].text(0.7, 0.9, f'Skewness: {skewness:.2f}', 
                        transform=axes[i].transAxes, bbox=dict(boxstyle="round", facecolor='wheat'))
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # RFM correlation
        st.markdown("### 🔗 RFM Correlation Matrix")
        
        fig, ax = plt.subplots(figsize=(8, 6))
        corr_matrix = df_rfm[rfm_cols].corr()
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', 
                   center=0, square=True, ax=ax, cbar_kws={'shrink': 0.8})
        ax.set_title('RFM Correlation Matrix', fontsize=14, fontweight='bold')
        st.pyplot(fig)
        
        # Step 4: RFM Segmentation Methods
        st.markdown('<h2 class="section-header">4. RFM Segmentation Methods</h2>', unsafe_allow_html=True)
        
        # Create tabs for different methods
        tab1, tab2, tab3 = st.tabs([
            "📊 Rule-based RFM", 
            "🎯 K-Means Clustering",
            "📈 Method Comparison"
        ])
        
        # Method 1: Rule-based RFM
        with tab1:
            st.markdown("### 📊 Rule-based RFM Segmentation")
            
            # Show RFM map
            st.markdown("#### 🗺️ RFM Segmentation Map")
            rfm_map_fig = create_rfm_map_visual()
            st.pyplot(rfm_map_fig)
            
            # Calculate RFM scores and apply rules
            df_rfm_scored = rfm_quartile_scoring(df_rfm)
            df_rfm_labeled = label_rfm_segments(df_rfm_scored)
            
            # Segment analysis
            segment_analysis = create_segment_analysis(df_rfm_labeled, 'segment_label')
            
            st.markdown("#### 📈 Segment Analysis")
            st.dataframe(segment_analysis)
            
            # Visualizations
            st.markdown("#### 📊 Segment Visualizations")
            
            # Count plot
            fig, ax = plt.subplots(figsize=(10, 6))
            segment_order = segment_analysis.sort_values('customer_count', ascending=True)['segment_label'].tolist()
            colors = [RFM_SEGMENT_RULES[seg]['color'] for seg in segment_order]
            
            bars = ax.barh(segment_order, segment_analysis.set_index('segment_label').loc[segment_order, 'customer_count'])
            for i, bar in enumerate(bars):
                bar.set_color(colors[i])
                width = bar.get_width()
                ax.text(width + 10, bar.get_y() + bar.get_height()/2, 
                       f'{int(width):,} ({segment_analysis.set_index("segment_label").loc[segment_order[i], "customer_pct"]:.1f}%)',
                       ha='left', va='center', fontweight='bold')
            
            ax.set_title('Customer Distribution by RFM Segments', fontsize=14, fontweight='bold')
            ax.set_xlabel('Number of Customers')
            ax.grid(axis='x', alpha=0.3)
            st.pyplot(fig)
            
            # 3D scatter plot
            fig = px.scatter_3d(
                df_rfm_labeled, 
                x='recency', y='frequency', z='monetary',
                color='segment_label',
                color_discrete_map={seg: RFM_SEGMENT_RULES[seg]['color'] for seg in RFM_SEGMENT_RULES},
                title='RFM 3D Scatter Plot by Segments',
                labels={'recency': 'Recency (days)', 'frequency': 'Frequency', 'monetary': 'Monetary ($)'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Method 2: K-Means Clustering
        with tab2:
            st.markdown("### 🎯 K-Means Clustering")
            
            # Feature engineering
            df_rfm_eng, scaler = feature_engineering_rfm(df_rfm)
            scaled_features = df_rfm_eng[['recency_scaled', 'frequency_scaled', 'monetary_scaled']]
            
            # Elbow method and silhouette analysis
            st.markdown("#### 📈 Optimal Number of Clusters")
            
            with st.spinner("Calculating optimal clusters..."):
                silhouette_scores, inertias = calculate_silhouette_scores(scaled_features)
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Elbow plot
            k_range = range(2, 11)
            ax1.plot(k_range, inertias, 'bo-')
            ax1.set_title('Elbow Method')
            ax1.set_xlabel('Number of Clusters (k)')
            ax1.set_ylabel('Inertia (Within-cluster Sum of Squares)')
            ax1.grid(True, alpha=0.3)
            
            # Silhouette plot
            ax2.plot(k_range, silhouette_scores, 'ro-')
            ax2.set_title('Silhouette Score')
            ax2.set_xlabel('Number of Clusters (k)')
            ax2.set_ylabel('Silhouette Score')
            ax2.grid(True, alpha=0.3)
            
            # Highlight selected k
            ax1.axvline(x=k_kmeans, color='red', linestyle='--', alpha=0.7, label=f'Selected k={k_kmeans}')
            ax2.axvline(x=k_kmeans, color='red', linestyle='--', alpha=0.7, label=f'Selected k={k_kmeans}')
            ax1.legend()
            ax2.legend()
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Perform K-means clustering
            kmeans_clusters, kmeans_model = kmeans_clustering(scaled_features, k_kmeans)
            df_rfm_kmeans = df_rfm_eng.copy()
            df_rfm_kmeans['cluster'] = kmeans_clusters
            
            # Cluster analysis
            st.markdown("#### 🔍 K-Means Cluster Analysis")
            
            kmeans_analysis = create_segment_analysis(df_rfm_kmeans, 'cluster')
            st.dataframe(kmeans_analysis)
            
            # Visualizations
            fig = px.scatter_3d(
                df_rfm_kmeans,
                x='recency', y='frequency', z='monetary',
                color='cluster',
                title='K-Means Clustering Results (3D)',
                labels={'recency': 'Recency (days)', 'frequency': 'Frequency', 'monetary': 'Monetary ($)'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Method 3: Comparison
        with tab3:
            st.markdown("### 📈 Method Comparison")
            
            # Calculate silhouette scores for comparison
            rfm_silhouette = silhouette_score(scaled_features, df_rfm_labeled['segment_label'].factorize()[0])
            kmeans_silhouette = silhouette_score(scaled_features, kmeans_clusters)
            
            # Comparison table
            comparison_data = {
                'Method': ['Rule-based RFM', 'K-Means'],
                'Number of Segments': [len(df_rfm_labeled['segment_label'].unique()), k_kmeans],
                'Silhouette Score': [rfm_silhouette, kmeans_silhouette],
                'Interpretability': ['High', 'Medium'],
                'Scalability': ['High', 'High'],
                'Business Logic': ['Strong', 'Weak']
            }
            
            comparison_df = pd.DataFrame(comparison_data)
            
            st.markdown("#### 📊 Performance Comparison")
            st.dataframe(comparison_df)
            
            # Visualization comparison
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            
            methods_data = [
                (df_rfm_labeled, 'segment_label', 'Rule-based RFM'),
                (df_rfm_kmeans, 'cluster', 'K-Means')
            ]
            
            for i, (data, col, title) in enumerate(methods_data):
                segment_counts = data[col].value_counts()
                axes[i].pie(segment_counts.values, labels=segment_counts.index, autopct='%1.1f%%')
                axes[i].set_title(f'{title}\n({len(segment_counts)} segments)')
            
            plt.tight_layout()
            st.pyplot(fig)
        
        # Step 5: Model Selection and Insights
        st.markdown('<h2 class="section-header">5. Model Selection & Business Insights</h2>', 
                   unsafe_allow_html=True)
        
        # Select best method based on business requirements
        st.markdown("### 🎯 Recommended Method: Rule-based RFM")
        
        st.markdown("""
        <div class="insight-box">
        <h4>🏆 Why Rule-based RFM is Recommended:</h4>
        <ul>
            <li><strong>High Interpretability:</strong> Each segment has clear business meaning</li>
            <li><strong>Actionable Insights:</strong> Direct connection to marketing strategies</li>
            <li><strong>Industry Standard:</strong> Widely adopted in retail and e-commerce</li>
            <li><strong>Easy Communication:</strong> Stakeholders can easily understand segments</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Detailed segment insights
        st.markdown("### 💡 Detailed Segment Insights")
        
        segment_insights = {
            "Champions": {
                "description": "Best customers - recent purchases, frequent, and high spending",
                "characteristics": "Low Recency, High Frequency, High Monetary",
                "business_value": "Contribute the highest revenue, high loyalty potential"
            },
            "Loyal": {
                "description": "Loyal customers - regular purchases with stable spending", 
                "characteristics": "Medium Recency, High Frequency, Medium-High Monetary",
                "business_value": "Backbone of the business, reliable"
            },
            "Potential": {
                "description": "Potential customers - can develop into Loyal/Champions",
                "characteristics": "Medium Recency, Medium Frequency, Medium Monetary", 
                "business_value": "Growth opportunity, needs nurturing"
            },
            "New": {
                "description": "New customers - just started purchasing",
                "characteristics": "Low Recency, Low Frequency, Low Monetary",
                "business_value": "Development potential, needs good onboarding"
            },
            "Can't Lose Them": {
                "description": "VIP customers at risk - high past spending but long absence",
                "characteristics": "High Recency, High Frequency, High Monetary",
                "business_value": "High-value customers at risk, needs win-back campaign"
            },
            "Needs Attention": {
                "description": "Customers needing attention - decreasing engagement",
                "characteristics": "High Recency, Medium Frequency, Medium Monetary",
                "business_value": "Risk segment, needs early intervention"
            },
            "Lost": {
                "description": "Lost customers - long absence and low value",
                "characteristics": "High Recency, Low Frequency, Low Monetary", 
                "business_value": "Cost to reactivate > potential value"
            }
        }
        
        for segment in segment_analysis['segment_label'].unique():
            if segment in segment_insights:
                insight = segment_insights[segment]
                
                with st.expander(f"🔍 {segment} Segment Analysis"):
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        seg_data = segment_analysis[segment_analysis['segment_label'] == segment].iloc[0]
                        st.metric("Customer Count", f"{int(seg_data['customer_count']):,}")
                        st.metric("Customer %", f"{seg_data['customer_pct']:.1f}%")
                        st.metric("Revenue %", f"{seg_data['revenue_pct']:.1f}%")
                        st.metric("Avg Monetary", f"${seg_data['monetary_mean']:.0f}")
                    
                    with col2:
                        st.write(f"**Description:** {insight['description']}")
                        st.write(f"**Characteristics:** {insight['characteristics']}")  
                        st.write(f"**Business Value:** {insight['business_value']}")
        
        # Step 6: Deployment & Recommendations
        st.markdown('<h2 class="section-header">6. Deployment & Marketing Recommendations</h2>', 
                   unsafe_allow_html=True)
        
        recommendations = {
            "Champions": {
                "strategy": "Reward & Retention",
                "actions": [
                    "🎁 Exclusive offers and early access to new products",
                    "💎 VIP customer program with special benefits", 
                    "🤝 Referral programs with attractive incentives",
                    "📞 Personal account manager for large purchases",
                    "🎉 Birthday and anniversary special offers"
                ]
            },
            "Loyal": {
                "strategy": "Maintain & Upsell",
                "actions": [
                    "📊 Product recommendations based on purchase history",
                    "💰 Volume discounts and bundle offers",
                    "📱 Mobile app with loyalty points system",
                    "📧 Regular newsletters with helpful content",
                    "🎯 Cross-sell complementary products"
                ]
            },
            "Potential": {
                "strategy": "Develop & Engage", 
                "actions": [
                    "📚 Educational content about product benefits",
                    "🎁 Free samples of premium products",
                    "💬 Personalized shopping assistance",
                    "🔔 Reminder campaigns for repeat purchases",
                    "🎯 Targeted promotions on preferred categories"
                ]
            },
            "New": {
                "strategy": "Welcome & Onboard",
                "actions": [
                    "👋 Welcome email series with store introduction", 
                    "🎁 New customer discount on second purchase",
                    "📋 Preference survey to understand needs",
                    "📞 Follow-up call to ensure satisfaction",
                    "🛒 Shopping guides and product tutorials"
                ]
            },
            "Can't Lose Them": {
                "strategy": "Win-back & Recovery",
                "actions": [
                    "🆘 Urgent win-back campaign with significant discounts",
                    "📞 Personal outreach to understand concerns",
                    "🎁 Special comeback offers with added value",
                    "📧 Apology campaign if service issues identified", 
                    "🔄 Reactivation series with compelling content"
                ]
            },
            "Needs Attention": {
                "strategy": "Re-engage & Prevent Churn",
                "actions": [
                    "⏰ Limited-time offers to create urgency",
                    "💡 Product education and usage tips",
                    "🎯 Targeted discounts on previously purchased items",
                    "📱 App notifications for special deals",
                    "🤝 Customer feedback survey and improvement promises"
                ]
            },
            "Lost": {
                "strategy": "Minimal Investment Recovery",
                "actions": [
                    "📧 Low-cost email campaigns with deep discounts",
                    "🆕 Highlight new products and improvements",
                    "🎁 One-time significant discount to test interest",
                    "📊 A/B testing for cost-effective approaches",
                    "⏸️ Consider removing from active campaigns if no response"
                ]
            }
        }
        
        st.markdown("### 🎯 Segment-Specific Marketing Strategies")
        
        for segment in ["Champions", "Loyal", "Potential", "New", "Can't Lose Them", "Needs Attention", "Lost"]:
            if segment in recommendations:
                rec = recommendations[segment]
                
                with st.expander(f"📈 {segment} - {rec['strategy']}"):
                    st.markdown("#### Recommended Actions:")
                    for action in rec['actions']:
                        st.markdown(f"- {action}")
        
        # Step 7: Implementation Timeline
        st.markdown('<h2 class="section-header">7. Implementation Timeline</h2>', 
                   unsafe_allow_html=True)
        
        timeline_data = {
            "Phase": ["Phase 1: Immediate", "Phase 2: Short-term", "Phase 3: Medium-term", "Phase 4: Long-term"],
            "Duration": ["Week 1-2", "Week 3-8", "Month 3-6", "Month 6+"],
            "Focus": [
                "Champions & Can't Lose Them campaigns",
                "Loyal customer retention & New customer onboarding", 
                "Potential customer development & At-risk prevention",
                "Lost customer reactivation & System optimization"
            ],
            "Expected Impact": ["High revenue protection", "Stable base growth", "Market expansion", "Efficiency gains"]
        }
        
        timeline_df = pd.DataFrame(timeline_data)
        st.dataframe(timeline_df, use_container_width=True)
        
        # Summary and Export Options
        st.markdown('<h2 class="section-header">8. Summary & Export</h2>', 
                   unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            <div class="insight-box">
            <h4>🎯 Project Summary</h4>
            <ul>
                <li><strong>Objective Achieved:</strong> Successfully segmented customers using RFM analysis</li>
                <li><strong>Total Customers Analyzed:</strong> {total_customers:,}</li>
                <li><strong>Recommended Method:</strong> Rule-based RFM segmentation</li>
                <li><strong>Key Insight:</strong> {champions_pct:.1f}% Champions generate {champions_rev:.1f}% of revenue</li>
                <li><strong>Business Impact:</strong> Clear actionable strategies for each customer segment</li>
            </ul>
            </div>
            """.format(
                total_customers=len(df_rfm),
                champions_pct=segment_analysis[segment_analysis['segment_label']=='Champions']['customer_pct'].iloc[0] if 'Champions' in segment_analysis['segment_label'].values else 0,
                champions_rev=segment_analysis[segment_analysis['segment_label']=='Champions']['revenue_pct'].iloc[0] if 'Champions' in segment_analysis['segment_label'].values else 0
            ), unsafe_allow_html=True)
        
        with col2:
            st.markdown("#### 📊 Export Options")
            
            # Prepare export data
            export_data = df_rfm_labeled[['customer_id', 'recency', 'frequency', 'monetary', 
                                        'r', 'f', 'm', 'rfm_segment', 'rfm_score', 'segment_label']]
            
            csv = export_data.to_csv(index=False)
            st.download_button(
                label="📥 Download Customer Segments (CSV)",
                data=csv,
                file_name=f"customer_segments_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
            
            # Analysis summary
            summary_csv = segment_analysis.to_csv(index=False)
            st.download_button(
                label="📊 Download Segment Analysis (CSV)", 
                data=summary_csv,
                file_name=f"segment_analysis_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        
        # Footer
        st.markdown("---")
        st.markdown("""
        <div style='text-align: center; color: #666; padding: 2rem;'>
            <h4>📊 Customer Segmentation Project Complete</h4>
            <p>Developed with ❤️ for Store X | Data-Driven Customer Intelligence</p>
        </div>
        """, unsafe_allow_html=True)
        
    else:
        # Instructions when no files uploaded
        st.info("👆 Please upload both CSV files to start the analysis")
        
        # Show sample data format
        st.markdown("### 📋 Expected Data Format")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Products_with_Categories.csv:**")
            sample_products = pd.DataFrame({
                'ProductId': [1, 2, 3],
                'ProductName': ['Apple', 'Milk', 'Bread'],
                'Price': [2.5, 3.0, 1.5],
                'Category': ['Fruit', 'Dairy', 'Bakery']
            })
            st.dataframe(sample_products)
        
        with col2:
            st.markdown("**Transactions.csv:**")
            sample_transactions = pd.DataFrame({
                'Member_number': [1001, 1002, 1001],
                'Date': ['01-01-2025', '02-01-2025', '03-01-2025'],
                'productId': [1, 2, 3],
                'Items': [2, 1, 3]
            })
            st.dataframe(sample_transactions)

if __name__ == "__main__":
    main()