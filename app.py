import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.stats import skew
import streamlit as st
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# Constants
DASH_LEN = 30
N_ROWS = 3
REPORT_DATE_FIXED = pd.to_datetime("2025-08-20")
DPI = 150
FIGSIZE = (1920 / DPI, 1080 / DPI)
RANDOM_STATE = 42
CASE_STUDY = "MLTT_"

# Segment rules
rfm_segment_rule_dict = {
    "Champions": {"rank": 3, "color": "#008080", "R_low": 3, "R_high": 4, "FM_low": 3, "FM_high": 4},
    "New": {"rank": 4, "color": "#3399FF", "R_low": 3, "R_high": 4, "FM_low": 0, "FM_high": 1},
    "Loyal": {"rank": 1, "color": "#66E0E0", "R_low": 2, "R_high": 4, "FM_low": 2, "FM_high": 4},
    "Potential": {"rank": 2, "color": "#B0C4DE", "R_low": 2, "R_high": 4, "FM_low": 0, "FM_high": 2},
    "Can't Lose Them": {"rank": 5, "color": "#FF6666", "R_low": 0, "R_high": 2, "FM_low": 3, "FM_high": 4},
    "Needs Attention": {"rank": 6, "color": "#6666FF", "R_low": 1, "R_high": 2, "FM_low": 0, "FM_high": 3},
    "Lost": {"rank": 7, "color": "#FFB266", "R_low": 0, "R_high": 1, "FM_low": 0, "FM_high": 3},
}

# Page Configuration
st.set_page_config(
    page_title="Customer Segmentation - Store X",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark theme
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
        font-weight: bold;
    }
    .section-header {
        color: #00BFFF;
        border-bottom: 2px solid #00BFFF;
        padding-bottom: 0.5rem;
        margin-top: 2rem;
        margin-bottom: 1rem;
        font-size: 1.8rem;
        font-weight: bold;
    }
    .subsection-header {
        color: #87CEEB;
        font-size: 1.3rem;
        font-weight: bold;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #252525;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
        margin: 0.5rem;
        color: #E0E0E0;
        border: 1px solid #444444;
    }
    .insight-box {
        background-color: #1E1E1E;
        padding: 1.5rem;
        border-left: 4px solid #00BFFF;
        margin: 1rem 0;
        color: #E0E0E0;
        border-radius: 0.5rem;
    }
    .recommendation-box {
        background-color: #2D2D2D;
        padding: 1.5rem;
        border-left: 4px solid #32CD32;
        margin: 1rem 0;
        color: #E0E0E0;
        border-radius: 0.5rem;
    }
    .warning-box {
        background-color: #2D1B1B;
        padding: 1.5rem;
        border-left: 4px solid #FF6B6B;
        margin: 1rem 0;
        color: #E0E0E0;
        border-radius: 0.5rem;
    }
    .stDataFrame {
        color: #E0E0E0;
        background-color: #252525;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Utility Functions
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
            "% Null": null_pct,
            "Data Type": str(df[col].dtype)
        })
    return pd.DataFrame(results)

@st.cache_data
def fn_date_format(df: pd.DataFrame, cols: list = None, date_pattern: str = "%d-%m-%Y") -> pd.DataFrame:
    if cols is None:
        cols = df.select_dtypes(include=['object']).columns
    df2 = df.copy()
    for col in cols:
        if col in df2.columns:
            df2[col] = pd.to_datetime(df2[col], format=date_pattern, errors="coerce", dayfirst=True)
            invalid_dates = df2[df2[col].isna()]
            if not invalid_dates.empty:
                st.warning(f"Found {len(invalid_dates)} invalid dates in '{col}' column. These will be removed.")
        else:
            st.error(f"Column '{col}' not found in DataFrame. Available columns: {list(df2.columns)}")
    return df2

def create_rfm_map_visual():
    """Create RFM segmentation map visualization"""
    fig, ax = plt.subplots(figsize=(10, 8))
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
                ha="center", va="center", fontsize=12, weight="bold", color='white')
    
    ax.set_xlim(0.5, 4.5)
    ax.set_ylim(0.5, 4.5)
    ax.set_xticks([1, 2, 3, 4])
    ax.set_yticks([1, 2, 3, 4])
    ax.set_xlabel("R (Recency Score)", fontsize=14, fontweight='bold')
    ax.set_ylabel("(F + M) / 2 (Average Frequency + Monetary)", fontsize=14, fontweight='bold')
    ax.set_title("RFM Customer Segmentation Map", fontsize=16, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig

@st.cache_data
def clean_and_process_data(df_products, df_transactions, report_from_dt, report_to_dt, date_col):
    """Clean and preprocess the data"""
    products_cols_ori = list(df_products.columns)
    transactions_cols_ori = list(df_transactions.columns)
    
    # Rename columns, preserving the original date column name until after date processing
    df_products = fn_rename(df_products, products_cols_ori,
                           ["product_id", "product_name", "product_price", "product_category"])
    df_transactions = fn_rename(df_transactions, transactions_cols_ori,
                               ["customer_id", date_col, "product_id", "quantity"])
    
    # Format the date column
    df_transactions = fn_date_format(df_transactions, cols=[date_col], date_pattern="%d-%m-%Y")
    
    # Rename date column to 'order_date' after formatting
    df_transactions = df_transactions.rename(columns={date_col: "order_date"})
    
    # Remove invalid dates (NaT or outside date range)
    invalid_dates = df_transactions[df_transactions["order_date"].isna() |
                                   (df_transactions["order_date"] < report_from_dt) |
                                   (df_transactions["order_date"] > report_to_dt)]
    if not invalid_dates.empty:
        st.warning(f"Removed {len(invalid_dates)} transactions with invalid dates (NaT or outside [{report_from_dt.date()}, {report_to_dt.date()}]).")
    df_transactions = df_transactions[df_transactions["order_date"].notna() &
                                     (df_transactions["order_date"] >= report_from_dt) &
                                     (df_transactions["order_date"] <= report_to_dt)]
    
    df_trans_pre = df_transactions.merge(df_products, on="product_id", how="left")
    df_trans_pre["amount"] = df_trans_pre["quantity"] * df_trans_pre["product_price"]
    
    invalid_amount = df_trans_pre[df_trans_pre["amount"] <= 0]
    invalid_name = df_trans_pre[df_trans_pre["product_name"].isna()]
    
    df_clean = df_trans_pre[
        (df_trans_pre["amount"] > 0) &
        (df_trans_pre["product_name"].notna())
    ].copy()
    
    return df_clean, len(invalid_amount), len(invalid_dates), len(invalid_name)

@st.cache_data
def create_rfm_table(df_trans_clean, report_to_dt):
    """Create RFM table from transaction data"""
    max_date = df_trans_clean["order_date"].max().date() if not df_trans_clean["order_date"].isna().all() else report_to_dt.date()
    
    def recency_calc(x):
        if x.isna().all() or x.max() is pd.NaT:
            return 0
        last_purchase = x.max().date()
        if last_purchase > report_to_dt.date():
            return 0
        return max(0, (report_to_dt.date() - last_purchase).days)
    
    df_rfm = df_trans_clean.groupby("customer_id").agg(
        recency=("order_date", recency_calc),
        frequency=("order_date", lambda x: len(x.unique())),
        monetary=("amount", lambda x: round(x.sum(), 2))
    ).reset_index()
    
    # Ensure recency is integer
    df_rfm["recency"] = df_rfm["recency"].astype(int)
    
    # Debugging
    st.write("Recency dtype:", df_rfm["recency"].dtype)
    st.write("Sample Recency Values:", df_rfm[["customer_id", "recency"]].head())
    
    if (df_rfm["recency"] < 0).any():
        st.error("Negative recency values detected. Please check date inputs and data.")
    if (df_rfm["recency"] == 0).all() and not df_trans_clean.empty:
        st.warning("All recency values are 0. Check if order_date is correctly parsed or within the date range.")
    
    return df_rfm, max_date

def rfm_quartile_scoring(df_rfm):
    """Calculate RFM quartile scores"""
    r_labels = range(4, 0, -1)
    f_labels = range(1, 5)
    m_labels = range(1, 5)
    
    df_scored = df_rfm.copy()
    
    if len(df_rfm) == 1 and 'r' not in df_rfm.columns:
        recency = df_rfm['recency'].iloc[0]
        df_scored['r'] = [4 if recency < 30 else 3 if recency < 60 else 2 if recency < 90 else 1]
        frequency = df_rfm['frequency'].iloc[0]
        df_scored['f'] = [4 if frequency > 10 else 3 if frequency >= 5 else 2 if frequency >= 1 else 1]
        monetary = df_rfm['monetary'].iloc[0]
        df_scored['m'] = [4 if monetary > 1000 else 3 if monetary >= 500 else 2 if monetary >= 100 else 1]
    elif 'r' in df_rfm.columns:
        df_scored['r'] = df_rfm['r'].clip(1, 4).astype(int)
        df_scored['f'] = df_rfm['f'].clip(1, 4).astype(int)
        df_scored['m'] = df_rfm['m'].clip(1, 4).astype(int)
    else:
        df_scored['r'] = pd.qcut(df_rfm["recency"].rank(method="first"), 4, labels=r_labels).astype(int)
        df_scored['f'] = pd.qcut(df_rfm["frequency"].rank(method="first"), 4, labels=f_labels).astype(int)
        df_scored['m'] = pd.qcut(df_rfm["monetary"].rank(method="first"), 4, labels=m_labels).astype(int)
    
    df_scored['rfm_segment'] = df_scored['r'].astype(str) + df_scored['f'].astype(str) + df_scored['m'].astype(str)
    df_scored['rfm_score'] = df_scored['r'] + df_scored['f'] + df_scored['m']
    
    return df_scored

def create_single_customer_rfm(r_value, f_value, m_value):
    """Create single customer RFM dataframe with direct R, F, M scores (1-4)"""
    return pd.DataFrame({
        'customer_id': ['CUSTOMER_001'],
        'r': [r_value],
        'f': [f_value],
        'm': [m_value]
    })

def label_rfm_segments(df_rfm_scored, rule_dict=rfm_segment_rule_dict):
    """Apply rule-based RFM labeling"""
    df_labeled = df_rfm_scored.copy()
    df_labeled["_r"] = df_labeled["r"]
    df_labeled["_fm"] = (df_labeled["f"] + df_labeled["m"]) / 2
    
    def assign_segment(row):
        r, fm = row["_r"], row["_fm"]
        for seg, rule in sorted(rule_dict.items(), key=lambda x: x[1]["rank"], reverse=True):
            if (rule["R_low"] < r <= rule["R_high"]) and (rule["FM_low"] < fm <= rule["FM_high"]):
                return seg
        return "Uncategorized"
    
    df_labeled["rfm_segment_labeled"] = df_labeled.apply(assign_segment, axis=1)
    df_labeled = df_labeled.drop(columns=["_r", "_fm"])
    return df_labeled

def feature_engineering_rfm(df_rfm):
    """Apply feature engineering for clustering"""
    df_eng = df_rfm.copy()
    
    df_eng["recency_log"] = np.log1p(df_eng["recency"])
    df_eng["monetary_log"] = np.log1p(df_eng["monetary"])
    
    scaler = RobustScaler()
    scaled_features = scaler.fit_transform(
        df_eng[['recency_log', 'frequency', 'monetary_log']]
    )
    
    df_eng[['recency_log_scaled', 'frequency_scaled', 'monetary_log_scaled']] = scaled_features
    return df_eng, scaler

def kmeans_clustering(df_scaled_features, n_clusters=4, random_state=RANDOM_STATE):
    """Perform K-means clustering"""
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10, max_iter=300, init='k-means++')
    clusters = kmeans.fit_predict(df_scaled_features)
    return clusters.astype(str), kmeans

def create_segment_analysis(df, segment_col):
    """Create comprehensive segment analysis"""
    df_agg = df.groupby(segment_col).agg(
        customer_count=("customer_id", "nunique"),
        customer_distribution=("customer_id", lambda x: 100 * x.nunique() / df["customer_id"].nunique()),
        revenue_distribution=("monetary", lambda x: 100 * x.sum() / df["monetary"].sum()),
        recency_mean=("recency", "mean"),
        frequency_mean=("frequency", "mean"),
        monetary_mean=("monetary", "mean")
    ).round(1).reset_index()
    
    return df_agg

def get_customer_recommendations(segment_label):
    """Get marketing recommendations for specific customer segment"""
    recommendations = {
        "Champions": {
            "strategy": "Retention & Enhanced Experience",
            "description": "Recent, frequent, and high-spending customers",
            "actions": [
                "🎁 Exclusive offers and early access to new products",
                "💎 VIP program with special benefits",
                "🤝 Referral program with incentives",
                "📞 Dedicated support for large orders",
                "🎉 Personalized birthday offers"
            ],
            "priority": "🔴 Highest Priority"
        },
        "Loyal": {
            "strategy": "Maintenance & Upselling",
            "description": "Consistent customers with stable spending",
            "actions": [
                "📊 Personalized product recommendations",
                "💰 Discounts on bulk purchases",
                "📱 Loyalty points system",
                "📧 Regular newsletters with updates",
                "🎯 Cross-sell complementary products"
            ],
            "priority": "🟡 High Priority"
        },
        "Potential": {
            "strategy": "Development & Engagement",
            "description": "Customers with potential to become Loyal",
            "actions": [
                "📚 Educational content about products",
                "🎁 Free samples of premium products",
                "💬 Personalized shopping support",
                "🔔 Repurchase reminder campaigns",
                "🎯 Targeted promotions"
            ],
            "priority": "🟢 Medium Priority"
        },
        "New": {
            "strategy": "Welcome & Orientation",
            "description": "New customers just starting to shop",
            "actions": [
                "👋 Welcome email series",
                "🎁 Discount on second purchase",
                "📋 Survey to understand preferences",
                "📞 Follow-up call for satisfaction",
                "🛒 Shopping guidance"
            ],
            "priority": "🟢 Medium Priority"
        },
        "Can't Lose Them": {
            "strategy": "Win-back & Recovery",
            "description": "High-value customers who haven’t returned recently",
            "actions": [
                "🆘 Urgent win-back campaign with offers",
                "📞 Direct contact to address issues",
                "🎁 Special return offer",
                "📧 Apology campaign if needed",
                "🔄 Reactivation series"
            ],
            "priority": "🔴 Highest Priority"
        },
        "Needs Attention": {
            "strategy": "Re-engagement & Churn Prevention",
            "description": "Customers with decreasing interaction",
            "actions": [
                "⏰ Time-limited offers",
                "💡 Product usage tips",
                "🎯 Discounts on past purchases",
                "📱 App notifications",
                "🤝 Customer feedback survey"
            ],
            "priority": "🟡 High Priority"
        },
        "Lost": {
            "strategy": "Cost-effective Recovery",
            "description": "Long-inactive customers with low value",
            "actions": [
                "📧 Low-cost email campaign with discounts",
                "🆕 Highlight new products",
                "🎁 One-time large discount",
                "📊 A/B test recovery methods",
                "⏸️ Pause campaigns if no response"
            ],
            "priority": "🔵 Low Priority"
        }
    }
    
    return recommendations.get(segment_label, {
        "strategy": "Unclassified",
        "description": "Customer not yet classified",
        "actions": ["Further analysis needed"],
        "priority": "🔵 Low Priority"
    })

def show_customer_recommendations(segment):
    """Display customer recommendations in a formatted way"""
    rec = get_customer_recommendations(segment)
    st.markdown(f"""
    <div class="recommendation-box">
        <h4>🎯 Strategy: {rec['strategy']}</h4>
        <p><strong>Description:</strong> {rec['description']}</p>
        <p><strong>Priority:</strong> {rec['priority']}</p>
        <h5>Recommended Actions:</h5>
        <ul>
            {"".join(f"<li>{action}</li>" for action in rec['actions'])}
        </ul>
    </div>
    """, unsafe_allow_html=True)

# Visualization Functions
def plot_distributions(df, cols, title):
    """Plot distributions of RFM metrics"""
    fig, axes = plt.subplots(len(cols), 1, figsize=FIGSIZE, dpi=DPI)
    axes = [axes] if len(cols) == 1 else axes
    for idx, col in enumerate(cols):
        sns.histplot(df[col], kde=True, ax=axes[idx], bins=20, edgecolor="black", color=sns.color_palette()[idx])
        skewness = skew(df[col].dropna())
        axes[idx].set_title(f'Distribution of "{col}" (skewness: {skewness:.2f})')
        axes[idx].set_xlabel(col.capitalize())
        axes[idx].set_ylabel("Count")
        axes[idx].grid(True, alpha=0.3)
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    return fig

def plot_segment_count(df, segment_col, order, colors, title):
    """Plot count of customers per segment"""
    fig, ax = plt.subplots(figsize=(FIGSIZE[0]/2, FIGSIZE[1]/2), dpi=DPI)
    sns.countplot(data=df, y=segment_col, order=order, hue=segment_col, palette=colors, edgecolor="black", ax=ax, legend=False)
    total = len(df)
    for p in ax.patches:
        count = int(p.get_width())
        ax.text(p.get_width() + 5, p.get_y() + p.get_height()/2, f"{count} ({count/total*100:.1f}%)", va="center")
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Count")
    ax.set_ylabel("Segment")
    plt.tight_layout()
    return fig

def plot_boxplots(df, segment_col, cols, order, colors, title):
    """Plot boxplots of RFM metrics by segment"""
    fig, axes = plt.subplots(1, len(cols), figsize=FIGSIZE, dpi=DPI)
    for i, col in enumerate(cols):
        sns.boxplot(x=segment_col, y=col, data=df, order=order, hue=segment_col, palette=colors, showfliers=True, linewidth=2, ax=axes[i], legend=False)
        axes[i].set_title(f"{col.capitalize()} by Segment")
        axes[i].tick_params(axis="x", rotation=25)
        axes[i].grid(True, linestyle="--", alpha=0.7)
    plt.suptitle(title, fontsize=14, y=1.05)
    plt.tight_layout()
    return fig

def plot_scatter_agg(df_agg, x_col, y_col, size_col, color_col, order, colors, title):
    """Plot scatter plot of segment aggregates"""
    fig = px.scatter(df_agg, x=x_col, y=y_col, size=size_col, color=color_col,
                     hover_name=color_col, text=color_col, size_max=80,
                     category_orders={color_col: order},
                     color_discrete_sequence=colors,
                     title=title)
    fig.update_layout(
        xaxis_title=x_col.replace('_', ' ').title(),
        yaxis_title=y_col.replace('_', ' ').title(),
        title_font_size=14,
        font_size=12
    )
    return fig

def plot_treemap_agg(df_agg, path_col, value_col, color_col, colors_dict, hover_data, title):
    """Plot treemap of segment aggregates"""
    fig = px.treemap(df_agg, path=[path_col], values=value_col, color=color_col,
                     color_discrete_map=colors_dict,
                     hover_data=hover_data)
    fig.update_traces(
        marker_line_width=1, marker_line_color="black",
        hovertemplate="<b>%{label}</b><br>Recency mean: %{customdata[0]:.0f}<br>Frequency mean: %{customdata[1]:.0f}<br>Monetary mean: %{customdata[2]:.0f}<br>Customers: %{customdata[3]:,}<br>Percent: %{customdata[4]:.1f}%<extra></extra>",
        texttemplate="<b>%{label}</b><br>%{customdata[0]:.0f} days<br>%{customdata[1]:.0f} orders<br>$ %{customdata[2]:.0f}<br>%{customdata[3]:,} customers (%{customdata[4]:.1f}%)"
    )
    fig.update_layout(title=title, margin=dict(t=50, l=0, r=0, b=0), width=700, height=700, font_size=12)
    return fig

def plot_heatmap(df, cols):
    """Plot correlation heatmap of RFM metrics"""
    fig, ax = plt.subplots(figsize=(8, 6), dpi=DPI)
    corr = df[cols].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', ax=ax, cbar=True)
    ax.set_title("Correlation Heatmap of RFM Metrics")
    plt.tight_layout()
    return fig

def plot_elbow_silhouette(K_range, sse, sil):
    """Plot elbow and silhouette scores for K-Means optimization"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=DPI)
    
    ax1.plot(K_range, sse, 'bx-')
    ax1.set_xlabel('k')
    ax1.set_ylabel('SSE')
    ax1.set_title('Elbow Method for Optimal k')
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(K_range, sil, 'ro-')
    ax2.set_xlabel('k')
    ax2.set_ylabel('Silhouette Score')
    ax2.set_title('Silhouette Score vs k')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

# Main Application
def main():
    # Initialize session state for storing processed data
    if 'df_clean' not in st.session_state:
        st.session_state.df_clean = None
    if 'report_from_dt' not in st.session_state:
        st.session_state.report_from_dt = None
    if 'report_to_dt' not in st.session_state:
        st.session_state.report_to_dt = None
    if 'date_col' not in st.session_state:
        st.session_state.date_col = None
    if 'invalid_amount' not in st.session_state:
        st.session_state.invalid_amount = None
    if 'invalid_date' not in st.session_state:
        st.session_state.invalid_date = None
    if 'invalid_name' not in st.session_state:
        st.session_state.invalid_name = None

    st.sidebar.title("🧭 Navigation")
    pages = [
        "🏠 Home - Business Problem",
        "⚡ Customer Segmentation & Recommendations",
        "🔧 Evaluations and Reports"
    ]
    
    selected_page = st.sidebar.selectbox("Select page:", pages)
    
    if selected_page == "🏠 Home - Business Problem":
        st.markdown('<h1 class="main-header">🏪 Customer Segmentation System - Store X</h1>',
                    unsafe_allow_html=True)
        
        st.markdown('<h2 class="section-header">1. Business Understanding</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.markdown("""
            <div class="insight-box">
            <h4>🎯 Business Problem</h4>
            <p><strong>Store X</strong> primarily sells essential products to customers such as:</p>
            <ul>
                <li>🥬 Fresh vegetables, fruits</li>
                <li>🥩 Meat, fish, seafood</li>
                <li>🥚 Eggs, dairy, dairy products</li>
                <li>🥤 Beverages</li>
                <li>🍞 Bread and other products</li>
            </ul>
            <p><strong>Customers:</strong> Retail customers</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="insight-box">
            <h4>🚀 Store Owner's Goals</h4>
            <ul>
                <li><strong>Increase Sales:</strong> Sell more goods</li>
                <li><strong>Targeted Marketing:</strong> Promote products to the right customer segments</li>
                <li><strong>Customer Care:</strong> Enhance satisfaction and retention</li>
                <li><strong>Optimize Profit:</strong> Improve marketing and sales efficiency</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="recommendation-box">
            <h4>💡 Proposed Solution</h4>
            <p><strong>Build a customer segmentation system</strong> based on RFM analysis to:</p>
            <ul>
                <li>📊 Identify different customer groups</li>
                <li>🎯 Develop tailored business strategies</li>
                <li>💝 Optimize customer care</li>
                <li>📈 Boost marketing and revenue effectiveness</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="insight-box">
            <h4>📈 What is RFM Analysis?</h4>
            <ul>
                <li><strong>R - Recency:</strong> How long since the last purchase?</li>
                <li><strong>F - Frequency:</strong> How often do customers buy?</li>
                <li><strong>M - Monetary:</strong> How much do customers spend?</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('<h3 class="subsection-header">🗺️ RFM Segmentation Map</h3>', unsafe_allow_html=True)
        rfm_map_fig = create_rfm_map_visual()
        st.pyplot(rfm_map_fig)
        
        st.markdown('<h2 class="section-header">💼 Expected Business Impact</h2>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
                <h3>📈 Revenue Increase</h3>
                <p>More precise marketing</p>
                <p>Higher conversion rates</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-card">
                <h3>🎯 Cost Efficiency</h3>
                <p>Reduced advertising costs</p>
                <p>Focus on high-value customers</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="metric-card">
                <h3>💝 Customer Satisfaction</h3>
                <p>Personalized service</p>
                <p>Meeting specific needs</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown("""
            <div class="metric-card">
                <h3>🔄 Customer Retention</h3>
                <p>Tailored care strategies</p>
                <p>Increased lifetime value</p>
            </div>
            """, unsafe_allow_html=True)
    
    elif selected_page == "⚡ Customer Segmentation & Recommendations":
        st.markdown('<h1 class="main-header">⚡ Customer Segmentation & Recommendations</h1>', unsafe_allow_html=True)
        
        tab1, tab2 = st.tabs(["Quick RFM Input (1-4)", "Full Analysis from Raw Files"])
        
        with tab1:
            st.markdown('<h2 class="section-header">1. Direct RFM Input (1-4 Range)</h2>', unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                r_value = st.number_input("Recency Score (1-4, 4=most recent)", min_value=1, max_value=4, value=4, step=1, key="r_quick")
            with col2:
                f_value = st.number_input("Frequency Score (1-4, 4=most frequent)", min_value=1, max_value=4, value=3, step=1, key="f_quick")
            with col3:
                m_value = st.number_input("Monetary Score (1-4, 4=highest spend)", min_value=1, max_value=4, value=3, step=1, key="m_quick")
            
            if st.button("Analyze Customer", key="analyze_quick"):
                df_single = create_single_customer_rfm(r_value, f_value, m_value)
                df_single_labeled = label_rfm_segments(df_single)
                
                segment = df_single_labeled['rfm_segment_labeled'].iloc[0]
                segment_count = 1
                st.markdown(f"""
                    <div class="insight-box">
                        <h4>📊 Customer Segment Result</h4>
                        <p><strong>Segment:</strong> {segment}</p>
                        <p><strong>RFM Score:</strong> {r_value}-{f_value}-{m_value}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with st.expander(f"🔍 {segment} - Strategy ({segment_count} customer)"):
                    show_customer_recommendations(segment)
        
        with tab2:
            st.markdown('<h2 class="section-header">2. Full Analysis from Raw Files</h2>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                products_file = st.file_uploader(
                    "📦 Products File",
                    type=["csv"],
                    key="products_full"
                )
            
            with col2:
                transactions_file = st.file_uploader(
                    "🛒 Transactions File",
                    type=["csv"],
                    key="transactions_full"
                )
            
            if transactions_file:
                df_transactions_raw = pd.read_csv(transactions_file)
                date_col = st.selectbox(
                    "Select the date column in Transactions file",
                    options=df_transactions_raw.columns.tolist(),
                    index=df_transactions_raw.columns.tolist().index('Date') if 'Date' in df_transactions_raw.columns else 0,
                    key="date_col_select"
                )
            
            if products_file and transactions_file and date_col:
                df_products_raw = pd.read_csv(products_file)
                
                st.sidebar.markdown("### 📅 Date Range")
                df_trans_temp = fn_date_format(df_transactions_raw, cols=[date_col])
                min_date = df_trans_temp[date_col].min() if date_col in df_trans_temp.columns else pd.to_datetime("2020-01-01")
                max_date = df_trans_temp[date_col].max() if date_col in df_trans_temp.columns else REPORT_DATE_FIXED
                if pd.isna(min_date) or pd.isna(max_date):
                    st.error("No valid dates found in the selected date column. Please check the data.")
                    return
                
                date_range = st.sidebar.date_input(
                    "Select report date range",
                    value=[min_date, max_date],  # Automatically set to min and max dates
                    min_value=min_date,
                    max_value=max_date,
                    key="date_range_full"
                )
                
                # Check if date_range is a tuple/list with two dates
                if not isinstance(date_range, (tuple, list)) or len(date_range) != 2:
                    st.error("Invalid date range selected. Please ensure both 'from' and 'to' dates are chosen.")
                    return
                
                report_from_dt, report_to_dt = date_range
                report_from_dt = pd.to_datetime(report_from_dt)
                report_to_dt = pd.to_datetime(report_to_dt)
                if report_from_dt > report_to_dt:
                    st.error("The 'from' date must be earlier than the 'to' date.")
                    return
                
                st.sidebar.markdown(f"**From:** {report_from_dt.date()}  **To:** {report_to_dt.date()}")
                st.sidebar.markdown(f"**Total Days:** {(report_to_dt - report_from_dt).days} days")
                
                df_clean, invalid_amount, invalid_date, invalid_name = clean_and_process_data(
                    df_products_raw, df_transactions_raw, report_from_dt, report_to_dt, date_col
                )
                
                # Store processed data in session state
                st.session_state.df_clean = df_clean
                st.session_state.report_from_dt = report_from_dt
                st.session_state.report_to_dt = report_to_dt
                st.session_state.date_col = date_col
                st.session_state.invalid_amount = invalid_amount
                st.session_state.invalid_date = invalid_date
                st.session_state.invalid_name = invalid_name
                
                df_rfm, max_transaction_date = create_rfm_table(df_clean, report_to_dt)
                
                df_rfm_scored = rfm_quartile_scoring(df_rfm)
                df_rfm_labeled = label_rfm_segments(df_rfm_scored)
                
                segment_order = ["Champions", "Loyal", "Potential", "New", "Can't Lose Them", "Needs Attention", "Lost"]
                segment_colors = [rfm_segment_rule_dict[seg]["color"] for seg in segment_order]
                
                rfm_agg = create_segment_analysis(df_rfm_labeled, "rfm_segment_labeled")
                rfm_agg = rfm_agg.reindex([segment_order.index(x) if x in segment_order else len(segment_order) for x in rfm_agg['rfm_segment_labeled']])
                
                st.markdown(f"""
                <div class="insight-box">
                    <h4>📊 RFM Segment Analysis</h4>
                    <p><strong>Number of Customers:</strong> {df_rfm_labeled['customer_id'].nunique()}</p>
                    <p><strong>Total Revenue:</strong> {df_rfm_labeled['monetary'].sum():,.2f} USD</p>
                </div>
                """, unsafe_allow_html=True)
                st.dataframe(rfm_agg)
                
                st.markdown('<h3 class="subsection-header">Visualizations</h3>', unsafe_allow_html=True)
                st.pyplot(plot_segment_count(df_rfm_labeled, "rfm_segment_labeled", segment_order, segment_colors, "Customer Distribution by RFM Segment"))
                st.plotly_chart(plot_scatter_agg(rfm_agg, "recency_mean", "monetary_mean", "frequency_mean", "rfm_segment_labeled", segment_order, segment_colors, "Customer Segmentation by RFM"), key="rfm_scatter_full")
                st.plotly_chart(plot_treemap_agg(rfm_agg, "rfm_segment_labeled", "customer_count", "rfm_segment_labeled", dict(zip(segment_order, segment_colors)),
                                                {"recency_mean": ":.0f", "frequency_mean": ":.0f", "monetary_mean": ":.0f", "customer_count": ":,", "customer_distribution": ":.1f"},
                                                "Customer Segmentation Distribution (Treemap)"), key="rfm_treemap_full")
                
                st.markdown('<h3 class="subsection-header">Recommendations</h3>', unsafe_allow_html=True)
                for segment in segment_order:
                    if segment in rfm_agg['rfm_segment_labeled'].values:
                        segment_count = rfm_agg[rfm_agg['rfm_segment_labeled'] == segment]["customer_count"].iloc[0]
                        with st.expander(f"🔍 {segment} - Strategy ({segment_count} customers)"):
                            show_customer_recommendations(segment)
    
    else:  # "🔧 Evaluations and Reports"
        st.markdown('<h1 class="main-header">🔧 Evaluations and Reports</h1>',
                    unsafe_allow_html=True)
        
        st.markdown("""
        <div class="warning-box">
        <h4>⚠️ Notice</h4>
        <p>This page is intended for those who inquire a deep understanding of the analysis process.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Check if data is available in session state
        if (st.session_state.df_clean is not None and
            st.session_state.report_from_dt is not None and
            st.session_state.report_to_dt is not None and
            st.session_state.date_col is not None):
            df_clean = st.session_state.df_clean
            report_from_dt = st.session_state.report_from_dt
            report_to_dt = st.session_state.report_to_dt
            invalid_amount = st.session_state.invalid_amount
            invalid_date = st.session_state.invalid_date
            invalid_name = st.session_state.invalid_name
            
            st.info("📊 Using preprocessed data from 'Customer Segmentation & Recommendations' page.")
        else:
            st.info("📁 Please upload data to view detailed technical analysis, or process data in 'Customer Segmentation & Recommendations' first.")
            
            col1, col2 = st.columns(2)
            
            with col1:
                products_file_tech = st.file_uploader(
                    "📦 Products File",
                    type=["csv"],
                    key="products_tech"
                )
            
            with col2:
                transactions_file_tech = st.file_uploader(
                    "🛒 Transactions File",
                    type=["csv"],
                    key="transactions_tech"
                )
            
            if transactions_file_tech:
                df_transactions_raw = pd.read_csv(transactions_file_tech)
                date_col = st.selectbox(
                    "Select the date column in Transactions file",
                    options=df_transactions_raw.columns.tolist(),
                    index=df_transactions_raw.columns.tolist().index('Date') if 'Date' in df_transactions_raw.columns else 0,
                    key="date_col_select_tech"
                )
            
            if products_file_tech and transactions_file_tech and date_col:
                df_products_raw = pd.read_csv(products_file_tech)
                
                st.sidebar.markdown("### 📅 Date Range")
                df_trans_temp = fn_date_format(df_transactions_raw, cols=[date_col])
                min_date = df_trans_temp[date_col].min() if date_col in df_trans_temp.columns else pd.to_datetime("2020-01-01")
                max_date = df_trans_temp[date_col].max() if date_col in df_trans_temp.columns else REPORT_DATE_FIXED
                if pd.isna(min_date) or pd.isna(max_date):
                    st.error("No valid dates found in the selected date column. Please check the data.")
                    return
                
                date_range = st.sidebar.date_input(
                    "Select report date range",
                    value=[min_date, max_date],  # Automatically set to min and max dates
                    min_value=min_date,
                    max_value=max_date,
                    key="date_range_tech"
                )
                
                # Check if date_range is a tuple/list with two dates
                if not isinstance(date_range, (tuple, list)) or len(date_range) != 2:
                    st.error("Invalid date range selected. Please ensure both 'from' and 'to' dates are chosen.")
                    return
                
                report_from_dt, report_to_dt = date_range
                report_from_dt = pd.to_datetime(report_from_dt)
                report_to_dt = pd.to_datetime(report_to_dt)
                if report_from_dt > report_to_dt:
                    st.error("The 'from' date must be earlier than the 'to' date.")
                    return
                
                df_clean, invalid_amount, invalid_date, invalid_name = clean_and_process_data(
                    df_products_raw, df_transactions_raw, report_from_dt, report_to_dt, date_col
                )
            else:
                return
        
        st.sidebar.markdown("### ⚙️ Technical Configuration")
        k_kmeans = st.sidebar.slider("Number of K-Means Clusters", 2, 10, 4, key="kmeans_clusters")
        
        st.sidebar.markdown(f"**From:** {report_from_dt.date()}  **To:** {report_to_dt.date()}")
        st.sidebar.markdown(f"**Total Days:** {(report_to_dt - report_from_dt).days} days")
        
        # Step 1: Data Preprocessing
        st.markdown('<h2 class="section-header">1. Data Preprocessing</h2>', unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="insight-box">
            <h4>📊 Processed Data Preview</h4>
        </div>
        """, unsafe_allow_html=True)
        st.dataframe(df_clean.head(N_ROWS))
        
        st.markdown(f"""
        <div class="warning-box">
            <h4>⚠️ Invalid Records Removed</h4>
            <p>Amount <= 0: {invalid_amount}</p>
            <p>Invalid Date: {invalid_date}</p>
            <p>Missing Product Name: {invalid_name}</p>
            <p>Total Removed: {invalid_amount + invalid_date + invalid_name}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Step 2: RFM Analysis
        st.markdown('<h2 class="section-header">2. RFM Analysis</h2>', unsafe_allow_html=True)
        
        # Create RFM table
        df_rfm, max_date = create_rfm_table(df_clean, report_to_dt)
        st.markdown(f"""
        <div class="insight-box">
            <h4>📊 RFM Table</h4>
            <p>Latest Transaction Date: {max_date}</p>
        </div>
        """, unsafe_allow_html=True)
        st.dataframe(df_rfm.head(N_ROWS))
        
        # RFM distributions
        rfm_cols = ["recency", "frequency", "monetary"]
        st.pyplot(plot_distributions(df_rfm, rfm_cols, "RFM Distributions Original"))
        
        # Correlation heatmap
        st.pyplot(plot_heatmap(df_rfm, rfm_cols))
        
        # Pairplot
        st.pyplot(sns.pairplot(df_rfm[rfm_cols], diag_kind="kde", plot_kws={"edgecolor": "black", "alpha": 0.7}).fig)
        
        # RFM quartile scoring
        df_rfm_scored = rfm_quartile_scoring(df_rfm)
        st.markdown(f"""
        <div class="insight-box">
            <h4>📊 RFM Scored</h4>
        </div>
        """, unsafe_allow_html=True)
        st.dataframe(df_rfm_scored.head(N_ROWS))
        
        # RFM labeling
        df_rfm_labeled = label_rfm_segments(df_rfm_scored)
        segment_order = ["Champions", "Loyal", "Potential", "New", "Can't Lose Them", "Needs Attention", "Lost"]
        segment_colors = [rfm_segment_rule_dict[seg]["color"] for seg in segment_order]
        st.markdown(f"""
        <div class="insight-box">
            <h4>📊 RFM Labeled</h4>
        </div>
        """, unsafe_allow_html=True)
        st.dataframe(df_rfm_labeled.head(N_ROWS))
        
        # Segment count plot
        st.pyplot(plot_segment_count(df_rfm_labeled, "rfm_segment_labeled", segment_order, segment_colors, "Customer Distribution by Labeled RFM Segment"))
        
        # Boxplots
        st.pyplot(plot_boxplots(df_rfm_labeled, "rfm_segment_labeled", rfm_cols, segment_order, segment_colors, "Distribution of RFM Metrics by Customer Segment"))
        
        # RFM aggregate
        rfm_agg = create_segment_analysis(df_rfm_labeled, "rfm_segment_labeled")
        rfm_agg = rfm_agg.reindex([segment_order.index(x) if x in segment_order else len(segment_order) for x in rfm_agg['rfm_segment_labeled']])
        st.markdown(f"""
        <div class="insight-box">
            <h4>📊 RFM Segment Analysis</h4>
        </div>
        """, unsafe_allow_html=True)
        st.dataframe(rfm_agg)
        
        # RFM scatter and treemap
        st.plotly_chart(plot_scatter_agg(rfm_agg, "recency_mean", "monetary_mean", "frequency_mean", "rfm_segment_labeled", segment_order, segment_colors, "Customer Segmentation by RFM"), key="rfm_scatter_tech")
        st.plotly_chart(plot_treemap_agg(rfm_agg, "rfm_segment_labeled", "customer_count", "rfm_segment_labeled", dict(zip(segment_order, segment_colors)),
                                        {"recency_mean": ":.0f", "frequency_mean": ":.0f", "monetary_mean": ":.0f", "customer_count": ":,", "customer_distribution": ":.1f"},
                                        "Customer Segmentation Distribution (Treemap)"), key="rfm_treemap_tech")
        
        # Step 3: Feature Engineering
        st.markdown('<h2 class="section-header">3. Feature Engineering</h2>', unsafe_allow_html=True)
        df_eng, scaler = feature_engineering_rfm(df_rfm)
        rfm_cols_scaled = ["recency_log_scaled", "frequency_scaled", "monetary_log_scaled"]
        st.pyplot(plot_distributions(df_eng, rfm_cols_scaled, "Distributions of RFM Features after Scaling"))
        
        # Step 4: K-Means Clustering
        st.markdown('<h2 class="section-header">4. K-Means Clustering</h2>', unsafe_allow_html=True)
        
        # K-Means optimization
        df_scaled = df_eng[rfm_cols_scaled].copy()
        sse, sil = [], []
        K_range = range(2, 11)
        for k in K_range:
            km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10, max_iter=300, init='k-means++')
            lbl = km.fit_predict(df_scaled)
            sse.append(km.inertia_)
            sil.append(silhouette_score(df_scaled, lbl))
        st.pyplot(plot_elbow_silhouette(K_range, sse, sil))
        
        # K-Means with optimal k
        clusters, kmeans_model = kmeans_clustering(df_scaled, n_clusters=k_kmeans)
        df_eng['cluster'] = clusters
        
        st.pyplot(plot_segment_count(df_eng, "cluster", sorted(df_eng['cluster'].unique()), sns.color_palette("Set2", k_kmeans).as_hex(), "Customer Distribution by K-Means Cluster"))
        st.pyplot(plot_boxplots(df_eng, "cluster", rfm_cols, sorted(df_eng['cluster'].unique()), sns.color_palette("Set2", k_kmeans).as_hex(), "Distribution of RFM Metrics by K-Means Cluster"))
        
        # K-Means aggregate
        kmeans_agg = create_segment_analysis(df_eng, "cluster")
        st.markdown(f"""
        <div class="insight-box">
            <h4>📊 K-Means Segment Analysis</h4>
        </div>
        """, unsafe_allow_html=True)
        st.dataframe(kmeans_agg)
        
        # K-Means scatter and treemap
        st.plotly_chart(plot_scatter_agg(kmeans_agg, "recency_mean", "monetary_mean", "frequency_mean", "cluster", kmeans_agg["cluster"], sns.color_palette("Set2", kmeans_agg["cluster"].nunique()).as_hex(), "Customer Segmentation by K-Means"), key="kmeans_scatter_tech")
        st.plotly_chart(plot_treemap_agg(kmeans_agg, "cluster", "customer_count", "cluster", dict(zip(kmeans_agg["cluster"], sns.color_palette("Set2", kmeans_agg["cluster"].nunique()).as_hex())),
                                        {"recency_mean": ":.0f", "frequency_mean": ":.0f", "monetary_mean": ":.0f", "customer_count": ":,", "customer_distribution": ":.1f"},
                                        "Customer Segmentation Distribution (Treemap)"), key="kmeans_treemap_tech")
        
        # K-Means cluster labeling
        cluster_no = ["2", "1", "0", "3"]
        cluster_order = ["Loyal", "Regular", "At-Risk", "Lost"]
        cluster_colors = ["#66E0E0", "#6666FF", "#B0C4DE", "#FFB266"]
        df_eng['cluster_label'] = df_eng['cluster'].map(dict(zip(cluster_no, cluster_order)))
        kmeans_agg['cluster_label'] = kmeans_agg['cluster'].map(dict(zip(cluster_no, cluster_order)))
        
        st.markdown(f"""
        <div class="insight-box">
            <h4>📊 Labeled K-Means Segment Analysis</h4>
        </div>
        """, unsafe_allow_html=True)
        st.dataframe(kmeans_agg)
        
        st.plotly_chart(plot_scatter_agg(kmeans_agg, "recency_mean", "monetary_mean", "frequency_mean", "cluster_label", cluster_order, cluster_colors, "Customer Segmentation by Labeled K-Means"), key="labeled_kmeans_scatter_tech")
        st.plotly_chart(plot_treemap_agg(kmeans_agg, "cluster_label", "customer_count", "cluster_label", dict(zip(cluster_order, cluster_colors)),
                                        {"recency_mean": ":.0f", "frequency_mean": ":.0f", "monetary_mean": ":.0f", "customer_count": ":,", "customer_distribution": ":.1f"},
                                        "Customer Segmentation Distribution (Treemap)"), key="labeled_kmeans_treemap_tech")
        st.pyplot(plot_boxplots(df_eng, "cluster_label", rfm_cols, cluster_order, dict(zip(cluster_order, cluster_colors)), "Distribution of RFM Metrics by Labeled K-Means Cluster"))
        
        # Step 5: Evaluation & Comparison
        st.markdown('<h2 class="section-header">5. Evaluation & Comparison</h2>', unsafe_allow_html=True)
        sil_rule = silhouette_score(df_eng[['recency_log_scaled', 'frequency_scaled', 'monetary_log_scaled']], df_rfm_labeled['rfm_segment_labeled'].factorize()[0]) if len(set(df_rfm_labeled['rfm_segment_labeled'])) > 1 else 0
        sil_kmeans = silhouette_score(df_eng[['recency_log_scaled', 'frequency_scaled', 'monetary_log_scaled']], df_eng['cluster'].astype(int))
        comparison = pd.DataFrame({
            "Method": ["RFM Rule-based", "RFM + K-Means"],
            "Silhouette Score": [sil_rule, sil_kmeans]
        })
        st.markdown(f"""
        <div class="insight-box">
            <h4>📊 Model Comparison</h4>
            <p>- <strong>RFM Rule-based:</strong> Segments based on predefined rules, easy to interpret but potentially subjective.</p>
            <p>- <strong>RFM + K-Means:</strong> Optimizes clusters based on distance, with silhouette score indicating quality at k={k_kmeans}.</p>
        </div>
        """, unsafe_allow_html=True)
        st.dataframe(comparison)
        st.markdown("**Conclusion**: The model with the highest silhouette score is selected for deployment.")

if __name__ == "__main__":
    main()