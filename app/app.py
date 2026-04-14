import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from src.utils import load_data

st.set_page_config(
    page_title="Credit Card Churn Dashboard",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# Styling
# -----------------------------
st.markdown("""
<style>
    .main {
        background: linear-gradient(180deg, #f8fbff 0%, #ffffff 100%);
    }
    .hero {
        padding: 1.8rem 1.6rem;
        border-radius: 24px;
        background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 50%, #2563eb 100%);
        color: white;
        box-shadow: 0 12px 30px rgba(15, 23, 42, 0.18);
        margin-bottom: 1.5rem;
    }
    .hero h1 {
        margin: 0;
        font-size: 2.3rem;
        font-weight: 800;
        letter-spacing: -0.03em;
    }
    .hero p {
        margin-top: 0.5rem;
        font-size: 1rem;
        opacity: 0.92;
    }
    .metric-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 20px;
        padding: 1rem 1.2rem;
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
    }
    .section-title {
        font-size: 1.2rem;
        font-weight: 700;
        color: #0f172a;
        margin: 0.2rem 0 0.8rem 0;
    }
    .small-note {
        color: #64748b;
        font-size: 0.92rem;
    }
    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #111827 100%);
    }
    div[data-testid="stSidebar"] * {
        color: white !important;
    }
    .sidebar-title {
        font-size: 1.25rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------
# Helpers
# -----------------------------
@st.cache_data
def load_churn_data():
    df = load_data("Credit_Card_Churn.csv")
    df.columns = [c.strip().replace(" ", "_") for c in df.columns]

    # Create churn target if needed
    if "Attrition_Flag" in df.columns and "churned" not in df.columns:
        df["churned"] = df["Attrition_Flag"].map(
            {"Attrited Customer": 1, "Existing Customer": 0}
        )

    return df


def safe_col(df, col):
    return col in df.columns


def kpi_value(series, is_pct=False, decimals=1):
    if is_pct:
        return f"{series * 100:.{decimals}f}%"
    if pd.api.types.is_numeric_dtype(series):
        return f"{series:.{decimals}f}"
    return str(series)


def fig_churn_donut(df):
    if "churned" not in df.columns:
        return None

    churn_counts = df["churned"].value_counts().sort_index()
    labels = ["Retained", "Churned"]
    values = [churn_counts.get(0, 0), churn_counts.get(1, 0)]

    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                hole=0.68,
                textinfo="label+percent",
                marker=dict(colors=["#3b82f6", "#ef4444"]),
            )
        ]
    )
    fig.update_layout(
        margin=dict(l=10, r=10, t=20, b=10),
        showlegend=False,
        height=320,
        paper_bgcolor="white",
    )
    return fig


def fig_age_churn(df):
    if not safe_col(df, "Customer_Age") or "churned" not in df.columns:
        return None

    temp = df.copy()
    temp["Age_Group"] = pd.cut(
        temp["Customer_Age"],
        bins=[0, 30, 40, 50, 60, 100],
        labels=["18-30", "31-40", "41-50", "51-60", "60+"]
    )
    grp = temp.groupby("Age_Group", observed=False)["churned"].mean().reset_index()

    fig = px.bar(
        grp,
        x="Age_Group",
        y="churned",
        text=grp["churned"].map(lambda x: f"{x*100:.1f}%"),
        title="Churn Rate by Age Group",
        labels={"churned": "Churn Rate", "Age_Group": "Age Group"},
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        yaxis_tickformat=".0%",
        height=360,
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


def fig_income_churn(df):
    if not safe_col(df, "Income_Category") or "churned" not in df.columns:
        return None

    order = [
        "Less than $40K",
        "$40K - $60K",
        "$60K - $80K",
        "$80K - $120K",
        "$120K +",
        "Unknown"
    ]
    temp = df.copy()
    grp = temp.groupby("Income_Category", observed=False)["churned"].mean().reindex(order).reset_index()

    fig = px.line(
        grp,
        x="Income_Category",
        y="churned",
        markers=True,
        title="Churn Rate by Income Category",
        labels={"churned": "Churn Rate", "Income_Category": "Income Category"},
    )
    fig.update_traces(line=dict(width=4), marker=dict(size=10))
    fig.update_layout(
        yaxis_tickformat=".0%",
        height=360,
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


def fig_numeric_distributions(df, num_col):
    if not safe_col(df, num_col):
        return None

    fig = px.histogram(
        df,
        x=num_col,
        nbins=30,
        color="churned" if "churned" in df.columns else None,
        marginal="box",
        title=f"Distribution of {num_col}",
        opacity=0.8,
    )
    fig.update_layout(
        height=360,
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


def fig_corr_heatmap(df):
    numeric_df = df.select_dtypes(include="number")
    if numeric_df.shape[1] < 2:
        return None

    corr = numeric_df.corr(numeric_only=True)
    fig = px.imshow(
        corr,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="Blues",
        title="Correlation Heatmap"
    )
    fig.update_layout(
        height=650,
        paper_bgcolor="white",
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


def fig_churn_by_category(df, col):
    if not safe_col(df, col) or "churned" not in df.columns:
        return None

    grp = df.groupby(col, observed=False)["churned"].mean().reset_index().sort_values("churned", ascending=False)

    fig = px.bar(
        grp,
        x=col,
        y="churned",
        title=f"Churn Rate by {col.replace('_', ' ')}",
        text=grp["churned"].map(lambda x: f"{x*100:.1f}%")
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        yaxis_tickformat=".0%",
        height=380,
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


# -----------------------------
# Load data
# -----------------------------
df = load_churn_data()

if "churned" not in df.columns:
    st.error("Could not find target column. Check your CSV structure and Attrition_Flag values.")
    st.stop()

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.markdown('<div class="sidebar-title">💳 Churn Control Panel</div>', unsafe_allow_html=True)
    st.markdown("Use filters to explore the dataset like a real analyst.")

    churn_filter = st.selectbox(
        "Churn Status",
        options=["All", "Retained", "Churned"]
    )

    gender_filter = None
    if safe_col(df, "Gender"):
        gender_filter = st.multiselect(
            "Gender",
            options=sorted(df["Gender"].dropna().astype(str).unique().tolist()),
            default=sorted(df["Gender"].dropna().astype(str).unique().tolist())
        )

    income_filter = None
    if safe_col(df, "Income_Category"):
        income_filter = st.multiselect(
            "Income Category",
            options=df["Income_Category"].dropna().astype(str).unique().tolist(),
            default=df["Income_Category"].dropna().astype(str).unique().tolist()
        )

    min_age, max_age = 0, 100
    if safe_col(df, "Customer_Age"):
        min_age, max_age = int(df["Customer_Age"].min()), int(df["Customer_Age"].max())
        age_range = st.slider(
            "Age Range",
            min_value=min_age,
            max_value=max_age,
            value=(min_age, max_age)
        )
    else:
        age_range = None

    st.markdown("---")
    st.markdown("### Quick Summary")
    st.write(f"Rows: **{len(df):,}**")
    st.write(f"Columns: **{df.shape[1]}**")

# -----------------------------
# Apply filters
# -----------------------------
filtered = df.copy()

if churn_filter != "All":
    if churn_filter == "Retained":
        filtered = filtered[filtered["churned"] == 0]
    else:
        filtered = filtered[filtered["churned"] == 1]

if gender_filter and safe_col(filtered, "Gender"):
    filtered = filtered[filtered["Gender"].astype(str).isin(gender_filter)]

if income_filter and safe_col(filtered, "Income_Category"):
    filtered = filtered[filtered["Income_Category"].astype(str).isin(income_filter)]

if age_range and safe_col(filtered, "Customer_Age"):
    filtered = filtered[
        (filtered["Customer_Age"] >= age_range[0]) &
        (filtered["Customer_Age"] <= age_range[1])
    ]

# -----------------------------
# Hero section
# -----------------------------
st.markdown("""
<div class="hero">
    <h1>Credit Card Churn Dashboard</h1>
    <p>Explore churn patterns, customer segments, and key business signals through a polished interactive interface.</p>
</div>
""", unsafe_allow_html=True)

# -----------------------------
# KPI cards
# -----------------------------
c1, c2, c3, c4 = st.columns(4)

with c1:
    churn_rate = filtered["churned"].mean()
    st.markdown(f"""
    <div class="metric-card">
        <div class="small-note">Churn Rate</div>
        <h2>{churn_rate:.1%}</h2>
    </div>
    """, unsafe_allow_html=True)

with c2:
    avg_age = filtered["Customer_Age"].mean() if safe_col(filtered, "Customer_Age") else None

    if avg_age is not None:
        avg_age_display = f"{avg_age:.1f}"
    else:
        avg_age_display = "N/A"

    st.markdown(f"""
    <div class="metric-card">
        <div class="small-note">Average Age</div>
        <h2>{avg_age_display}</h2>
    </div>
    """, unsafe_allow_html=True)

with c3:
    avg_limit = filtered["Credit_Limit"].mean() if safe_col(filtered, "Credit_Limit") else None
    if avg_limit is not None:
        avg_limit_display = f"{avg_limit:,.0f}"
    else:
        avg_limit_display = "N/A"
    st.markdown(f"""
    <div class="metric-card">
        <div class="small-note">Average Credit Limit</div>
        <h2>{avg_limit_display}</h2>
    </div>
    """, unsafe_allow_html=True)

with c4:
    total_customers = len(filtered)
    st.markdown(f"""
    <div class="metric-card">
        <div class="small-note">Customers Shown</div>
        <h2>{total_customers:,}</h2>
    </div>
    """, unsafe_allow_html=True)

st.write("")

# -----------------------------
# Tabs
# -----------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "Overview",
    "Customer Segments",
    "Feature Distributions",
    "Data Preview"
])

with tab1:
    left, right = st.columns([1, 1])

    with left:
        st.markdown('<div class="section-title">Churn Split</div>', unsafe_allow_html=True)
        fig = fig_churn_donut(filtered)
        if fig:
            st.plotly_chart(fig, use_container_width=True)

    with right:
        st.markdown('<div class="section-title">Key Notes</div>', unsafe_allow_html=True)
        st.info(
            "Use the sidebar filters to see how churn changes by age, gender, and income group. "
            "This makes the project look interactive and business-ready."
        )

        if "churned" in filtered.columns:
            churn_by_gender = None
            if safe_col(filtered, "Gender"):
                churn_by_gender = filtered.groupby("Gender")["churned"].mean().sort_values(ascending=False)
            if churn_by_gender is not None and len(churn_by_gender) > 0:
                st.markdown("**Churn by Gender**")
                st.dataframe((churn_by_gender * 100).round(1).rename("Churn %"))

        st.markdown("**Suggested business insight:**")
        st.success("Focus retention campaigns on segments with high churn rates and lower engagement.")

    st.markdown("")
    st.markdown('<div class="section-title">Correlation View</div>', unsafe_allow_html=True)
    corr_fig = fig_corr_heatmap(filtered)
    if corr_fig:
        st.plotly_chart(corr_fig, use_container_width=True)

with tab2:
    c1, c2 = st.columns(2)

    with c1:
        fig = fig_age_churn(filtered)
        if fig:
            st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig = fig_income_churn(filtered)
        if fig:
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("")
    st.markdown('<div class="section-title">Categorical Churn Analysis</div>', unsafe_allow_html=True)

    cat_cols = []
    for col in ["Gender", "Education_Level", "Marital_Status", "Card_Category"]:
        if safe_col(filtered, col):
            cat_cols.append(col)

    if cat_cols:
        selected_cat = st.selectbox("Choose a category to inspect", cat_cols)
        fig = fig_churn_by_category(filtered, selected_cat)
        if fig:
            st.plotly_chart(fig, use_container_width=True)

with tab3:
    numeric_candidates = [
        "Customer_Age",
        "Dependent_count",
        "Months_on_book",
        "Total_Relationship_Count",
        "Months_Inactive_12_mon",
        "Contacts_Count_12_mon",
        "Credit_Limit",
        "Total_Revolving_Bal",
        "Avg_Open_To_Buy",
        "Total_Amt_Chng_Q4_Q1",
        "Total_Trans_Amt",
        "Total_Trans_Ct",
        "Total_Ct_Chng_Q4_Q1",
        "Avg_Utilization_Ratio"
    ]
    available_numeric = [c for c in numeric_candidates if safe_col(filtered, c)]

    if available_numeric:
        chosen_num = st.selectbox("Choose a numeric feature", available_numeric)
        fig = fig_numeric_distributions(filtered, chosen_num)
        if fig:
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Numeric Summary")
    if available_numeric:
        summary = filtered[available_numeric].describe().T[["mean", "std", "min", "max"]]
        st.dataframe(summary.round(2), use_container_width=True)

with tab4:
    st.markdown('<div class="section-title">Filtered Data Preview</div>', unsafe_allow_html=True)
    st.dataframe(filtered.head(50), use_container_width=True)

    csv = filtered.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Filtered Data",
        data=csv,
        file_name="filtered_credit_card_churn.csv",
        mime="text/csv"
    )

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.markdown(
    "<div class='small-note'>Built for a portfolio: clean visuals, interactive filters, and business-focused insights.</div>",
    unsafe_allow_html=True
)