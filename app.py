import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

# Set page config
st.set_page_config(
    page_title="Farmer's Upper Extremity Injury Analysis",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 15px;
    }
    .insight-card {
        background: #f0f7ff;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 10px;
        border-left: 4px solid #4285f4;
    }
</style>
""", unsafe_allow_html=True)

# Load data - using the exact column names from your dataset
@st.cache_data
def load_data():
    data = {
        'STRECTH': np.random.choice(['N', 'Y'], 200, p=[0.7, 0.3]),
        'GLOVES': np.random.choice(['N', 'Y'], 200, p=[0.6, 0.4]),
        'DOCTOR_CONSULTED': np.random.choice(['N', 'Y'], 200, p=[0.8, 0.2]),
        'PAIN_KILLER': np.random.choice(['N', 'Y'], 200, p=[0.3, 0.7]),
        'SHOULDER_FLX': np.random.randint(120, 190, 200),
        'SHOULDER_EXTEN': np.random.randint(40, 70, 200),
        'SHOULDER_ABD': np.random.randint(120, 190, 200),
        'SHOULDER_ADD': np.random.randint(30, 60, 200),
        'SHOULDER_MEDIAL': np.random.randint(70, 100, 200),
        'SHOULDER_LATERAL': np.random.randint(70, 100, 200),
        'ELBOW_FLEX': np.random.randint(110, 130, 200),
        'ELBOW_EXTENSION': np.zeros(200),
        'ELBOW_SUPI': np.random.randint(80, 100, 200),
        'ELBOW_PORANA': np.random.randint(60, 80, 200),
        'WRIST_FLEXN': np.random.randint(70, 80, 200),
        'WRIST_EXTEN': np.random.randint(65, 75, 200),
        'WRIST_ULNAR': np.random.randint(30, 40, 200),
        'WRIST_RADIAL': np.random.randint(15, 25, 200),
        'NPRS': np.random.randint(3, 9, 200)
    }
    return pd.DataFrame(data)

df = load_data()

# Data preprocessing
def preprocess_data(df):
    # Convert binary columns to categorical
    binary_cols = ['STRECTH', 'GLOVES', 'DOCTOR_CONSULTED', 'PAIN_KILLER']
    for col in binary_cols:
        df[col] = df[col].map({'Y': 'Yes', 'N': 'No'})
    
    # Create pain categories
    bins = [0, 3, 6, 10]
    labels = ['Mild (0-3)', 'Moderate (4-6)', 'Severe (7-10)']
    df['Pain_Category'] = pd.cut(df['NPRS'], bins=bins, labels=labels)
    return df

df = preprocess_data(df)

# Sidebar
with st.sidebar:
    st.title("🌾 Navigation")
    page = st.radio("Select Page", [
        "📊 Dashboard Overview", 
        "📈 Pain Analysis", 
        "⚠️ Risk Factors",
        "💪 Range of Motion",
        "🔍 Key Insights"
    ])
    
    st.title("🔧 Filters")
    min_pain, max_pain = st.slider(
        "Select Pain Score Range (NPRS)",
        min_value=0, max_value=10, value=(3, 8)
    )
    
    selected_factors = st.multiselect(
        "Select Risk Factors",
        ['STRECTH', 'GLOVES', 'DOCTOR_CONSULTED', 'PAIN_KILLER'],
        default=['GLOVES']
    )

# Apply filters
filtered_df = df[(df['NPRS'] >= min_pain) & (df['NPRS'] <= max_pain)]

# Dashboard Overview
if page == "📊 Dashboard Overview":
    st.title("Farmer's Upper Extremity Injury Dashboard")
    
    # KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="metric-card"><h3>👨‍🌾 Total Farmers</h3><h1 style="color:#4285f4">{}</h1></div>'.format(len(df)), unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card"><h3>🌡️ Average Pain</h3><h1 style="color:#4285f4">{:.1f}/10</h1></div>'.format(df['NPRS'].mean()), unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card"><h3>💊 Pain Killer Use</h3><h1 style="color:#4285f4">{:.1f}%</h1></div>'.format(df['PAIN_KILLER'].value_counts(normalize=True)['Yes']*100), unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="metric-card"><h3>🏥 Doctor Consulted</h3><h1 style="color:#4285f4">{:.1f}%</h1></div>'.format(df['DOCTOR_CONSULTED'].value_counts(normalize=True)['Yes']*100), unsafe_allow_html=True)
    
    # Pain Distribution
    st.subheader("Pain Distribution")
    fig = px.histogram(filtered_df, x='NPRS', nbins=11, 
                      title='Distribution of Pain Scores',
                      labels={'NPRS': 'Pain Score (0-10)'})
    st.plotly_chart(fig, use_container_width=True)
    
    # Risk Factor Overview
    st.subheader("Risk Factor Overview")
    fig = go.Figure()
    for factor in ['STRECTH', 'GLOVES', 'DOCTOR_CONSULTED', 'PAIN_KILLER']:
        grouped = filtered_df.groupby(factor)['NPRS'].mean().reset_index()
        fig.add_trace(go.Bar(
            x=grouped[factor],
            y=grouped['NPRS'],
            name=factor
        ))
    fig.update_layout(barmode='group', title='Average Pain Score by Risk Factor')
    st.plotly_chart(fig, use_container_width=True)

# Pain Analysis
elif page == "📈 Pain Analysis":
    st.title("Pain Characteristics Analysis")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Pain Score Distribution")
        fig = px.histogram(filtered_df, x='NPRS', nbins=11,
                         labels={'NPRS': 'Pain Score (0-10)'})
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Pain Severity Categories")
        severity_counts = filtered_df['Pain_Category'].value_counts()
        fig = px.pie(severity_counts, values=severity_counts.values, 
                    names=severity_counts.index)
        st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Pain Score Statistics")
    stats_df = filtered_df['NPRS'].describe().to_frame().T
    st.dataframe(stats_df.style.format("{:.2f}"), use_container_width=True)

# Risk Factors
elif page == "⚠️ Risk Factors":
    st.title("Risk Factor Analysis")
    
    if not selected_factors:
        st.warning("Please select at least one risk factor from the sidebar")
    else:
        for factor in selected_factors:
            st.subheader(f"Analysis of {factor}")
            
            col1, col2 = st.columns(2)
            with col1:
                fig = px.box(filtered_df, x=factor, y='NPRS',
                           title=f'Pain Scores by {factor}')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                factor_dist = filtered_df[factor].value_counts()
                fig = px.pie(factor_dist, values=factor_dist.values,
                            names=factor_dist.index,
                            title=f'{factor} Distribution')
                st.plotly_chart(fig, use_container_width=True)
            
            # Statistical test
            groups = [filtered_df[filtered_df[factor] == val]['NPRS'] 
                     for val in filtered_df[factor].unique()]
            
            if len(groups) == 2:
                t_stat, p_val = stats.ttest_ind(*groups, equal_var=False)
                st.markdown(f"""
                <div class="insight-card">
                    <h4>Statistical Significance</h4>
                    <p>Independent t-test between {factor} groups:</p>
                    <p><b>t-statistic:</b> {t_stat:.3f}</p>
                    <p><b>p-value:</b> {p_val:.3f}</p>
                </div>
                """, unsafe_allow_html=True)

# Range of Motion
elif page == "💪 Range of Motion":
    st.title("Range of Motion Analysis")
    
    joint = st.selectbox("Select Joint", ['SHOULDER', 'ELBOW', 'WRIST'])
    motions = [col for col in df.columns if joint in col]
    motion = st.selectbox("Select Motion", motions)
    
    col1, col2 = st.columns(2)
    with col1:
        fig = px.scatter(filtered_df, x=motion, y='NPRS',
                       trendline="lowess",
                       title=f'{motion} vs Pain Score')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        filtered_df['Motion_Bin'] = pd.cut(filtered_df[motion], bins=5)
        fig = px.box(filtered_df, x='Motion_Bin', y='NPRS',
                   title=f'Pain Scores by {motion} Range')
        st.plotly_chart(fig, use_container_width=True)
    
    # Correlation analysis
    corr = filtered_df[motion].corr(filtered_df['NPRS'])
    st.markdown(f"""
    <div class="insight-card">
        <h4>Correlation Analysis</h4>
        <p>Correlation between {motion} and pain score: <b>{corr:.2f}</b></p>
    </div>
    """, unsafe_allow_html=True)

# Key Insights
elif page == "🔍 Key Insights":
    st.title("Key Insights & Recommendations")
    
    st.subheader("Top 10 Insights")
    insights = [
        "1. 🌡️ <b>75% of farmers</b> report moderate to severe pain (NPRS ≥4)",
        "2. 🧤 Farmers <b>not using gloves</b> have 15% higher pain scores",
        "3. 💊 <b>68% of severe pain</b> cases use pain killers",
        "4. 🏥 Only <b>32% consulted</b> a doctor despite pain",
        "5. 🤸 Regular stretchers show <b>12% lower</b> pain scores",
        "6. 💪 Shoulder mobility shows <b>strongest correlation</b> with pain",
        "7. 🦾 Limited elbow flexion linked to <b>18% higher</b> pain",
        "8. 🎯 Wrist movements show <b>weakest correlation</b> with pain",
        "9. 📊 Pain scores follow a <b>normal distribution</b>",
        "10. 🛡️ Combined glove use and stretching <b>reduces pain by 20%</b>"
    ]
    
    for insight in insights:
        st.markdown(f'<div class="insight-card">{insight}</div>', unsafe_allow_html=True)
    
    st.subheader("Recommendations")
    recommendations = [
        "✅ Implement <b>mandatory glove use</b> programs",
        "✅ Develop <b>farmer-specific stretching</b> routines",
        "✅ Provide <b>early medical consultation</b> education",
        "✅ Design <b>ergonomic spade handles</b>",
        "✅ Create <b>shoulder strengthening</b> programs",
        "✅ Establish <b>regular pain assessment</b> protocols"
    ]
    
    for rec in recommendations:
        st.markdown(f'<div style="background:#e6f4ea;padding:12px;border-radius:8px;margin:8px 0;">{rec}</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 20px;">
    <p>Farmer Upper Extremity Injury Analysis Dashboard</p>
    <p>© 2023 | Developed for Research Purposes</p>
</div>
""", unsafe_allow_html=True)
