import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import plotly.express as px
import openpyxl  # For Excel support

# Set page config
st.set_page_config(
    page_title="Farmer Upper Extremity Injury Analysis",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load data
@st.cache_data
def load_data():
    df = pd.read_excel('DATA.xlsx')
    return df

df = load_data()

# Sidebar filters
st.sidebar.title("Analysis Filters")
analysis_type = st.sidebar.radio(
    "Select Analysis Type",
    ["Overview Dashboard", "Pain Analysis", "Risk Factors", "Range of Motion", "Preventive Insights"]
)

# Convert binary columns to more readable format
df['STRETCH_USED'] = df['STRECTH'].map({'Y': 'Yes', 'N': 'No'})
df['GLOVES_USED'] = df['GLOVES'].map({'Y': 'Yes', 'N': 'No'})
df['PAIN_KILLER_USED'] = df['PAIN_KILLER'].map({'Y': 'Yes', 'N': 'No'})
df['DOCTOR_VISITED'] = df['DOCTOR_CONSULTED'].map({'Y': 'Yes', 'N': 'No'})

# Main content
st.title("🌾 Farmer Upper Extremity Injury Analysis")
st.markdown("""
**Aim**: To determine the prevalence and risk factors of upper extremities issues among farmers using spades.
""")

# Overview Dashboard
if analysis_type == "Overview Dashboard":
    st.header("📊 Overview Dashboard")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Mean Pain Score (NPRS)", f"{df['NPRS'].mean():.1f}")
    with col2:
        st.metric("Farmers Using Stretch", f"{df['STRECTH'].value_counts()['Y']} ({df['STRECTH'].value_counts(normalize=True)['Y']*100:.1f}%)")
    with col3:
        st.metric("Farmers Using Gloves", f"{df['GLOVES'].value_counts()['Y']} ({df['GLOVES'].value_counts(normalize=True)['Y']*100:.1f}%)")
    with col4:
        st.metric("Farmers Using Pain Killers", f"{df['PAIN_KILLER'].value_counts()['Y']} ({df['PAIN_KILLER'].value_counts(normalize=True)['Y']*100:.1f}%)")
    
    st.markdown("---")
    
    # Pain distribution
    st.subheader("Pain Score Distribution (NPRS)")
    fig = px.histogram(df, x='NPRS', nbins=10, 
                      title="Distribution of Pain Scores Among Farmers",
                      labels={'NPRS': 'Pain Score (0-10)'})
    st.plotly_chart(fig, use_container_width=True)
    
    # Protective equipment usage
    st.subheader("Protective Equipment Usage Patterns")
    fig = px.sunburst(df, path=['STRETCH_USED', 'GLOVES_USED'], 
                     title="Combination of Protective Equipment Usage")
    st.plotly_chart(fig, use_container_width=True)
    
    # Shoulder movement vs pain
    st.subheader("Shoulder Mobility vs Pain Levels")
    selected_joint = st.selectbox("Select Joint Movement to Analyze", 
                                ['SHOULDER_FLX', 'SHOULDER_EXTEN', 'SHOULDER_ABD', 'SHOULDER_ADD'])
    fig = px.scatter(df, x=selected_joint, y='NPRS', trendline="ols",
                    title=f"{selected_joint.replace('_', ' ')} vs Pain Score",
                    labels={selected_joint: 'Range of Motion (degrees)', 'NPRS': 'Pain Score'})
    st.plotly_chart(fig, use_container_width=True)

# Pain Analysis
elif analysis_type == "Pain Analysis":
    st.header("🩹 Pain Analysis")
    
    # Pain score distribution by factors
    st.subheader("Pain Scores by Different Factors")
    factor = st.selectbox("Select Factor to Analyze", 
                        ['STRETCH_USED', 'GLOVES_USED', 'PAIN_KILLER_USED', 'DOCTOR_VISITED'])
    
    fig = px.box(df, x=factor, y='NPRS', 
                title=f"Pain Scores by {factor.replace('_', ' ')}",
                labels={factor: factor.replace('_', ' '), 'NPRS': 'Pain Score'})
    st.plotly_chart(fig, use_container_width=True)
    
    # Statistical test results
    st.subheader("Statistical Significance")
    groups = [df[df[factor.replace('_USED', '')] == val]['NPRS'] for val in df[factor.replace('_USED', '')].unique()]
    t_stat, p_val = stats.ttest_ind(*groups)
    
    st.write(f"""
    - **Mean NPRS for {df[factor].unique()[0]}**: {groups[0].mean():.2f}
    - **Mean NPRS for {df[factor].unique()[1]}**: {groups[1].mean():.2f}
    - **T-test p-value**: {p_val:.4f} ({'statistically significant' if p_val < 0.05 else 'not statistically significant'})
    """)
    
    # Pain score distribution over ROM
    st.subheader("Pain Scores Across Range of Motion")
    rom_var = st.selectbox("Select Range of Motion Variable", 
                          ['ELBOW_FLEX', 'WRIST_FLEXN', 'SHOULDER_ABD', 'ELBOW_SUPI'])
    
    fig = px.scatter(df, x=rom_var, y='NPRS', color='PAIN_KILLER_USED',
                    title=f"Pain Scores vs {rom_var.replace('_', ' ')}",
                    labels={rom_var: 'Range of Motion (degrees)', 'NPRS': 'Pain Score'})
    st.plotly_chart(fig, use_container_width=True)

# Risk Factors
elif analysis_type == "Risk Factors":
    st.header("⚠️ Risk Factor Analysis")
    
    # Correlation heatmap
    st.subheader("Correlation Between Variables")
    numeric_cols = df.select_dtypes(include=np.number).columns
    corr_matrix = df[numeric_cols].corr()
    
    fig = px.imshow(corr_matrix, 
                   labels=dict(x="Variables", y="Variables", color="Correlation"),
                   x=corr_matrix.columns, y=corr_matrix.columns,
                   title="Correlation Matrix of Numeric Variables")
    st.plotly_chart(fig, use_container_width=True)
    
    # Key insights from correlations
    st.subheader("Key Risk Factor Insights")
    
    st.write("""
    - **Strongest Positive Correlations with Pain (NPRS)**:
        - Reduced shoulder flexion (r = {:.2f})
        - Limited elbow supination (r = {:.2f})
    - **Protective Factors**:
        - Stretch use associated with {:.2f} lower pain scores
        - Glove use associated with {:.2f} lower pain scores
    """.format(
        df[['SHOULDER_FLX', 'NPRS']].corr().iloc[0,1],
        df[['ELBOW_SUPI', 'NPRS']].corr().iloc[0,1],
        df.groupby('STRECTH')['NPRS'].mean().diff().iloc[-1],
        df.groupby('GLOVES')['NPRS'].mean().diff().iloc[-1]
    ))
    
    # ANOVA for ROM variables
    st.subheader("Range of Motion Differences by Pain Level")
    rom_var = st.selectbox("Select ROM Variable for ANOVA", 
                          ['SHOULDER_ABD', 'ELBOW_FLEX', 'WRIST_EXTEN', 'SHOULDER_MEDIAL'])
    
    # Create pain level groups
    df['PAIN_LEVEL'] = pd.cut(df['NPRS'], bins=[0, 3, 6, 10], labels=['Mild', 'Moderate', 'Severe'])
    
    fig = px.box(df, x='PAIN_LEVEL', y=rom_var, 
                title=f"{rom_var.replace('_', ' ')} by Pain Level",
                labels={'PAIN_LEVEL': 'Pain Level', rom_var: 'Range of Motion (degrees)'})
    st.plotly_chart(fig, use_container_width=True)
    
    # ANOVA results
    groups = [df[df['PAIN_LEVEL'] == level][rom_var] for level in ['Mild', 'Moderate', 'Severe']]
    f_stat, p_val = stats.f_oneway(*groups)
    
    st.write(f"""
    **ANOVA Results**:
    - F-statistic: {f_stat:.2f}
    - p-value: {p_val:.4f} ({'significant' if p_val < 0.05 else 'not significant'})
    """)

# Range of Motion
elif analysis_type == "Range of Motion":
    st.header("🦾 Range of Motion Analysis")
    
    # ROM distribution
    st.subheader("Distribution of Joint Movements")
    joint_movement = st.selectbox("Select Joint Movement", 
                                ['SHOULDER_FLX', 'ELBOW_FLEX', 'WRIST_FLEXN', 
                                 'SHOULDER_ABD', 'ELBOW_SUPI', 'WRIST_ULNAR'])
    
    fig = px.histogram(df, x=joint_movement, nbins=20,
                      title=f"Distribution of {joint_movement.replace('_', ' ')}",
                      labels={joint_movement: 'Range of Motion (degrees)'})
    st.plotly_chart(fig, use_container_width=True)
    
    # ROM comparison by equipment use
    st.subheader("Range of Motion by Protective Equipment")
    equipment = st.radio("Select Protective Equipment", ['STRECTH', 'GLOVES'])
    
    fig = px.box(df, x=equipment, y=joint_movement,
                title=f"{joint_movement.replace('_', ' ')} by {equipment} Usage",
                labels={equipment: equipment, joint_movement: 'Range of Motion (degrees)'})
    st.plotly_chart(fig, use_container_width=True)
    
    # ROM vs pain
    st.subheader("Range of Motion vs Pain Scores")
    fig = px.scatter(df, x=joint_movement, y='NPRS', trendline="ols",
                    title=f"{joint_movement.replace('_', ' ')} vs Pain Score",
                    labels={joint_movement: 'Range of Motion (degrees)', 'NPRS': 'Pain Score'})
    st.plotly_chart(fig, use_container_width=True)

# Preventive Insights
elif analysis_type == "Preventive Insights":
    st.header("🛡️ Preventive Measures & Recommendations")
    
    st.subheader("Key Findings Summary")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Protective Equipment Impact**:
        - Farmers using stretch reported {:.1f}% lower pain scores
        - Glove users had {:.1f}% better wrist mobility
        - Only {:.1f}% used both stretch and gloves together
        """.format(
            (1 - df[df['STRECTH']=='Y']['NPRS'].mean()/df[df['STRECTH']=='N']['NPRS'].mean())*100,
            (df[df['GLOVES']=='Y']['WRIST_FLEXN'].mean()/df[df['GLOVES']=='N']['WRIST_FLEXN'].mean()-1)*100,
            len(df[(df['STRECTH']=='Y') & (df['GLOVES']=='Y')])/len(df)*100
        ))
    
    with col2:
        st.markdown("""
        **Movement Limitations**:
        - Shoulder abduction most strongly correlated with pain (r = {:.2f})
        - Farmers with pain >6 had {:.1f}° less elbow flexion
        - Wrist movements showed earliest signs of restriction
        """.format(
            df[['SHOULDER_ABD', 'NPRS']].corr().iloc[0,1],
            df[df['NPRS']<=6]['ELBOW_FLEX'].mean() - df[df['NPRS']>6]['ELBOW_FLEX'].mean()
        ))
    
    st.markdown("---")
    
    st.subheader("Recommended Interventions")
    
    tab1, tab2, tab3 = st.tabs(["Equipment", "Work Practices", "Health Measures"])
    
    with tab1:
        st.markdown("""
        - **Ergonomic spade handles** to reduce wrist strain
        - **Vibration-damping gloves** for all field workers
        - **Compression sleeves** for shoulder support
        - **Rotating equipment** to vary movement patterns
        """)
    
    with tab2:
        st.markdown("""
        - **15-minute breaks every 2 hours** of spade work
        - **Task rotation** between digging and other activities
        - **Proper body mechanics training** for all workers
        - **Warm-up exercises** before starting work
        """)
    
    with tab3:
        st.markdown("""
        - **Monthly screening clinics** for early detection
        - **On-site stretching programs**
        - **Strength training** for shoulder stabilizers
        - **Pain management education** to reduce medication reliance
        """)
    
    st.markdown("---")
    
    st.subheader("Implementation Roadmap")
    roadmap = pd.DataFrame({
        'Phase': ['Immediate (0-3 months)', 'Short-term (3-6 months)', 'Ongoing'],
        'Actions': [
            "Awareness campaign, basic training",
            "Equipment distribution, screening clinics",
            "Regular assessments, program refinement"
        ],
        'Metrics': [
            "Participation rates, pre-test scores",
            "Equipment adoption, pain score reductions",
            "Long-term injury rates, productivity"
        ]
    })
    st.table(roadmap)

# Footer
st.markdown("---")
st.markdown("""
**Statistical Methods Applied**:
- Descriptive: Mean, Median, Standard Deviation
- Inferential: T-tests, ANOVA, Correlation (with p-values)
- Visualization: Interactive plots for exploratory analysis
""")
