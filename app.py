import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go

# Set page config
st.set_page_config(
    page_title="Farmer's Upper Extremity Injury Analysis",
    page_icon="🌾",
    layout="wide"
)

# Load data
@st.cache_data
def load_data():
    # Load the dataset
    df = pd.read_excel("DATA.xlsx")
    return df

df = load_data()

# Title
st.title("🌾 Prevalence & Risk Factors of Upper Extremities Injury in Farmers Using Spades")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select Page", [
    "📊 Overview",
    "🩹 Pain Analysis",
    "⚠️ Risk Factors",
    "💪 Range of Motion",
    "📈 Statistical Tests",
    "🛡️ Recommendations",
    "⏱️ Work Patterns"
])

# Sidebar filters (available on all pages)
st.sidebar.header("Filters")
pain_threshold = st.sidebar.slider("Filter by Pain Score (NPRS)", 0, 10, (0, 10))
filtered_df = df[(df['NPRS'] >= pain_threshold[0]) & (df['NPRS'] <= pain_threshold[1])]

# Helper function for consistent styling
def styled_metric(label, value, help_text=None):
    return f"""
    <div style="
        border-radius: 10px;
        padding: 15px;
        background-color: #f0f2f6;
        margin: 10px 0;
    ">
        <div style="font-size: 14px; color: #555;">{label}</div>
        <div style="font-size: 24px; font-weight: bold; color: #333;">{value}</div>
        {f'<div style="font-size: 12px; color: #777;">{help_text}</div>' if help_text else ''}
    </div>
    """

# Overview Page
if page == "📊 Overview":
    st.header("📊 Dataset Overview")
    
    # Key metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(styled_metric(
            "Total Farmers", 
            len(df),
            f"Filtered: {len(filtered_df)}"
        ), unsafe_allow_html=True)
    with col2:
        st.markdown(styled_metric(
            "Average Pain Score", 
            f"{df['NPRS'].mean():.1f}/10",
            f"Filtered: {filtered_df['NPRS'].mean():.1f}"
        ), unsafe_allow_html=True)
    with col3:
        pain_killer_rate = df['PAIN_KILLER'].value_counts(normalize=True).get('Y', 0)*100
        st.markdown(styled_metric(
            "Using Pain Killers", 
            f"{pain_killer_rate:.1f}%",
            "Percentage of farmers"
        ), unsafe_allow_html=True)
    
    # Data preview
    st.subheader("Data Preview")
    st.dataframe(filtered_df.head(10))
    
    # Pain distribution
    st.subheader("Pain Score Distribution")
    fig = px.histogram(filtered_df, x='NPRS', nbins=11, 
                      title='Distribution of Pain Scores (NPRS 0-10)',
                      labels={'NPRS': 'Pain Score', 'count': 'Number of Farmers'},
                      opacity=0.7)
    fig.update_traces(marker_color='#4CAF50', 
                     hovertemplate="Pain Score: %{x}<br>Count: %{y}")
    st.plotly_chart(fig, use_container_width=True)
    
    # Quick insights
    st.subheader("🔍 Quick Insights")
    insights = [
        f"1. {len(df[df['NPRS'] >= 7])} farmers ({len(df[df['NPRS'] >= 7])/len(df)*100:.1f}%) experience severe pain (NPRS ≥7)",
        f"2. Farmers using pain killers report {df[df['PAIN_KILLER']=='Y']['NPRS'].mean():.1f} average pain vs {df[df['PAIN_KILLER']=='N']['NPRS'].mean():.1f} for non-users",
        f"3. Shoulder flexion shows {df['SHOULDER_FLX'].corr(df['NPRS']):.2f} correlation with pain scores",
        f"4. Only {len(df[df['GLOVES']=='Y'])} farmers ({len(df[df['GLOVES']=='Y'])/len(df)*100:.1f}%) use gloves regularly"
    ]
    for insight in insights:
        st.markdown(f"<div style='padding: 10px; background-color: #f8f9fa; border-left: 4px solid #4CAF50; margin: 5px 0;'>{insight}</div>", unsafe_allow_html=True)

# Pain Analysis Page
elif page == "🩹 Pain Analysis":
    st.header("🩹 Detailed Pain Analysis")
    
    # Pain score distribution by category
    st.subheader("Pain Score Distribution by Factors")
    factor = st.selectbox("Select factor to analyze", 
                         ['PAIN_KILLER', 'GLOVES', 'DOCTOR_CONSULTED', 'STRECTH'])
    
    fig = px.box(filtered_df, x=factor, y='NPRS', 
                title=f'Pain Score Distribution by {factor}',
                labels={'NPRS': 'Pain Score', factor: factor.replace('_', ' ').title()},
                color=factor)
    fig.update_traces(hovertemplate="%{x}<br>Pain Score: %{y}")
    st.plotly_chart(fig, use_container_width=True)
    
    # Pain score statistics
    st.subheader("Pain Score Statistics")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Overall Statistics**")
        overall_stats = filtered_df['NPRS'].describe().to_frame()
        st.dataframe(overall_stats.style.format("{:.2f}"))
    
    with col2:
        st.markdown(f"**Statistics by {factor}**")
        group_stats = filtered_df.groupby(factor)['NPRS'].agg(['mean', 'median', 'std', 'count'])
        st.dataframe(group_stats.style.format("{:.2f}"))
    
    # Pain score trends
    st.subheader("Pain Score Trends")
    motion_type = st.selectbox("Select motion to analyze", 
                              ['SHOULDER_FLX', 'SHOULDER_EXTEN', 'ELBOW_FLEX', 'WRIST_FLEXN'])
    
    fig = px.scatter(filtered_df, x=motion_type, y='NPRS', 
                    trendline="ols",
                    title=f'Pain Scores vs {motion_type}',
                    labels={motion_type: f'{motion_type} (degrees)', 'NPRS': 'Pain Score'},
                    hover_data=[motion_type, 'NPRS'])
    fig.update_traces(marker=dict(size=8, opacity=0.6))
    st.plotly_chart(fig, use_container_width=True)
    
    # Calculate correlation
    corr = filtered_df[motion_type].corr(filtered_df['NPRS'])
    st.markdown(f"**Correlation coefficient:** {corr:.2f}")
    if corr > 0.3:
        st.info("Moderate positive correlation - Higher values may be associated with more pain")
    elif corr < -0.3:
        st.info("Moderate negative correlation - Lower values may be associated with more pain")
    else:
        st.info("Weak or no correlation observed")

# Risk Factors Page
elif page == "⚠️ Risk Factors":
    st.header("⚠️ Risk Factor Analysis")
    
    # Binary risk factors
    st.subheader("Binary Risk Factors")
    binary_factors = ['STRECTH', 'GLOVES', 'DOCTOR_CONSULTED', 'PAIN_KILLER']
    
    cols = st.columns(2)
    for i, factor in enumerate(binary_factors):
        with cols[i%2]:
            st.markdown(f"**{factor.replace('_', ' ').title()}**")
            if factor in filtered_df.columns:
                # Calculate percentages
                counts = filtered_df[factor].value_counts(normalize=True).reset_index()
                counts.columns = [factor, 'percentage']
                counts['percentage'] = counts['percentage'] * 100
                
                # Create pie chart
                fig = px.pie(counts, values='percentage', names=factor,
                            hover_data=['percentage'],
                            labels={'percentage': 'Percentage'},
                            hole=0.3)
                fig.update_traces(textposition='inside', 
                                 textinfo='percent+label',
                                 hovertemplate="%{label}<br>Percentage: %{percent}")
                st.plotly_chart(fig, use_container_width=True)
                
                # Calculate mean NPRS difference
                if len(counts) > 1:
                    mean_diff = filtered_df[filtered_df[factor]=='Y']['NPRS'].mean() - filtered_df[filtered_df[factor]=='N']['NPRS'].mean()
                    st.markdown(f"Mean NPRS difference: {mean_diff:.2f}")
    
    # Range of motion factors
    st.subheader("Range of Motion Risk Factors")
    motion_factors = [col for col in df.columns if any(x in col for x in ['SHOULDER', 'ELBOW', 'WRIST'])]
    selected_motion = st.selectbox("Select motion factor", motion_factors)
    
    # Create bins for analysis
    filtered_df['motion_bin'] = pd.cut(filtered_df[selected_motion], bins=5)
    filtered_df['motion_bin_str'] = filtered_df['motion_bin'].astype(str)  # Convert to string for Plotly
    
    fig = px.box(filtered_df, x='motion_bin_str', y='NPRS', 
                title=f'Pain Scores by {selected_motion} Range',
                labels={'motion_bin_str': f'{selected_motion} Range (degrees)', 'NPRS': 'Pain Score'},
                color='motion_bin_str')
    fig.update_traces(hovertemplate="Range: %{x}<br>Pain Score: %{y}")
    st.plotly_chart(fig, use_container_width=True)
    
    # ANOVA test
    groups = [filtered_df[filtered_df['motion_bin'] == bin]['NPRS'] for bin in filtered_df['motion_bin'].unique()]
    f_stat, p_val = stats.f_oneway(*groups)
    
    st.markdown(f"**ANOVA Test Results for {selected_motion}:**")
    st.markdown(f"- F-statistic: {f_stat:.3f}")
    st.markdown(f"- p-value: {p_val:.3f}")
    if p_val < 0.05:
        st.success("Statistically significant differences exist between motion ranges (p < 0.05)")
    else:
        st.warning("No statistically significant differences found (p ≥ 0.05)")

# Range of Motion Page
elif page == "💪 Range of Motion":
    st.header("💪 Range of Motion Analysis")
    
    # Select joint to analyze
    joint = st.radio("Select Joint", ['Shoulder', 'Elbow', 'Wrist'])
    
    if joint == 'Shoulder':
        motions = ['SHOULDER_FLX', 'SHOULDER_EXTEN', 'SHOULDER_ABD', 'SHOULDER_ADD', 'SHOULDER_MEDIAL', 'SHOULDER_LATERAL']
    elif joint == 'Elbow':
        motions = ['ELBOW_FLEX', 'ELBOW_EXTENSION', 'ELBOW_SUPI', 'ELBOW_PORANA']
    else:
        motions = ['WRIST_FLEXN', 'WRIST_EXTEN', 'WRIST_ULNAR', 'WRIST_RADIAL']
    
    # Motion vs pain correlation matrix
    st.subheader(f"{joint} Motion Correlations with Pain")
    corr_matrix = filtered_df[motions + ['NPRS']].corr()
    
    fig = px.imshow(corr_matrix,
                   text_auto=True,
                   aspect="auto",
                   labels=dict(x="Variable", y="Variable", color="Correlation"),
                   x=corr_matrix.columns,
                   y=corr_matrix.columns,
                   color_continuous_scale='RdBu',
                   zmin=-1,
                   zmax=1)
    fig.update_xaxes(side="top")
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed motion analysis
    st.subheader("Detailed Motion Analysis")
    selected_motion = st.selectbox("Select specific motion", motions)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Motion Distribution**")
        fig = px.histogram(filtered_df, x=selected_motion, 
                          nbins=15,
                          labels={selected_motion: 'Degrees of Motion', 'count': 'Count'},
                          opacity=0.7)
        fig.update_traces(marker_color='#4CAF50',
                         hovertemplate="Degrees: %{x}<br>Count: %{y}")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("**Motion vs Pain**")
        fig = px.scatter(filtered_df, x=selected_motion, y='NPRS',
                        labels={selected_motion: 'Degrees of Motion', 'NPRS': 'Pain Score'},
                        opacity=0.6)
        fig.update_traces(marker=dict(color='#2196F3', size=8),
                         hovertemplate="Degrees: %{x}<br>Pain Score: %{y}")
        st.plotly_chart(fig, use_container_width=True)
    
    # Motion statistics
    st.markdown("**Motion Statistics**")
    motion_stats = filtered_df[selected_motion].describe().to_frame()
    st.dataframe(motion_stats.style.format("{:.2f}"))

# Statistical Tests Page
elif page == "📈 Statistical Tests":
    st.header("📈 Statistical Analysis")
    
    # T-test for binary factors
    st.subheader("T-tests for Binary Factors")
    binary_factor = st.selectbox("Select binary factor", 
                                ['GLOVES', 'PAIN_KILLER', 'DOCTOR_CONSULTED', 'STRECTH'])
    
    if len(filtered_df[binary_factor].unique()) == 2:
        group1 = filtered_df[filtered_df[binary_factor] == filtered_df[binary_factor].unique()[0]]['NPRS']
        group2 = filtered_df[filtered_df[binary_factor] == filtered_df[binary_factor].unique()[1]]['NPRS']
        
        # Display group statistics
        st.markdown(f"**Group Statistics for {binary_factor}**")
        stats_df = pd.DataFrame({
            'Group': [filtered_df[binary_factor].unique()[0], filtered_df[binary_factor].unique()[1]],
            'Mean NPRS': [group1.mean(), group2.mean()],
            'Std Dev': [group1.std(), group2.std()],
            'Count': [len(group1), len(group2)]
        })
        st.dataframe(stats_df)
        
        # Perform t-test
        t_stat, p_val = stats.ttest_ind(group1, group2)
        
        st.markdown("**Independent Samples T-test Results**")
        st.markdown(f"- t-statistic: {t_stat:.3f}")
        st.markdown(f"- p-value: {p_val:.3f}")
        
        if p_val < 0.05:
            st.success("Statistically significant difference between groups (p < 0.05)")
            st.markdown(f"**Effect Size (Cohen's d):** {abs((group1.mean() - group2.mean()) / np.sqrt((group1.std()**2 + group2.std()**2)/2)):.2f}")
        else:
            st.warning("No statistically significant difference (p ≥ 0.05)")
    else:
        st.warning("Selected factor doesn't have exactly 2 groups for t-test")
    
    # ANOVA for continuous factors
    st.subheader("ANOVA for Continuous Factors")
    continuous_factor = st.selectbox("Select continuous factor", 
                                   ['SHOULDER_FLX', 'SHOULDER_ABD', 'ELBOW_FLEX', 'WRIST_FLEXN'])
    
    # Create bins
    filtered_df['factor_bin'] = pd.cut(filtered_df[continuous_factor], bins=5)
    filtered_df['factor_bin_str'] = filtered_df['factor_bin'].astype(str)  # Convert to string for display
    
    # Display bin statistics
    st.markdown(f"**Binned Statistics for {continuous_factor}**")
    bin_stats = filtered_df.groupby('factor_bin_str')['NPRS'].agg(['mean', 'std', 'count'])
    st.dataframe(bin_stats)
    
    # Perform ANOVA
    groups = [filtered_df[filtered_df['factor_bin'] == bin]['NPRS'] for bin in filtered_df['factor_bin'].unique()]
    f_stat, p_val = stats.f_oneway(*groups)
    
    st.markdown("**ANOVA Results**")
    st.markdown(f"- F-statistic: {f_stat:.3f}")
    st.markdown(f"- p-value: {p_val:.3f}")
    
    if p_val < 0.05:
        st.success("Statistically significant differences between groups (p < 0.05)")
    else:
        st.warning("No statistically significant differences (p ≥ 0.05)")

# Recommendations Page
elif page == "🛡️ Recommendations":
    st.header("🛡️ Preventive Measures & Recommendations")
    
    # Key findings
    st.subheader("Key Findings Summary")
    
    findings = [
        ("1. High Pain Prevalence", f"{len(df[df['NPRS'] >= 5])/len(df)*100:.1f}% of farmers report moderate to severe pain (NPRS ≥5)"),
        ("2. Protective Effect of Gloves", f"Glove users report {df[df['GLOVES']=='Y']['NPRS'].mean():.1f} vs {df[df['GLOVES']=='N']['NPRS'].mean():.1f} average pain for non-users"),
        ("3. Shoulder Mobility", f"Shoulder flexion shows {df['SHOULDER_FLX'].corr(df['NPRS']):.2f} correlation with pain scores"),
        ("4. Pain Medication Use", f"{df['PAIN_KILLER'].value_counts(normalize=True).get('Y', 0)*100:.1f}% use pain killers regularly"),
        ("5. Doctor Consultations", f"Only {df['DOCTOR_CONSULTED'].value_counts(normalize=True).get('Y', 0)*100:.1f}% have consulted a doctor")
    ]
    
    for title, content in findings:
        with st.expander(title):
            st.markdown(content)
    
    # Recommendations
    st.subheader("Evidence-Based Recommendations")
    
    recommendations = [
        ("👐 Promote Glove Usage", "Farmers not using gloves report higher pain scores. Provide ergonomic gloves to reduce strain."),
        ("🏋️ Shoulder Strengthening", "Targeted exercises to improve shoulder mobility may reduce pain based on correlation analysis."),
        ("💊 Reduce Pain Killer Reliance", "High usage suggests underlying issues need addressing rather than symptomatic treatment."),
        ("🩺 Encourage Medical Consultations", "Low consultation rates indicate potential under-treatment of chronic issues."),
        ("🧘 Implement Stretching Programs", "Regular stretching may help prevent repetitive strain injuries.")
    ]
    
    for title, content in recommendations:
        with st.expander(title):
            st.markdown(content)
    
    # Implementation plan
    st.subheader("Suggested Implementation Plan")
    st.markdown("""
    1. **Short-term (0-3 months):**
       - Distribute ergonomic gloves to all farmers
       - Conduct pain awareness workshops
       
    2. **Medium-term (3-6 months):**
       - Implement on-site stretching programs
       - Arrange regular medical check-ups
       
    3. **Long-term (6+ months):**
       - Develop farmer-specific exercise programs
       - Establish ongoing monitoring system
    """)

# New Work Patterns Page
elif page == "⏱️ Work Patterns":
    st.header("⏱️ Work Pattern Analysis")


    # Experience Level Graph
    st.subheader("Experience Level Distribution")
    experience_map = {
        'A': 'Less than 1 year',
        'B': '1-5 years',
        'C': '6-10 years',
        'D': 'More than 10 years'
    }
    
    df['EXPERIENCE_LABEL'] = df['EXPERIENCE'].map(experience_map)
    
    fig_exp = px.histogram(df, x='EXPERIENCE_LABEL',
                         category_orders={"EXPERIENCE_LABEL": ['Less than 1 year', '1-5 years', '6-10 years', 'More than 10 years']},
                         title='Distribution of Farmers by Experience Level',
                         labels={'EXPERIENCE_LABEL': 'Experience Level', 'count': 'Number of Farmers'},
                         color='EXPERIENCE_LABEL',
                         color_discrete_sequence=px.colors.qualitative.Pastel)
    fig_exp.update_layout(showlegend=False)
    fig_exp.update_traces(hovertemplate="Experience: %{x}<br>Count: %{y}")
    st.plotly_chart(fig_exp, use_container_width=True)
    
    # Duration Level Graph
    st.subheader("Daily Working Duration Distribution")
    duration_map = {
        'A': 'Less than 4 hours',
        'B': '4-6 hours',
        'C': '7-8 hours',
        'D': 'More than 9 hours'
    }
    
    df['HOURS_DAY_LABEL'] = df['HOURS_DAY'].map(duration_map)
    
    fig_dur = px.histogram(df, x='HOURS_DAY_LABEL',
                          category_orders={"HOURS_DAY_LABEL": ['Less than 4 hours', '4-6 hours', '7-8 hours', 'More than 9 hours']},
                          title='Distribution of Daily Working Hours',
                          labels={'HOURS_DAY_LABEL': 'Hours per Day', 'count': 'Number of Farmers'},
                          color='HOURS_DAY_LABEL',
                          color_discrete_sequence=px.colors.qualitative.Pastel)
    fig_dur.update_layout(showlegend=False)
    fig_dur.update_traces(hovertemplate="Hours: %{x}<br>Count: %{y}")
    st.plotly_chart(fig_dur, use_container_width=True)


    
    # Experience analysis
    st.subheader("Experience Level Analysis")
    experience_map = {
        'A': 'Less than 1 year',
        'B': '1-5 years',
        'C': '6-10 years',
        'D': 'More than 10 years'
    }
    
    df['EXPERIENCE_LABEL'] = df['EXPERIENCE'].map(experience_map)
    
    fig = px.box(df, x='EXPERIENCE_LABEL', y='NPRS', 
                category_orders={"EXPERIENCE_LABEL": ['Less than 1 year', '1-5 years', '6-10 years', 'More than 10 years']},
                title='Pain Scores by Farming Experience',
                labels={'EXPERIENCE_LABEL': 'Experience Level', 'NPRS': 'Pain Score'},
                color='EXPERIENCE_LABEL')
    fig.update_traces(hovertemplate="Experience: %{x}<br>Pain Score: %{y}")
    st.plotly_chart(fig, use_container_width=True)
    
    # Hours per day analysis
    st.subheader("Working Hours Analysis")
    
    hours_map = {
        'A': 'Less than 4 hours',
        'B': '4-6 hours',
        'C': '7-8 hours',
        'D': 'More than 9 hours'
    }
    
    df['HOURS_DAY_LABEL'] = df['HOURS_DAY'].map(hours_map)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Distribution of Daily Working Hours**")
        fig = px.histogram(df, x='HOURS_DAY_LABEL',
                          category_orders={"HOURS_DAY_LABEL": ['Less than 4 hours', '4-6 hours', '7-8 hours', 'More than 9 hours']},
                          labels={'HOURS_DAY_LABEL': 'Hours per Day', 'count': 'Count'},
                          color='HOURS_DAY_LABEL')
        fig.update_traces(hovertemplate="Hours: %{x}<br>Count: %{y}")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("**Pain Scores by Daily Working Hours**")
        fig = px.box(df, x='HOURS_DAY_LABEL', y='NPRS',
                    category_orders={"HOURS_DAY_LABEL": ['Less than 4 hours', '4-6 hours', '7-8 hours', 'More than 9 hours']},
                    labels={'HOURS_DAY_LABEL': 'Hours per Day', 'NPRS': 'Pain Score'},
                    color='HOURS_DAY_LABEL')
        fig.update_traces(hovertemplate="Hours: %{x}<br>Pain Score: %{y}")
        st.plotly_chart(fig, use_container_width=True)
    
    # Break duration analysis
    st.subheader("Break Duration Analysis")
    
    break_map = {
        'A': 'Less than 5 min',
        'B': '5-10 min',
        'C': 'More than 10 min'
    }
    
    df['HOW_LONG_BREAK_LABEL'] = df['HOW_LONG_BREAK'].map(break_map)
    
    fig = px.box(df, x='HOW_LONG_BREAK_LABEL', y='NPRS',
                category_orders={"HOW_LONG_BREAK_LABEL": ['Less than 5 min', '5-10 min', 'More than 10 min']},
                title='Pain Scores by Break Duration',
                labels={'HOW_LONG_BREAK_LABEL': 'Break Duration', 'NPRS': 'Pain Score'},
                color='HOW_LONG_BREAK_LABEL')
    fig.update_traces(hovertemplate="Break Duration: %{x}<br>Pain Score: %{y}")
    st.plotly_chart(fig, use_container_width=True)
    
    # Affected area analysis
    st.subheader("Most Affected Area Analysis")
    
    area_map = {
        'A': 'Shoulder',
        'B': 'Elbow',
        'C': 'Wrist',
        'D': 'Multiple areas'
    }
    
    df['AREA_LABEL'] = df['AREA'].map(area_map)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Distribution of Affected Areas**")
        fig = px.histogram(df, x='AREA_LABEL',
                          category_orders={"AREA_LABEL": ['Shoulder', 'Elbow', 'Wrist', 'Multiple areas']},
                          labels={'AREA_LABEL': 'Affected Area', 'count': 'Count'},
                          color='AREA_LABEL')
        fig.update_traces(hovertemplate="Area: %{x}<br>Count: %{y}")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("**Pain Scores by Affected Area**")
        fig = px.box(df, x='AREA_LABEL', y='NPRS',
                    category_orders={"AREA_LABEL": ['Shoulder', 'Elbow', 'Wrist', 'Multiple areas']},
                    labels={'AREA_LABEL': 'Affected Area', 'NPRS': 'Pain Score'},
                    color='AREA_LABEL')
        fig.update_traces(hovertemplate="Area: %{x}<br>Pain Score: %{y}")
        st.plotly_chart(fig, use_container_width=True)
    
    # Pain frequency analysis
    st.subheader("Pain Frequency Analysis")
    
    freq_map = {
        'A': 'Daily',
        'B': 'Weekly',
        'C': 'Occasionally',
        'D': 'Never'
    }
    
    df['FREQUENTLY_LABEL'] = df['FREQUENTLY'].map(freq_map)
    
    fig = px.histogram(df, x='FREQUENTLY_LABEL',
                      category_orders={"FREQUENTLY_LABEL": ['Daily', 'Weekly', 'Occasionally', 'Never']},
                      title='Distribution of Pain Frequency',
                      labels={'FREQUENTLY_LABEL': 'Pain Frequency', 'count': 'Count'},
                      color='FREQUENTLY_LABEL')
    fig.update_traces(hovertemplate="Frequency: %{x}<br>Count: %{y}")
    st.plotly_chart(fig, use_container_width=True)
    
    # Stiffness and numbness analysis
    st.subheader("Stiffness and Numbness Analysis")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Joint Stiffness (STIFFNESS)**")
        stiffness_counts = df['STIFFNESS'].value_counts().reset_index()
        stiffness_counts.columns = ['STIFFNESS', 'count']
        fig = px.pie(stiffness_counts, values='count', names='STIFFNESS',
                    hover_data=['count'],
                    labels={'count': 'Count'},
                    hole=0.3)
        fig.update_traces(textposition='inside', 
                         textinfo='percent+label',
                         hovertemplate="%{label}<br>Count: %{value}")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("**Hand Numbness (NUMBNESS)**")
        numbness_counts = df['NUMBNESS'].value_counts().reset_index()
        numbness_counts.columns = ['NUMBNESS', 'count']
        fig = px.pie(numbness_counts, values='count', names='NUMBNESS',
                    hover_data=['count'],
                    labels={'count': 'Count'},
                    hole=0.3)
        fig.update_traces(textposition='inside', 
                         textinfo='percent+label',
                         hovertemplate="%{label}<br>Count: %{value}")
        st.plotly_chart(fig, use_container_width=True)
    
    # Statistical test for affected area
    st.subheader("Statistical Analysis for Affected Area")
    
    # Perform ANOVA
    groups = [df[df['AREA_LABEL'] == area]['NPRS'] for area in df['AREA_LABEL'].unique()]
    f_stat, p_val = stats.f_oneway(*groups)
    
    st.markdown("**ANOVA Results for Affected Area vs Pain Score:**")
    st.markdown(f"- F-statistic: {f_stat:.3f}")
    st.markdown(f"- p-value: {p_val:.3f}")
    if p_val < 0.05:
        st.success("Statistically significant differences exist between affected areas (p < 0.05)")
    else:
        st.warning("No statistically significant differences found (p ≥ 0.05)")

# Footer
st.markdown("---")
st.markdown("""
**About This Dashboard:**  
This interactive tool analyzes upper extremity injuries in farmers using spades, with objectives to:
1. Assess pain frequency
2. Identify risk factors
3. Suggest preventive measures
""")
