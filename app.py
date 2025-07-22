import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

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
    
    # Create mappings for categorical variables
    experience_map = {
        'A': '0-1 years',
        'B': '1-5 years',
        'C': '6-10 years',
        'D': '10+ years'
    }
    
    hours_map = {
        'A': '<4 hrs',
        'B': '4-6 hrs',
        'C': '7-8 hrs',
        'D': '9+ hrs'
    }
    
    break_map = {
        'A': '<5 min',
        'B': '5-10 min',
        'C': '10+ min'
    }
    
    area_map = {
        'A': 'Shoulder',
        'B': 'Elbow',
        'C': 'Wrist',
        'D': 'Multiple'
    }
    
    freq_map = {
        'A': 'Daily',
        'B': 'Weekly',
        'C': 'Occasionally',
        'D': 'Never'
    }
    
    # Add labeled columns
    df['EXPERIENCE_LABEL'] = df['EXPERIENCE'].map(experience_map)
    df['HOURS_DAY_LABEL'] = df['HOURS_DAY'].map(hours_map)
    df['HOW_LONG_BREAK_LABEL'] = df['HOW_LONG_BREAK'].map(break_map)
    df['AREA_LABEL'] = df['AREA'].map(area_map)
    df['FREQUENTLY_LABEL'] = df['FREQUENTLY'].map(freq_map)
    
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
age_range = st.sidebar.slider("Filter by Age", 
                             min_value=int(df['AGE'].min()), 
                             max_value=int(df['AGE'].max()),
                             value=(int(df['AGE'].min()), int(df['AGE'].max())))

gender_filter = st.sidebar.multiselect("Filter by Gender",
                                     options=df['GENDER'].unique(),
                                     default=df['GENDER'].unique())

# Apply filters
filtered_df = df[
    (df['AGE'] >= age_range[0]) & 
    (df['AGE'] <= age_range[1]) & 
    (df['GENDER'].isin(gender_filter))
]

# Helper function for consistent styling
def styled_metric(label, value, help_text=None):
    return f"""
    <div style="
        border-radius: 10px;
        padding: 15px;
        background-color: #f0f2f6;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
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
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(styled_metric(
            "Total Farmers", 
            len(df),
            f"Filtered: {len(filtered_df)}"
        ), unsafe_allow_html=True)
    with col2:
        st.markdown(styled_metric(
            "Average Age", 
            f"{filtered_df['AGE'].mean():.1f} years",
            f"Range: {age_range[0]}-{age_range[1]}"
        ), unsafe_allow_html=True)
    with col3:
        gender_dist = filtered_df['GENDER'].value_counts(normalize=True).get('M', 0)*100
        st.markdown(styled_metric(
            "Gender Distribution", 
            f"{gender_dist:.1f}% Male",
            f"{100-gender_dist:.1f}% Female"
        ), unsafe_allow_html=True)
    with col4:
        pain_killer_rate = filtered_df['PAIN_KILLER'].value_counts(normalize=True).get('Y', 0)*100
        st.markdown(styled_metric(
            "Pain Killer Usage", 
            f"{pain_killer_rate:.1f}%",
            "Percentage of farmers"
        ), unsafe_allow_html=True)
    
    # Data preview
    st.subheader("Data Preview")
    st.dataframe(filtered_df.head(10), height=300)
    
    # Pain distribution
    st.subheader("Pain Score Distribution")
    
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.histplot(data=filtered_df, x='NPRS', bins=11, kde=True, color='#4CAF50', alpha=0.7)
    ax.set_title('Distribution of Pain Scores (NPRS 0-10)', fontsize=14, pad=20)
    ax.set_xlabel('Pain Score (NPRS)', fontsize=12)
    ax.set_ylabel('Number of Farmers', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.3)
    st.pyplot(fig)
    
    # Interactive distribution plot
    st.subheader("Interactive Distribution Explorer")
    
    dist_col1, dist_col2 = st.columns(2)
    with dist_col1:
        dist_var = st.selectbox("Select variable to explore", 
                              ['AGE', 'EXPERIENCE_LABEL', 'HOURS_DAY_LABEL', 'AREA_LABEL'])
    with dist_col2:
        plot_type = st.radio("Select plot type", ['Histogram', 'Boxplot', 'Violin'])
    
    fig, ax = plt.subplots(figsize=(12, 6))
    if plot_type == 'Histogram':
        if filtered_df[dist_var].dtype in ['int64', 'float64']:
            sns.histplot(data=filtered_df, x=dist_var, kde=True, color='#2196F3', bins=15)
        else:
            sns.countplot(data=filtered_df, x=dist_var, palette='viridis')
            plt.xticks(rotation=45)
    elif plot_type == 'Boxplot':
        sns.boxplot(data=filtered_df, x=dist_var, y='NPRS', palette='coolwarm')
        plt.xticks(rotation=45)
    else:
        sns.violinplot(data=filtered_df, x=dist_var, y='NPRS', palette='muted')
        plt.xticks(rotation=45)
    
    ax.set_title(f'Distribution of {dist_var}', fontsize=14)
    ax.set_xlabel(dist_var, fontsize=12)
    ax.set_ylabel('Count' if plot_type == 'Histogram' else 'Pain Score (NPRS)', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.3)
    st.pyplot(fig)
    
    # Quick insights
    st.subheader("🔍 Quick Insights")
    insights = [
        f"1. {len(filtered_df[filtered_df['NPRS'] >= 7])} farmers ({len(filtered_df[filtered_df['NPRS'] >= 7])/len(filtered_df)*100:.1f}%) experience severe pain (NPRS ≥7)",
        f"2. Farmers using pain killers report {filtered_df[filtered_df['PAIN_KILLER']=='Y']['NPRS'].mean():.1f} average pain vs {filtered_df[filtered_df['PAIN_KILLER']=='N']['NPRS'].mean():.1f} for non-users",
        f"3. {filtered_df['GLOVES'].value_counts(normalize=True).get('N', 0)*100:.1f}% don't use gloves during work",
        f"4. Most common affected area: {filtered_df['AREA_LABEL'].mode()[0]} ({filtered_df['AREA_LABEL'].value_counts(normalize=True).iloc[0]*100:.1f}%)"
    ]
    for insight in insights:
        st.markdown(f"""
        <div style='padding: 12px; 
                    background-color: #f8f9fa; 
                    border-left: 4px solid #4CAF50; 
                    margin: 8px 0;
                    border-radius: 4px;
                    box-shadow: 0 1px 2px rgba(0,0,0,0.1)'>
            {insight}
        </div>
        """, unsafe_allow_html=True)

# Pain Analysis Page
elif page == "🩹 Pain Analysis":
    st.header("🩹 Detailed Pain Analysis")
    
    # Pain score distribution by category
    st.subheader("Pain Score Distribution by Factors")
    
    col1, col2 = st.columns(2)
    with col1:
        factor = st.selectbox("Select factor to analyze", 
                            ['PAIN_KILLER', 'GLOVES', 'DOCTOR_CONSULTED', 'STRECTH', 
                             'EXPERIENCE_LABEL', 'HOURS_DAY_LABEL', 'AREA_LABEL'])
    with col2:
        plot_style = st.radio("Plot style", ['Boxplot', 'Violin', 'Swarm'])
    
    fig, ax = plt.subplots(figsize=(12, 6))
    if plot_style == 'Boxplot':
        sns.boxplot(data=filtered_df, x=factor, y='NPRS', palette='Set2')
    elif plot_style == 'Violin':
        sns.violinplot(data=filtered_df, x=factor, y='NPRS', palette='Set2', inner="quartile")
    else:
        sns.swarmplot(data=filtered_df, x=factor, y='NPRS', palette='Set2', size=4)
    
    ax.set_title(f'Pain Score Distribution by {factor}', fontsize=14, pad=15)
    ax.set_xlabel(factor, fontsize=12)
    ax.set_ylabel('Pain Score (NPRS)', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.3)
    plt.xticks(rotation=45)
    st.pyplot(fig)
    
    # Pain score statistics
    st.subheader("Pain Score Statistics")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Overall Statistics**")
        overall_stats = filtered_df['NPRS'].describe().to_frame()
        st.dataframe(overall_stats.style.format("{:.2f}").background_gradient(cmap='YlOrBr'))
    
    with col2:
        st.markdown(f"**Statistics by {factor}**")
        group_stats = filtered_df.groupby(factor)['NPRS'].agg(['mean', 'median', 'std', 'count'])
        st.dataframe(group_stats.style.format("{:.2f}").background_gradient(cmap='YlOrBr'))
    
    # Pain score trends
    st.subheader("Pain Score Trends by Continuous Variables")
    
    motion_type = st.selectbox("Select variable to analyze", 
                             ['AGE', 'SHOULDER_FLX', 'SHOULDER_EXTEN', 'ELBOW_FLEX', 'WRIST_FLEXN'])
    
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.regplot(data=filtered_df, x=motion_type, y='NPRS', 
               scatter_kws={'alpha':0.6, 'color':'#2196F3'}, 
               line_kws={'color':'red', 'linewidth':2})
    
    ax.set_title(f'Pain Scores vs {motion_type}', fontsize=14, pad=15)
    ax.set_xlabel(f'{motion_type}', fontsize=12)
    ax.set_ylabel('Pain Score (NPRS)', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.3)
    st.pyplot(fig)
    
    # Calculate correlation
    corr = filtered_df[motion_type].corr(filtered_df['NPRS'])
    st.markdown(f"**Correlation coefficient:** `{corr:.2f}`")
    
    if abs(corr) > 0.5:
        st.success("Strong correlation observed")
    elif abs(corr) > 0.3:
        st.info("Moderate correlation observed")
    elif abs(corr) > 0.1:
        st.warning("Weak correlation observed")
    else:
        st.error("Negligible or no correlation observed")

# Risk Factors Page
elif page == "⚠️ Risk Factors":
    st.header("⚠️ Risk Factor Analysis")
    
    # Binary risk factors
    st.subheader("Binary Risk Factors Analysis")
    
    binary_factors = ['STRECTH', 'GLOVES', 'DOCTOR_CONSULTED', 'PAIN_KILLER', 'TRAINING', 'WARM_UP']
    
    cols = st.columns(2)
    for i, factor in enumerate(binary_factors):
        with cols[i%2]:
            st.markdown(f"**{factor.replace('_', ' ').title()}**")
            if factor in filtered_df.columns:
                # Calculate percentages
                counts = filtered_df[factor].value_counts(normalize=True).sort_index()
                
                # Create pie chart
                fig, ax = plt.subplots(figsize=(6, 4))
                counts.plot(kind='pie', autopct='%1.1f%%', 
                          colors=['#ff9999','#66b3ff'], 
                          startangle=90,
                          wedgeprops={'linewidth': 1, 'edgecolor': 'white'},
                          textprops={'fontsize': 10})
                ax.set_ylabel('')
                ax.set_title(f'{factor} Distribution', pad=10)
                st.pyplot(fig)
                
                # Calculate mean NPRS difference
                if len(counts) > 1:
                    mean_diff = filtered_df[filtered_df[factor]=='Y']['NPRS'].mean() - filtered_df[filtered_df[factor]=='N']['NPRS'].mean()
                    direction = "higher" if mean_diff > 0 else "lower"
                    st.markdown(f"<div style='font-size:14px; padding:8px; background:#f0f0f0; border-radius:4px'>"
                              f"Mean NPRS is <b>{abs(mean_diff):.2f}</b> points {direction} when {factor} = Y"
                              f"</div>", unsafe_allow_html=True)
    
    # Range of motion factors
    st.subheader("Range of Motion Risk Factors")
    
    motion_factors = [col for col in df.columns if any(x in col for x in ['SHOULDER', 'ELBOW', 'WRIST'])]
    selected_motion = st.selectbox("Select motion factor", motion_factors)
    
    # Create bins for analysis
    filtered_df['motion_bin'] = pd.cut(filtered_df[selected_motion], bins=5)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=filtered_df, x='motion_bin', y='NPRS', palette='coolwarm')
    ax.set_title(f'Pain Scores by {selected_motion} Range', fontsize=14, pad=15)
    ax.set_xlabel(f'{selected_motion} Range (degrees)', fontsize=12)
    ax.set_ylabel('Pain Score (NPRS)', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.3)
    plt.xticks(rotation=45)
    st.pyplot(fig)
    
    # ANOVA test
    groups = [filtered_df[filtered_df['motion_bin'] == bin]['NPRS'] for bin in filtered_df['motion_bin'].unique()]
    f_stat, p_val = stats.f_oneway(*groups)
    
    st.markdown("**ANOVA Test Results:**")
    st.markdown(f"- **F-statistic:** `{f_stat:.3f}`")
    st.markdown(f"- **p-value:** `{p_val:.3f}`")
    
    if p_val < 0.05:
        st.success("Statistically significant differences exist between motion ranges (p < 0.05)")
        # Calculate effect size (eta squared)
        ss_between = sum(len(g) * (g.mean() - filtered_df['NPRS'].mean())**2 for g in groups)
        ss_total = sum((filtered_df['NPRS'] - filtered_df['NPRS'].mean())**2)
        eta_sq = ss_between / ss_total
        st.markdown(f"- **Effect size (η²):** `{eta_sq:.3f}`")
    else:
        st.warning("No statistically significant differences found (p ≥ 0.05)")

# Range of Motion Page
elif page == "💪 Range of Motion":
    st.header("💪 Range of Motion Analysis")
    
    # Select joint to analyze
    joint = st.radio("Select Joint", ['Shoulder', 'Elbow', 'Wrist'], horizontal=True)
    
    if joint == 'Shoulder':
        motions = ['SHOULDER_FLX', 'SHOULDER_EXTEN', 'SHOULDER_ABD', 'SHOULDER_ADD', 'SHOULDER_MEDIAL', 'SHOULDER_LATERAL']
    elif joint == 'Elbow':
        motions = ['ELBOW_FLEX', 'ELBOW_EXTENSION', 'ELBOW_SUPI', 'ELBOW_PORANA']
    else:
        motions = ['WRIST_FLEXN', 'WRIST_EXTEN', 'WRIST_ULNAR', 'WRIST_RADIAL']
    
    # Motion vs pain correlation matrix
    st.subheader(f"{joint} Motion Correlations with Pain")
    
    corr_matrix = filtered_df[motions + ['NPRS']].corr()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax,
               annot_kws={"size": 10}, fmt=".2f")
    ax.set_title(f'Correlation Between {joint} Motions and Pain Score', fontsize=14, pad=15)
    st.pyplot(fig)
    
    # Detailed motion analysis
    st.subheader("Detailed Motion Analysis")
    
    selected_motion = st.selectbox("Select specific motion", motions)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Motion Distribution**")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(filtered_df[selected_motion], kde=True, color='#4CAF50', bins=15)
        ax.set_xlabel('Degrees of Motion', fontsize=10)
        ax.set_title(f'Distribution of {selected_motion}', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.3)
        st.pyplot(fig)
    
    with col2:
        st.markdown("**Motion vs Pain**")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.scatterplot(data=filtered_df, x=selected_motion, y='NPRS', 
                       alpha=0.6, color='#2196F3', size=filtered_df['AGE'],
                       sizes=(20, 200), hue=filtered_df['GENDER'])
        ax.set_xlabel('Degrees of Motion', fontsize=10)
        ax.set_ylabel('Pain Score (NPRS)', fontsize=10)
        ax.set_title(f'{selected_motion} vs Pain Score', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.legend(title='Gender', bbox_to_anchor=(1.05, 1), loc='upper left')
        st.pyplot(fig)
    
    # Motion statistics
    st.markdown("**Motion Statistics**")
    motion_stats = filtered_df[selected_motion].describe().to_frame()
    st.dataframe(motion_stats.style.format("{:.2f}").background_gradient(cmap='YlOrBr'))

# Statistical Tests Page
elif page == "📈 Statistical Tests":
    st.header("📈 Statistical Analysis")
    
    # T-test for binary factors
    st.subheader("T-tests for Binary Factors")
    binary_factor = st.selectbox("Select binary factor", 
                               ['GLOVES', 'PAIN_KILLER', 'DOCTOR_CONSULTED', 'STRECTH', 'TRAINING'])
    
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
        st.dataframe(stats_df.style.format("{:.2f}").background_gradient(cmap='YlOrBr'))
        
        # Perform t-test
        t_stat, p_val = stats.ttest_ind(group1, group2)
        
        st.markdown("**Independent Samples T-test Results**")
        st.markdown(f"- **t-statistic:** `{t_stat:.3f}`")
        st.markdown(f"- **p-value:** `{p_val:.3f}`")
        
        if p_val < 0.05:
            st.success("Statistically significant difference between groups (p < 0.05)")
            # Calculate Cohen's d
            pooled_std = np.sqrt((group1.std()**2 + group2.std()**2)/2)
            cohen_d = abs((group1.mean() - group2.mean()) / pooled_std)
            st.markdown(f"- **Effect Size (Cohen's d):** `{cohen_d:.2f}`")
            
            if cohen_d > 0.8:
                st.info("Large effect size")
            elif cohen_d > 0.5:
                st.info("Medium effect size")
            else:
                st.info("Small effect size")
        else:
            st.warning("No statistically significant difference (p ≥ 0.05)")
    else:
        st.warning("Selected factor doesn't have exactly 2 groups for t-test")
    
    # ANOVA for continuous factors
    st.subheader("ANOVA for Continuous Factors")
    continuous_factor = st.selectbox("Select continuous factor", 
                                   ['AGE', 'SHOULDER_FLX', 'SHOULDER_ABD', 'ELBOW_FLEX', 'WRIST_FLEXN'])
    
    # Create bins
    filtered_df['factor_bin'] = pd.cut(filtered_df[continuous_factor], bins=5)
    
    # Display bin statistics
    st.markdown(f"**Binned Statistics for {continuous_factor}**")
    bin_stats = filtered_df.groupby('factor_bin')['NPRS'].agg(['mean', 'std', 'count'])
    st.dataframe(bin_stats.style.format("{:.2f}").background_gradient(cmap='YlOrBr'))
    
    # Perform ANOVA
    groups = [filtered_df[filtered_df['factor_bin'] == bin]['NPRS'] for bin in filtered_df['factor_bin'].unique()]
    f_stat, p_val = stats.f_oneway(*groups)
    
    st.markdown("**ANOVA Results**")
    st.markdown(f"- **F-statistic:** `{f_stat:.3f}`")
    st.markdown(f"- **p-value:** `{p_val:.3f}`")
    
    if p_val < 0.05:
        st.success("Statistically significant differences between groups (p < 0.05)")
        # Calculate eta squared
        ss_between = sum(len(g) * (g.mean() - filtered_df['NPRS'].mean())**2 for g in groups)
        ss_total = sum((filtered_df['NPRS'] - filtered_df['NPRS'].mean())**2)
        eta_sq = ss_between / ss_total
        st.markdown(f"- **Effect Size (η²):** `{eta_sq:.3f}`")
    else:
        st.warning("No statistically significant differences (p ≥ 0.05)")

# Recommendations Page
elif page == "🛡️ Recommendations":
    st.header("🛡️ Preventive Measures & Recommendations")
    
    # Key findings
    st.subheader("Key Findings Summary")
    
    findings = [
        ("1. High Pain Prevalence", 
         f"{len(filtered_df[filtered_df['NPRS'] >= 5])/len(filtered_df)*100:.1f}% of farmers report moderate to severe pain (NPRS ≥5)",
         "#FF6B6B"),
        ("2. Protective Effect of Gloves", 
         f"Glove users report {filtered_df[filtered_df['GLOVES']=='Y']['NPRS'].mean():.1f} vs {filtered_df[filtered_df['GLOVES']=='N']['NPRS'].mean():.1f} average pain for non-users",
         "#4ECDC4"),
        ("3. Shoulder Mobility Issues", 
         f"Shoulder flexion shows {filtered_df['SHOULDER_FLX'].corr(filtered_df['NPRS']):.2f} correlation with pain scores",
         "#45B7D1"),
        ("4. Pain Medication Use", 
         f"{filtered_df['PAIN_KILLER'].value_counts(normalize=True).get('Y', 0)*100:.1f}% use pain killers regularly",
         "#FFA07A"),
        ("5. Low Doctor Consultations", 
         f"Only {filtered_df['DOCTOR_CONSULTED'].value_counts(normalize=True).get('Y', 0)*100:.1f}% have consulted a doctor",
         "#98D8C8")
    ]
    
    cols = st.columns(5)
    for i, (title, content, color) in enumerate(findings):
        with cols[i]:
            st.markdown(f"""
            <div style="
                padding: 15px;
                border-radius: 10px;
                background-color: {color};
                color: white;
                height: 180px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            ">
                <h4>{title}</h4>
                <p>{content}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Recommendations
    st.subheader("Evidence-Based Recommendations")
    
    recommendations = [
        ("👐 Promote Glove Usage", 
         "Farmers not using gloves report higher pain scores. Provide ergonomic gloves to reduce strain.",
         ["Immediate distribution of gloves", "Training on proper glove use", "Regular replacement program"]),
        
        ("🏋️ Shoulder Strengthening", 
         "Targeted exercises to improve shoulder mobility may reduce pain based on correlation analysis.",
         ["Daily stretching routines", "Resistance band exercises", "Posture correction training"]),
        
        ("💊 Reduce Pain Killer Reliance", 
         "High usage suggests underlying issues need addressing rather than symptomatic treatment.",
         ["Alternative pain management", "Physical therapy referrals", "Regular health check-ups"]),
        
        ("🩺 Encourage Medical Consultations", 
         "Low consultation rates indicate potential under-treatment of chronic issues.",
         ["Mobile health clinics", "Awareness campaigns", "Subsidized medical care"]),
        
        ("🧘 Implement Stretching Programs", 
         "Regular stretching may help prevent repetitive strain injuries.",
         ["Pre-shift warm-ups", "Scheduled stretch breaks", "Supervisor-led routines"]),
        
        ("⏱️ Work Pattern Optimization", 
         "Adjust work hours and breaks based on pain patterns observed.",
         ["Shorter work intervals", "Mandatory rest periods", "Job rotation system"])
    ]
    
    for title, content, actions in recommendations:
        with st.expander(f"### {title}"):
            st.markdown(f"**Rationale:** {content}")
            st.markdown("**Implementation Actions:**")
            for action in actions:
                st.markdown(f"- {action}")
    
    # Implementation plan
    st.subheader("Suggested Implementation Timeline")
    
    timeline = [
        ("Immediate (0-1 month)", 
         ["Distribute ergonomic gloves", "Conduct pain awareness workshops", "Initiate pilot stretching program"]),
        
        ("Short-term (1-3 months)", 
         ["Implement on-site stretching programs", "Arrange regular medical check-ups", "Train supervisors in ergonomics"]),
        
        ("Medium-term (3-6 months)", 
         ["Develop farmer-specific exercise programs", "Establish health monitoring system", "Evaluate program effectiveness"]),
        
        ("Long-term (6+ months)", 
         ["Institutionalize preventive measures", "Expand to other farming communities", "Continuous improvement process"])
    ]
    
    for phase, tasks in timeline:
        with st.expander(f"**{phase}**"):
            for task in tasks:
                st.markdown(f"- {task}")

# Work Patterns Page
elif page == "⏱️ Work Patterns":
    st.header("⏱️ Work Pattern Analysis")
    
    # Experience analysis
    st.subheader("Experience Level Analysis")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=filtered_df, x='EXPERIENCE_LABEL', y='NPRS', 
                order=['0-1 years', '1-5 years', '6-10 years', '10+ years'],
                palette='viridis')
    ax.set_title('Pain Scores by Farming Experience', fontsize=14, pad=15)
    ax.set_xlabel('Experience Level', fontsize=12)
    ax.set_ylabel('Pain Score (NPRS)', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.3)
    st.pyplot(fig)
    
    # Hours per day analysis
    st.subheader("Working Hours Analysis")
    
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.countplot(data=filtered_df, x='HOURS_DAY_LABEL', 
                     order=['<4 hrs', '4-6 hrs', '7-8 hrs', '9+ hrs'],
                     palette='Blues')
        ax.set_title('Distribution of Daily Working Hours', fontsize=14, pad=15)
        ax.set_xlabel('Hours per Day', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.3)
        st.pyplot(fig)
    
    with col2:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(data=filtered_df, x='HOURS_DAY_LABEL', y='NPRS',
                   order=['<4 hrs', '4-6 hrs', '7-8 hrs', '9+ hrs'],
                   palette='Blues')
        ax.set_title('Pain Scores by Daily Working Hours', fontsize=14, pad=15)
        ax.set_xlabel('Hours per Day', fontsize=12)
        ax.set_ylabel('Pain Score (NPRS)', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.3)
        st.pyplot(fig)
    
    # Break duration analysis
    st.subheader("Break Duration Analysis")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=filtered_df, x='HOW_LONG_BREAK_LABEL', y='NPRS',
               order=['<5 min', '5-10 min', '10+ min'],
               palette='coolwarm')
    ax.set_title('Pain Scores by Break Duration', fontsize=14, pad=15)
    ax.set_xlabel('Break Duration', fontsize=12)
    ax.set_ylabel('Pain Score (NPRS)', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.3)
    st.pyplot(fig)
    
    # Affected area analysis
    st.subheader("Most Affected Area Analysis")
    
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.countplot(data=filtered_df, x='AREA_LABEL', 
                     order=['Shoulder', 'Elbow', 'Wrist', 'Multiple'],
                     palette='Set2')
        ax.set_title('Distribution of Affected Areas', fontsize=14, pad=15)
        ax.set_xlabel('Affected Area', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.3)
        st.pyplot(fig)
    
    with col2:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(data=filtered_df, x='AREA_LABEL', y='NPRS',
                   order=['Shoulder', 'Elbow', 'Wrist', 'Multiple'],
                   palette='Set2')
        ax.set_title('Pain Scores by Affected Area', fontsize=14, pad=15)
        ax.set_xlabel('Affected Area', fontsize=12)
        ax.set_ylabel('Pain Score (NPRS)', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.3)
        st.pyplot(fig)
    
    # Pain frequency analysis
    st.subheader("Pain Frequency Analysis")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.countplot(data=filtered_df, x='FREQUENTLY_LABEL', 
                 order=['Daily', 'Weekly', 'Occasionally', 'Never'],
                 palette='magma')
    ax.set_title('Distribution of Pain Frequency', fontsize=14, pad=15)
    ax.set_xlabel('Pain Frequency', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.3)
    st.pyplot(fig)
    
    # Stiffness and numbness analysis
    st.subheader("Stiffness and Numbness Analysis")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Joint Stiffness (STIFFNESS)**")
        stiffness_counts = filtered_df['STIFFNESS'].value_counts()
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.pie(stiffness_counts, labels=['Yes', 'No'], autopct='%1.1f%%',
               colors=['#ff9999','#66b3ff'], startangle=90,
               wedgeprops={'linewidth': 1, 'edgecolor': 'white'})
        ax.set_title('Percentage with Joint Stiffness', fontsize=14, pad=15)
        st.pyplot(fig)
    
    with col2:
        st.markdown("**Hand Numbness (NUMBNESS)**")
        numbness_counts = filtered_df['NUMBNESS'].value_counts()
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.pie(numbness_counts, labels=['Yes', 'No'], autopct='%1.1f%%',
               colors=['#ff9999','#66b3ff'], startangle=90,
               wedgeprops={'linewidth': 1, 'edgecolor': 'white'})
        ax.set_title('Percentage with Hand Numbness', fontsize=14, pad=15)
        st.pyplot(fig)
    
    # Statistical test for affected area
    st.subheader("Statistical Analysis for Affected Area")
    
    # Perform ANOVA
    groups = [filtered_df[filtered_df['AREA_LABEL'] == area]['NPRS'] for area in filtered_df['AREA_LABEL'].unique()]
    f_stat, p_val = stats.f_oneway(*groups)
    
    st.markdown("**ANOVA Results for Affected Area vs Pain Score:**")
    st.markdown(f"- **F-statistic:** `{f_stat:.3f}`")
    st.markdown(f"- **p-value:** `{p_val:.3f}`")
    
    if p_val < 0.05:
        st.success("Statistically significant differences exist between affected areas (p < 0.05)")
        # Calculate eta squared
        ss_between = sum(len(g) * (g.mean() - filtered_df['NPRS'].mean())**2 for g in groups)
        ss_total = sum((filtered_df['NPRS'] - filtered_df['NPRS'].mean())**2)
        eta_sq = ss_between / ss_total
        st.markdown(f"- **Effect Size (η²):** `{eta_sq:.3f}`")
        
        # Post-hoc tests
        st.markdown("**Post-hoc Tukey HSD Test Results:**")
        from statsmodels.stats.multicomp import pairwise_tukeyhsd
        tukey = pairwise_tukeyhsd(endog=filtered_df['NPRS'],
                                groups=filtered_df['AREA_LABEL'],
                                alpha=0.05)
        st.text(str(tukey))
    else:
        st.warning("No statistically significant differences found (p ≥ 0.05)")

# Footer
st.markdown("---")
st.markdown("""
<div style="
    padding: 15px;
    background-color: #f0f2f6;
    border-radius: 5px;
    margin-top: 20px;
">
    <h4>About This Dashboard</h4>
    <p>This interactive tool analyzes upper extremity injuries in farmers using spades, with objectives to:</p>
    <ol>
        <li>Assess pain frequency and severity</li>
        <li>Identify key risk factors</li>
        <li>Evaluate range of motion limitations</li>
        <li>Suggest evidence-based preventive measures</li>
    </ol>
    <p><i>Data collected from {n} farmers across multiple regions.</i></p>
</div>
""".format(n=len(df)), unsafe_allow_html=True)
