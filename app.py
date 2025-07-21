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
    # Load the dataset from CSV
    df = pd.read_csv('farmers_data.csv')  # Assuming the data is saved as CSV
    return df

df = load_data()

# Title
st.title("🌾 Prevalence & Risk Factors of Upper Extremities Injury in Farmers Using Spades")

# Sidebar filters
st.sidebar.header("Filter Data")
show_raw_data = st.sidebar.checkbox("Show raw data")

# Data overview
st.header("📊 Data Overview")
col1, col2, col3 = st.columns(3)
col1.metric("Total Farmers", len(df))
col2.metric("Average Pain Score (NPRS)", round(df['NPRS'].mean(), 2))
col3.metric("Farmers Using Pain Killers", f"{round(df['PAIN_KILLER'].value_counts(normalize=True)['Y']*100, 1)}%")

if show_raw_data:
    st.subheader("Raw Data")
    st.dataframe(df)

# Pain distribution analysis
st.header("🩹 Pain Distribution Analysis")
fig, ax = plt.subplots(figsize=(10, 6))
sns.histplot(data=df, x='NPRS', bins=10, kde=True, ax=ax)
ax.set_title('Distribution of Pain Scores (NPRS)')
ax.set_xlabel('Numeric Pain Rating Scale (0-10)')
ax.set_ylabel('Number of Farmers')
st.pyplot(fig)

# Pain score statistics
st.subheader("Pain Score Statistics")
pain_stats = df['NPRS'].describe().to_frame().T
st.dataframe(pain_stats.style.format("{:.2f}"))

# Risk factor analysis
st.header("⚠️ Risk Factor Analysis")

# Binary factors
binary_factors = ['STRECTH', 'GLOVES', 'DOCTOR_CONSULTED', 'PAIN_KILLER']
selected_factor = st.selectbox("Select risk factor to analyze", binary_factors)

fig, ax = plt.subplots(figsize=(8, 6))
sns.boxplot(data=df, x=selected_factor, y='NPRS', ax=ax)
ax.set_title(f'Pain Score Distribution by {selected_factor}')
st.pyplot(fig)

# Calculate statistics for selected factor
st.subheader(f"Statistical Analysis for {selected_factor}")
grouped_stats = df.groupby(selected_factor)['NPRS'].agg(['mean', 'median', 'std', 'count'])
st.dataframe(grouped_stats.style.format("{:.2f}"))

# T-test for binary factors
if len(df[selected_factor].unique()) == 2:
    group1 = df[df[selected_factor] == df[selected_factor].unique()[0]]['NPRS']
    group2 = df[df[selected_factor] == df[selected_factor].unique()[1]]['NPRS']
    t_stat, p_val = stats.ttest_ind(group1, group2)
    
    st.write(f"**Independent t-test results:**")
    st.write(f"- t-statistic: {t_stat:.3f}")
    st.write(f"- p-value: {p_val:.3f}")
    if p_val < 0.05:
        st.success("The difference in pain scores between groups is statistically significant (p < 0.05)")
    else:
        st.warning("The difference in pain scores between groups is not statistically significant (p ≥ 0.05)")

# Range of motion analysis
st.header("💪 Range of Motion Analysis")

# Select motion to analyze
motion_cols = [col for col in df.columns if 'SHOULDER' in col or 'ELBOW' in col or 'WRIST' in col]
selected_motion = st.selectbox("Select motion to analyze", motion_cols)

# Plot motion vs pain
fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(data=df, x=selected_motion, y='NPRS', ax=ax)
ax.set_title(f'{selected_motion} vs Pain Score')
ax.set_xlabel('Range of Motion (degrees)')
ax.set_ylabel('Pain Score (NPRS)')
st.pyplot(fig)

# Correlation analysis
corr = df[selected_motion].corr(df['NPRS'])
st.write(f"**Correlation between {selected_motion} and pain score:** {corr:.2f}")

if abs(corr) > 0.3:
    st.info("There appears to be a moderate correlation between this motion and pain score.")
elif abs(corr) > 0.5:
    st.info("There appears to be a strong correlation between this motion and pain score.")
else:
    st.info("There appears to be little to no correlation between this motion and pain score.")

# ANOVA for multiple groups
st.header("📈 Analysis of Variance (ANOVA)")

# Create bins for range of motion
df['motion_bin'] = pd.cut(df[selected_motion], bins=5)
fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(data=df, x='motion_bin', y='NPRS', ax=ax)
ax.set_title(f'Pain Scores by {selected_motion} Range')
ax.set_xlabel(f'{selected_motion} Range (degrees)')
ax.set_ylabel('Pain Score (NPRS)')
plt.xticks(rotation=45)
st.pyplot(fig)

# Perform ANOVA
groups = [df[df['motion_bin'] == bin]['NPRS'] for bin in df['motion_bin'].unique()]
f_stat, p_val = stats.f_oneway(*groups)

st.write(f"**ANOVA results for {selected_motion}:**")
st.write(f"- F-statistic: {f_stat:.3f}")
st.write(f"- p-value: {p_val:.3f}")
if p_val < 0.05:
    st.success("There are statistically significant differences in pain scores between different ranges of motion (p < 0.05)")
else:
    st.warning("No statistically significant differences found between different ranges of motion (p ≥ 0.05)")

# Preventive measures suggestions
st.header("🛡️ Preventive Measures Suggestions")

if selected_factor == 'GLOVES' and 'N' in df[selected_factor].unique():
    st.info("**Recommendation:** Encourage glove use as farmers not using gloves appear to have higher pain scores.")

if 'SHOULDER' in selected_motion and corr < -0.2:
    st.info(f"**Recommendation:** Farmers with limited {selected_motion} range show higher pain scores. Consider exercises to improve mobility.")

if 'PAIN_KILLER' in binary_factors and 'Y' in df['PAIN_KILLER'].unique():
    usage_rate = df['PAIN_KILLER'].value_counts(normalize=True)['Y']
    if usage_rate > 0.5:
        st.warning(f"High pain killer usage ({usage_rate:.0%}). Investigate root causes of pain rather than relying on medication.")

# Conclusion
st.header("🎯 Key Findings")
st.write("""
1. **Pain Prevalence:** The average pain score among farmers is {:.1f}/10, indicating moderate discomfort.
2. **Risk Factors:** {} appears to be significantly associated with higher pain scores.
3. **Range of Motion:** {} shows {} correlation with pain levels.
4. **Protective Factors:** Farmers who {} tend to report lower pain scores.
""".format(
    df['NPRS'].mean(),
    selected_factor,
    selected_motion,
    "a moderate" if 0.3 <= abs(corr) < 0.5 else "a strong" if abs(corr) >= 0.5 else "little",
    "use gloves" if selected_factor == 'GLOVES' else "stretch regularly" if selected_factor == 'STRECTH' else "consult doctors"
))

# Footer
st.markdown("---")
st.markdown("### About This Dashboard")
st.markdown("""
This interactive dashboard analyzes upper extremity injuries in farmers using spades, with the following objectives:
1. Assess frequency of pain and discomfort
2. Identify major risk factors
3. Suggest preventive measures
""")
