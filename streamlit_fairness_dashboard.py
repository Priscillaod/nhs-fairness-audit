# NHS Fairness Audit Dashboard (Streamlit)


# Step 1: Import libraries
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Step 2: Define your fairness metrics (replace these with your real results)
data = {
    "Protected Attribute": ["Gender", "Ethnicity", "IMD Quintile"],
    "Demographic Parity Difference (LR)": [0.250, 0.924, 0.529],
    "Equalised Odds Difference (LR)": [0.284, 0.952, 0.559],
    "Demographic Parity Difference (XGB)": [0.046, 0.554, 0.239],
    "Equalised Odds Difference (XGB)": [0.127, 0.667, 0.305]
}
df = pd.DataFrame(data)

# Step 3: Add sidebar for model selection
st.sidebar.title(" Fairness Audit Controls")
model_choice = st.sidebar.radio("Select Model:", ["Logistic Regression", "XGBoost"])

# Step 4: Add main dashboard title and introduction
st.title("NHS Fairness Audit Dashboard")
st.write("""
This interactive dashboard visualises bias and fairness results 
for NHS-style predictive models across key protected attributes 
(Gender, Ethnicity, and IMD Quintile).
""")

# Step 5: Display fairness summary table
st.subheader(f" Fairness Metrics — {model_choice}")
if model_choice == "Logistic Regression":
    st.dataframe(df[["Protected Attribute", 
                     "Demographic Parity Difference (LR)", 
                     "Equalised Odds Difference (LR)"]])
else:
    st.dataframe(df[["Protected Attribute", 
                     "Demographic Parity Difference (XGB)", 
                     "Equalised Odds Difference (XGB)"]])

# Step 6: Add dropdown to select one protected attribute
selected_attr = st.selectbox("Select Protected Attribute:", df["Protected Attribute"])

# Step 7: Display key metrics (like dashboard cards)
row = df[df["Protected Attribute"] == selected_attr].iloc[0]
if model_choice == "Logistic Regression":
    st.metric("Demographic Parity Difference", round(row["Demographic Parity Difference (LR)"], 3))
    st.metric("Equalised Odds Difference", round(row["Equalised Odds Difference (LR)"], 3))
else:
    st.metric("Demographic Parity Difference", round(row["Demographic Parity Difference (XGB)"], 3))
    st.metric("Equalised Odds Difference", round(row["Equalised Odds Difference (XGB)"], 3))

# Step 8: Example fairness chart (static demo)
st.subheader("Example Group Fairness Chart")
metrics = ["Accuracy", "Precision", "Recall", "F1"]
values = [0.48, 0.36, 0.79, 0.45]  # replace with your group-wise metrics

fig, ax = plt.subplots()
ax.bar(metrics, values, color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"])
ax.set_ylim(0, 1)
ax.set_ylabel("Metric Value")
ax.set_title(f"{model_choice} - Example Group Fairness ({selected_attr})")
st.pyplot(fig)

# Step 9: Explanatory text
st.info(f"""
For {selected_attr}, the {model_choice} model shows variation in performance metrics.
Lower Demographic Parity and Equalised Odds differences indicate improved fairness alignment.
""")

