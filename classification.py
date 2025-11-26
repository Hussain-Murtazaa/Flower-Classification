import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report, accuracy_score
from sklearn.preprocessing import LabelBinarizer
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Iris Flower Classifier",
    page_icon="🌸",
    layout="centered",
    initial_sidebar_state="expanded"
)

st.title("🌸 Iris Flower Species Classifier")
st.write("Adjust the sliders in the sidebar to predict the iris flower species.")

# ---------------------------
# Load Data
# ---------------------------
@st.cache_data
def load_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['Species'] = iris.target
    return df, iris.target_names

df, target_names = load_data()

# ---------------------------
# Train Model
# ---------------------------
@st.cache_resource
def train_model(df):
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=5,
        random_state=42
    )
    model.fit(df.iloc[:, :-1], df["Species"])
    return model

model = train_model(df)

# ---------------------------
# Sidebar Inputs
# ---------------------------
with st.sidebar.form("input_form"):
    st.subheader("Input Features")
    sepal_length = st.slider("Sepal Length", *df['sepal length (cm)'].agg(['min', 'max']).astype(float))
    sepal_width = st.slider("Sepal Width", *df['sepal width (cm)'].agg(['min', 'max']).astype(float))
    petal_length = st.slider("Petal Length", *df['petal length (cm)'].agg(['min', 'max']).astype(float))
    petal_width = st.slider("Petal Width", *df['petal width (cm)'].agg(['min', 'max']).astype(float))

    submitted = st.form_submit_button("Predict")

# ---------------------------
# Prediction Logic
# ---------------------------
if submitted:
    input_data = [[sepal_length, sepal_width, petal_length, petal_width]]
    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]
    predicted_species = target_names[prediction]

    st.success(f"🌼 Predicted Species: **{predicted_species}**")

    st.write("### Prediction Probabilities")
    prob_df = pd.DataFrame({
        "Species": target_names,
        "Probability": probabilities
    })
    st.bar_chart(prob_df.set_index("Species"))

    # Download Log
    with st.expander("📥 Download Prediction Log"):
        log_df = pd.DataFrame({
            "Sepal Length": [sepal_length],
            "Sepal Width": [sepal_width],
            "Petal Length": [petal_length],
            "Petal Width": [petal_width],
            "Predicted Species": [predicted_species]
        })

        csv = log_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", csv, "prediction_log.csv", "text/csv")

# ---------------------------
# Dataset View
# ---------------------------
with st.expander("📊 View Iris Dataset"):
    st.dataframe(df)

# ---------------------------
# Feature Importance
# ---------------------------
with st.expander("🔍 Feature Importance"):
    importances = model.feature_importances_
    feat_df = pd.DataFrame({
        "Feature": df.columns[:-1],
        "Importance": importances
    }).sort_values("Importance")
    st.bar_chart(feat_df.set_index("Feature"))

with st.expander("🧠 Feature Importance Explanation"):
    st.write("""
    - **Petal length** and **petal width** dominate because the three iris species differ mainly in petal structure.
    - **Sepal measurements** matter less because their ranges overlap across species.
    - The model learns boundaries mostly from petal features.
    """)

# ---------------------------
# Confusion Matrix
# ---------------------------
with st.expander("📉 Confusion Matrix"):
    cm = confusion_matrix(df['Species'], model.predict(df.iloc[:, :-1]))
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, cmap="Blues", fmt="d",
                xticklabels=target_names,
                yticklabels=target_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    st.pyplot(fig)

# ---------------------------
# ROC Curves
# ---------------------------
with st.expander("📈 ROC Curves"):
    lb = LabelBinarizer()
    y_true = lb.fit_transform(df["Species"])
    y_score = model.predict_proba(df.iloc[:, :-1])

    fig, ax = plt.subplots()
    for i, label in enumerate(target_names):
        fpr, tpr, _ = roc_curve(y_true[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{label} (AUC = {roc_auc:.2f})")

    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    st.pyplot(fig)

# ---------------------------
# Evaluation
# ---------------------------
with st.expander("📊 Model Evaluation"):
    y_pred = model.predict(df.iloc[:, :-1])
    accuracy = accuracy_score(df["Species"], y_pred)

    st.write(f"**Accuracy:** {accuracy:.3f}")
    st.text("Classification Report:")
    st.text(classification_report(df["Species"], y_pred, target_names=target_names))

st.markdown("---")
st.caption("Built with ❤️ using Streamlit By Hussain-Murtazaa")
