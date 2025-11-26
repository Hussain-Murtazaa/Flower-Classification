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
    page_icon="🌺",
    layout="centered",
    initial_sidebar_state="expanded"
)

st.title("🌸 Iris Flower Species Classifier")
st.write("Adjust the sliders to predict the species in real-time.")

# Load Data
@st.cache_data
def load_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['Species'] = iris.target
    return df, iris.target_names

df, target_names = load_data()

# Train Model
@st.cache_resource
def train_model(df):
    model = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42)
    model.fit(df.iloc[:, :-1], df["Species"])
    return model

model = train_model(df)

# Sidebar Inputs
with st.sidebar:
    st.header("📏 Input Flower Measurements")
    sepal_length = st.slider("Sepal Length (cm)", float(df['sepal length (cm)'].min()), float(df['sepal length (cm)'].max()), float(df['sepal length (cm)'].mean()))
    sepal_width  = st.slider("Sepal Width (cm)",  float(df['sepal width (cm)'].min()),  float(df['sepal width (cm)'].max()),  float(df['sepal width (cm)'].mean()))
    petal_length = st.slider("Petal Length (cm)", float(df['petal length (cm)'].min()), float(df['petal length (cm)'].max()), float(df['petal length (cm)'].mean()))
    petal_width  = st.slider("Petal Width (cm)",  float(df['petal width (cm)'].min()),  float(df['petal width (cm)'].max()),  float(df['petal width (cm)'].mean()))
    
    predict = st.button("🔮 Predict Species", type="primary")

# Prediction
if predict:
    input_data = [[sepal_length, sepal_width, petal_length, petal_width]]
    pred = model.predict(input_data)[0]
    probs = model.predict_proba(input_data)[0]
    species = target_names[pred]

    st.success(f"Predicted Species: **{species}** 🌷")

    st.subheader("Confidence Scores")
    prob_df = pd.DataFrame({"Species": target_names, "Probability": probs})
    st.bar_chart(prob_df.set_index("Species"))

    with st.expander("💾 Download This Prediction"):
        log = pd.DataFrame({
            "Sepal Length (cm)": [sepal_length],
            "Sepal Width (cm)":  [sepal_width],
            "Petal Length (cm)": [petal_length],
            "Petal Width (cm)":  [petal_width],
            "Prediction": [species]
        })
        csv = log.to_csv(index=False).encode()
        st.download_button("Download CSV", csv, "iris_prediction.csv", "text/csv")

# Expandable Sections (all unique emojis)
with st.expander("🌿 View Full Iris Dataset"):
    st.dataframe(df, use_container_width=True)

with st.expander("⚖️ Feature Importance"):
    imp = pd.DataFrame({
        "Feature": df.columns[:-1],
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=True)
    st.bar_chart(imp.set_index("Feature"))

with st.expander("🧭 Why These Features Matter"):
    st.write("""
    - Petal length & width are the strongest signals — species separate cleanly here  
    - Sepal dimensions overlap a lot, so the model trusts them less  
    - Random Forest automatically discovered the famous "petal rule" botanists use!
    """)

with st.expander("🎯 Confusion Matrix"):
    cm = confusion_matrix(df['Species'], model.predict(df.iloc[:, :-1]))
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=target_names, yticklabels=target_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

with st.expander("📈 ROC Curves (One-vs-Rest)"):
    lb = LabelBinarizer()
    y_bin = lb.fit_transform(df["Species"])
    y_prob = model.predict_proba(df.iloc[:, :-1])

    fig, ax = plt.subplots()
    for i, name in enumerate(target_names):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.3f})")
    ax.plot([0,1],[0,1],"k--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    st.pyplot(fig)

with st.expander("🏆 Model Performance Summary"):
    y_pred = model.predict(df.iloc[:, :-1])
    st.metric("Accuracy", f"{accuracy_score(df['Species'], y_pred):.3%}")
    st.text(classification_report(df['Species'], y_pred, target_names=target_names))

st.markdown("---")
st.caption("Built with ❤️ using Streamlit • Model: Random Forest")
