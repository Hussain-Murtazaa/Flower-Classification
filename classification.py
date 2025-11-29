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
    layout="wide",
    initial_sidebar_state="expanded"
)

# Beautiful CSS Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Poppins', sans-serif;
    }
    
    body {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    .main {
        background: transparent;
    }
    
    /* Header styling */
    h1 {
        font-size: 48px !important;
        font-weight: 700 !important;
        margin-bottom: 10px !important;
        color: #667eea !important;
    }
    
    h2, h3 {
        color: #667eea !important;
        font-weight: 600 !important;
    }
    
    /* Sidebar */
    .stSidebar {
        background: rgba(102, 126, 234, 0.1) !important;
        backdrop-filter: blur(10px);
        border-right: 2px solid rgba(118, 75, 162, 0.3);
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea, #764ba2) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 12px 24px !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.3) !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 12px 30px rgba(102, 126, 234, 0.5) !important;
    }
    
    /* Sliders */
    .stSlider > div > div > div {
        background: linear-gradient(90deg, #667eea, #764ba2) !important;
    }
    
    /* Success box */
    .stSuccess {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15), rgba(52, 211, 153, 0.1)) !important;
        border-left: 4px solid #10b981 !important;
        border-radius: 10px !important;
        padding: 20px !important;
    }
    
    /* Expandable sections */
    .streamlit-expanderHeader {
        background: linear-gradient(90deg, rgba(102, 126, 234, 0.1), rgba(118, 75, 162, 0.1)) !important;
        border-radius: 12px !important;
        border: 1px solid rgba(118, 75, 162, 0.2) !important;
    }
    
    /* Metric boxes */
    .stMetric {
        background: rgba(102, 126, 234, 0.1) !important;
        backdrop-filter: blur(10px);
        border-radius: 12px !important;
        border: 1px solid rgba(118, 75, 162, 0.2) !important;
        padding: 15px !important;
    }
    
    /* Download button */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #10b981, #34d399) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        width: 100% !important;
    }
    
    /* Dataframe */
    .stDataFrame {
        border-radius: 12px !important;
        border: 1px solid rgba(118, 75, 162, 0.2) !important;
    }
</style>
""", unsafe_allow_html=True)

# Title with subtitle
col1, col2 = st.columns([3, 1])
with col1:
    st.title("🌺 Iris Flower Species Classifier")
    st.markdown("*Predict iris species using machine learning* 🤖")

with col2:
    st.markdown("")
    st.markdown("### ML Model: 🌿")
    st.markdown("**Random Forest**")

st.markdown("---")

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
    st.header("📏 Flower Measurements")
    st.markdown("*Adjust the sliders to predict*")
    st.markdown("---")
    
    sepal_length = st.slider("🌷 Sepal Length (cm)", 
                             float(df['sepal length (cm)'].min()), 
                             float(df['sepal length (cm)'].max()), 
                             float(df['sepal length (cm)'].mean()))
    
    sepal_width = st.slider("🌷 Sepal Width (cm)", 
                            float(df['sepal width (cm)'].min()), 
                            float(df['sepal width (cm)'].max()), 
                            float(df['sepal width (cm)'].mean()))
    
    petal_length = st.slider("🌹 Petal Length (cm)", 
                             float(df['petal length (cm)'].min()), 
                             float(df['petal length (cm)'].max()), 
                             float(df['petal length (cm)'].mean()))
    
    petal_width = st.slider("🌹 Petal Width (cm)", 
                            float(df['petal width (cm)'].min()), 
                            float(df['petal width (cm)'].max()), 
                            float(df['petal width (cm)'].mean()))
    
    st.markdown("---")
    predict = st.button("🔮 Predict Species", use_container_width=True, type="primary")

# Main content area
if predict:
    input_data = [[sepal_length, sepal_width, petal_length, petal_width]]
    pred = model.predict(input_data)[0]
    probs = model.predict_proba(input_data)[0]
    species = target_names[pred]

    # Success message
    st.success(f"✨ **Predicted Species: {species}** ✨", icon="🌸")
    
    # Confidence scores in columns
    st.subheader("📊 Confidence Scores")
    
    cols = st.columns(3)
    for i, name in enumerate(target_names):
        with cols[i]:
            st.metric(name, f"{probs[i]:.1%}")
    
    # Visualization
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Probability Distribution")
        prob_df = pd.DataFrame({"Species": target_names, "Probability": probs})
        st.bar_chart(prob_df.set_index("Species"))
    
    with col2:
        st.markdown("### Input Measurements")
        input_df = pd.DataFrame({
            "Measurement": ["Sepal Length", "Sepal Width", "Petal Length", "Petal Width"],
            "Value (cm)": [sepal_length, sepal_width, petal_length, petal_width]
        })
        st.table(input_df)
    
    # Download prediction
    with st.expander("💾 Download This Prediction"):
        log = pd.DataFrame({
            "Sepal Length": [sepal_length],
            "Sepal Width": [sepal_width],
            "Petal Length": [petal_length],
            "Petal Width": [petal_width],
            "Prediction": [species]
        })
        csv = log.to_csv(index=False).encode()
        st.download_button("📥 Download as CSV", csv, "iris_prediction.csv", "text/csv", use_container_width=True)

# Information Sections
st.markdown("---")
st.subheader("📚 Model Information & Analysis")

col1, col2 = st.columns(2)

with col1:
    with st.expander("🌿 View Full Iris Dataset", expanded=False):
        st.dataframe(df, use_container_width=True)
    
    with st.expander("⚖️ Feature Importance", expanded=False):
        imp = pd.DataFrame({
            "Feature": df.columns[:-1],
            "Importance": model.feature_importances_
        }).sort_values("Importance", ascending=True)
        fig, ax = plt.subplots()
        ax.barh(imp["Feature"], imp["Importance"], color="#667eea")
        ax.set_xlabel("Importance Score")
        st.pyplot(fig)
    
    with st.expander("🎯 Confusion Matrix", expanded=False):
        cm = confusion_matrix(df['Species'], model.predict(df.iloc[:, :-1]))
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="RdPu", 
                    xticklabels=target_names, yticklabels=target_names, ax=ax, cbar_kws={"label": "Count"})
        ax.set_xlabel("Predicted", fontweight="bold")
        ax.set_ylabel("Actual", fontweight="bold")
        ax.set_title("Confusion Matrix", fontweight="bold", pad=20)
        st.pyplot(fig)

with col2:
    with st.expander("🧭 Why These Features Matter", expanded=False):
        st.markdown("""
        ✨ **Key Insights:**
        
        - **Petal features** (length & width) are the strongest signals — species separate cleanly here
        
        - **Sepal dimensions** overlap a lot, so the model trusts them less
        
        - 🎯 **The Famous "Petal Rule"**: Random Forest automatically discovered the botanists' rule!
        
        - 📈 Model learns that petal measurements alone can classify most species correctly
        """)
    
    with st.expander("📈 ROC Curves (One-vs-Rest)", expanded=False):
        lb = LabelBinarizer()
        y_bin = lb.fit_transform(df["Species"])
        y_prob = model.predict_proba(df.iloc[:, :-1])

        fig, ax = plt.subplots(figsize=(8, 6))
        colors = ['#667eea', '#764ba2', '#f093fb']
        for i, name in enumerate(target_names):
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.3f})", linewidth=2.5, color=colors[i])
        ax.plot([0,1],[0,1],"k--", linewidth=2, label="Random Classifier")
        ax.set_xlabel("False Positive Rate", fontweight="bold")
        ax.set_ylabel("True Positive Rate", fontweight="bold")
        ax.set_title("ROC Curves", fontweight="bold", pad=20)
        ax.legend(loc="lower right")
        ax.grid(alpha=0.3)
        st.pyplot(fig)
    
    with st.expander("🏆 Model Performance Summary", expanded=False):
        y_pred = model.predict(df.iloc[:, :-1])
        accuracy = accuracy_score(df['Species'], y_pred)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("🎯 Accuracy", f"{accuracy:.1%}")
        with col2:
            st.metric("📊 Total Samples", len(df))
        
        st.markdown("**Classification Report:**")
        st.text(classification_report(df['Species'], y_pred, target_names=target_names))

st.markdown("---")
st.caption("🌺 Built with ❤️ using Streamlit • Model: Random Forest • Dataset: Iris Flowers")
