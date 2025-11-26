import streamlit as st # user interface front end 
import pandas as pd  # to create data in DataFrame
from sklearn.datasets import load_iris # importing iris dataset from scikit-learn
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(
    page_title="Iris Flower Classifier",
    page_icon="🌸",
    layout="centered",   # or "wide"
    initial_sidebar_state="expanded"
)



st.title("🌸 Iris Flower Species Classifier")
st.write("Adjust the sliders in the sidebar to predict the iris flower species.")

@st.cache_data
def load_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data,columns=iris.feature_names)
    df['Species'] = iris.target
    return df,iris.target_names

df, target_names = load_data()

#Train model
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


model = RandomForestClassifier(
    n_estimators=200,
    max_depth=5,
    random_state=42
)


model.fit(df.iloc[:,:-1],df['Species'])

with st.sidebar.form("input_form"):
    st.subheader("Input Features")
    sepal_length = st.slider("Sepal Length", *df['sepal length (cm)'].agg(['min','max']).astype(float))
    sepal_width  = st.slider("Sepal Width",  *df['sepal width (cm)'].agg(['min','max']).astype(float))
    petal_length = st.slider("Petal Length", *df['petal length (cm)'].agg(['min','max']).astype(float))
    petal_width  = st.slider("Petal Width",  *df['petal width (cm)'].agg(['min','max']).astype(float))

    submitted = st.form_submit_button("Predict")

    input_data = [[sepal_length, sepal_width, petal_length, petal_width]]
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


with st.expander("📊 View Iris Dataset"):
    st.dataframe(df)
with st.expander("🔍 Feature Importance"):
    importances = model.feature_importances_
    feat_df = pd.DataFrame({
        "Feature": df.columns[:-1],
        "Importance": importances
    }).sort_values("Importance")

    st.bar_chart(feat_df.set_index("Feature"))


# prediction

prediction = model.predict(input_data)

predicted_species = target_names[prediction[0]]

st.markdown("---")
st.caption("Built with ❤️ using Streamlit By Hussain-Murtazaa")