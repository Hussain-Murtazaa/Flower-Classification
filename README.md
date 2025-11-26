🌸 Iris Flower Species Classifier
A simple, interactive ML web app built with Streamlit

This project is a machine learning web application that predicts the species of an Iris flower based on four botanical features:

Sepal Length

Sepal Width

Petal Length

Petal Width

The app is powered by a Random Forest Classifier trained on the classic Iris dataset from scikit-learn.
Users can interact with the model in real-time using intuitive sliders in the sidebar.

🚀 Features
✔ Interactive UI

Built using Streamlit, offering a clean and responsive interface.

✔ Real-Time Predictions

As the user adjusts the sliders, the model predicts:

Setosa

Versicolor

Virginica

✔ Prediction Confidence

Displays a probability chart showing how confident the model is in its prediction.

✔ Feature Importance

Visualizes which features contribute most to the model’s decisions.

✔ Dataset Viewer

Built-in expandable section to browse the Iris dataset.

✔ Cached Model & Data

Prevents repeated computation and improves performance.

🧠 Machine Learning Model

The app uses a RandomForestClassifier with:

n_estimators=200
max_depth=5
random_state=42


These hyperparameters make the model more stable, accurate, and reproducible.

📦 Technologies Used
Tool / Library	Purpose
Python	Core language
Streamlit	Web UI framework
Pandas	Data handling
scikit-learn	ML model & dataset
Random Forest	Classification algorithm
🛠 Installation & Setup

Clone the repository:

git clone https://github.com/Huddsin-Murtazaa/Flower-Classification.git
cd your-repo-name


Install dependencies:

pip install -r requirements.txt


Run the Streamlit app:

streamlit run app.py

📁 Project Structure
.
├── app.py              # Main Streamlit application
├── README.md           # Project documentation
├── requirements.txt     # Python dependencies

🖼️ User Interface Overview

Sidebar: Sliders to adjust flower measurements

Main Area:

Predicted species

Probability chart

Feature importance chart

Dataset preview

🎯 How It Works

The Iris dataset is loaded and cached to speed up performance.

A Random Forest model is trained on the dataset.

User inputs are captured from sliders.

The model predicts the flower species.

Confidence levels and charts are displayed dynamically.

❤️ Author

Hussain Murtazaa

Built with passion using Python & Streamlit.
