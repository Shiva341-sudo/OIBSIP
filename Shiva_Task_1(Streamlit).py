import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import numpy as np

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Iris Classifier",
    page_icon="🌸",
    layout="wide",
)

st.title("🌸 Iris Flower Classification")
st.markdown("Upload your `Iris.csv`, explore the data, train models, and predict species.")

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Settings")
uploaded_file = st.sidebar.file_uploader("Upload Iris CSV", type=["csv"])
model_choice  = st.sidebar.selectbox(
    "Select Model",
    ["Decision Tree", "Logistic Regression", "KNN", "Random Forest", "SVM"],
)
test_size = st.sidebar.slider("Test Size", 0.1, 0.4, 0.2, 0.05)

MODEL_MAP = {
    "Decision Tree":       DecisionTreeClassifier(random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=200, random_state=42),
    "KNN":                 KNeighborsClassifier(),
    "Random Forest":       RandomForestClassifier(random_state=42),
    "SVM":                 SVC(random_state=42),
}

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    if "Id" in df.columns:
        df = df.drop("Id", axis=1)
    return df

if uploaded_file:
    df = load_data(uploaded_file)
else:
    st.info("No file uploaded — using the built-in sklearn Iris dataset.")
    from sklearn.datasets import load_iris
    iris = load_iris(as_frame=True)
    df   = iris.frame
    df.columns = [*iris.feature_names, "Species"]
    df["Species"] = df["Species"].map(dict(enumerate(iris.target_names)))

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["📊 EDA", "🤖 Model", "📈 Results", "🔮 Predict"])

# ─── TAB 1 : EDA ─────────────────────────────────────────────────────────────
with tab1:
    st.subheader("Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.write("**Shape:**", df.shape)
        st.write("**Missing values:**", df.isnull().sum().sum())
    with c2:
        st.write(df.describe())

    st.subheader("Class Distribution")
    fig, ax = plt.subplots(figsize=(5, 3))
    sns.countplot(x="Species", data=df, ax=ax, palette="Set2")
    ax.set_title("Class Distribution")
    st.pyplot(fig)
    plt.close()

    st.subheader("Pair Plot")
    fig2 = sns.pairplot(df, hue="Species", palette="Set1")
    st.pyplot(fig2.figure)
    plt.close()

    st.subheader("Correlation Heatmap")
    fig3, ax3 = plt.subplots(figsize=(6, 4))
    sns.heatmap(df.drop("Species", axis=1).corr(), annot=True, cmap="coolwarm", ax=ax3)
    ax3.set_title("Feature Correlation Heatmap")
    st.pyplot(fig3)
    plt.close()

# ─── Prepare X, y (shared across tabs) ──────────────────────────────────────
X  = df.drop("Species", axis=1)
le = LabelEncoder()
y  = le.fit_transform(df["Species"])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=42, stratify=y
)

model = MODEL_MAP[model_choice]
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# ─── TAB 2 : Model info ──────────────────────────────────────────────────────
with tab2:
    st.subheader(f"Selected Model: {model_choice}")
    st.write(f"**Training samples:** {len(X_train)}  |  **Test samples:** {len(X_test)}")

    cv_scores = cross_val_score(MODEL_MAP[model_choice].__class__(random_state=42)
                                if model_choice != "KNN" else KNeighborsClassifier(),
                                X, y, cv=5)
    st.metric("5-Fold CV Accuracy", f"{cv_scores.mean():.4f}", f"±{cv_scores.std():.4f}")

    st.write("**Model Parameters:**")
    st.json({k: str(v) for k, v in model.get_params().items()})

# ─── TAB 3 : Results ─────────────────────────────────────────────────────────
with tab3:
    st.subheader("Model Performance")
    acc = accuracy_score(y_test, y_pred)
    st.metric("Test Accuracy", f"{acc:.4f}")

    st.text("Classification Report")
    report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose().round(3), use_container_width=True)

    st.subheader("Confusion Matrix")
    fig4, ax4 = plt.subplots(figsize=(5, 4))
    cm   = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
    disp.plot(cmap="Blues", ax=ax4)
    ax4.set_title("Confusion Matrix")
    st.pyplot(fig4)
    plt.close()

    if hasattr(model, "feature_importances_"):
        st.subheader("Feature Importance")
        fi = pd.DataFrame({"Feature": X.columns, "Importance": model.feature_importances_})
        fi = fi.sort_values("Importance", ascending=False)
        fig5, ax5 = plt.subplots(figsize=(6, 3))
        sns.barplot(x="Importance", y="Feature", data=fi, ax=ax5, palette="viridis")
        ax5.set_title("Feature Importance")
        st.pyplot(fig5)
        plt.close()

# ─── TAB 4 : Predict ─────────────────────────────────────────────────────────
with tab4:
    st.subheader("Predict a New Sample")
    cols = X.columns.tolist()

    c1, c2 = st.columns(2)
    inputs = {}
    for i, col in enumerate(cols):
        with (c1 if i % 2 == 0 else c2):
            inputs[col] = st.number_input(
                col,
                float(X[col].min()),
                float(X[col].max()),
                float(X[col].mean()),
                step=0.1,
            )

    if st.button("🔮 Predict Species"):
        sample = pd.DataFrame([inputs])
        pred   = model.predict(sample)[0]
        species = le.inverse_transform([pred])[0]
        emoji  = {"setosa": "🌺", "versicolor": "🌼", "virginica": "🌷"}.get(species, "🌸")
        st.success(f"**Predicted Species:** {emoji} **{species.capitalize()}**")
