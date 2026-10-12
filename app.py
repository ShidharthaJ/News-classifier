import json
import joblib
import pandas as pd
import streamlit as st
from preprocess import clean_text

st.set_page_config(page_title="News Headline Classifier", page_icon="📰", layout="centered")

@st.cache_resource
def load_artifacts():
    art = joblib.load("models/news_classifier.joblib")
    with open("models/metrics.json") as f:
        metrics = json.load(f)
    return art["vectorizer"], art["model"], metrics

vec, model, metrics = load_artifacts()

st.title("📰 News Classification Model")
st.caption("TF-IDF + Logistic Regression trained on Times of India headlines (8 news categories).")

EXAMPLES = {
    "Business": "Sensex jumps 400 points as RBI keeps repo rate unchanged",
    "Sports": "Kohli century guides India to a thrilling win over Australia in the ODI",
    "Technology": "Apple unveils new iPhone with faster chip and improved camera",
    "World": "US and China hold fresh trade talks amid rising tensions",
}
choice = st.selectbox("Try an example (optional)", ["-- type your own --"] + list(EXAMPLES))
default = "" if choice.startswith("--") else EXAMPLES[choice]
text = st.text_area("Enter a news headline or short article", value=default, height=140,
                    placeholder="e.g. Government announces new GST relief for small businesses")

if st.button("Classify", type="primary"):
    cleaned = clean_text(text)
    if not cleaned:
        st.warning("Please enter some meaningful text.")
    else:
        probs = model.predict_proba(vec.transform([cleaned]))[0]
        out = pd.DataFrame({"Category": model.classes_, "Probability": probs}).sort_values("Probability", ascending=False)
        top = out.iloc[0]
        st.success(f"Predicted category: **{top.Category}**  (confidence {top.Probability:.1%})")
        st.bar_chart(out.set_index("Category"))
        with st.expander("See the preprocessed text"):
            st.code(cleaned)

with st.sidebar:
    st.header("About the model")
    st.write(f"**Model:** {metrics['model']}")
    st.write(f"**Test accuracy:** {metrics['accuracy']:.1%}")
    st.write(f"**Macro F1:** {metrics['macro_f1']:.3f}")
    st.write("**Classes:**")
    for c in metrics["classes"]:
        st.write(f"- {c}")
    st.caption("Best on short headlines. 'City / Local' and 'India / National' overlap, so they are the hardest to separate.")
