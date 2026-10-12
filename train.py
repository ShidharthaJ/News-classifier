import time, json, joblib, pandas as pd, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from preprocess import clean_text, CATEGORY_MAP

SEED, PER_CLASS = 42, 30000
raw = pd.read_csv("/home/claude/data/india-news-headlines.csv")
raw["prefix"] = raw.headline_category.str.split(".").str[0]
df = raw[raw.prefix.isin(CATEGORY_MAP)].copy()
df["label"] = df.prefix.map(CATEGORY_MAP)
before = df.label.value_counts()
df = pd.concat([g.sample(min(len(g), PER_CLASS), random_state=SEED) for _, g in df.groupby("label")])
df["clean"] = df.headline_text.map(clean_text)
df = df[df.clean.str.len() > 0]
df[["publish_date","headline_text","label"]].sample(2000, random_state=SEED).to_csv("data/sample_headlines.csv", index=False)

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
before.sort_values().plot.barh(ax=ax[0], color="#4C78A8"); ax[0].set_title("Class counts (raw, after mapping)")
df.label.value_counts().sort_values().plot.barh(ax=ax[1], color="#59A14F"); ax[1].set_title("Class counts (balanced sample)")
plt.tight_layout(); plt.savefig("figures/class_distribution.png", dpi=130); plt.close()

X_tr, X_te, y_tr, y_te = train_test_split(df.clean, df.label, test_size=0.2, stratify=df.label, random_state=SEED)
vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=100000, sublinear_tf=True)
Xtr, Xte = vec.fit_transform(X_tr), vec.transform(X_te)
print("train", Xtr.shape, "test", Xte.shape)

models = {
    "Multinomial Naive Bayes": MultinomialNB(alpha=0.3),
    "Linear SVM": LinearSVC(C=0.5),
    "Logistic Regression": LogisticRegression(C=5, max_iter=300),
}
rows, fitted = [], {}
for name, m in models.items():
    t = time.time(); m.fit(Xtr, y_tr); p = m.predict(Xte); fitted[name] = (m, p)
    rows.append({"Model": name, "Accuracy": accuracy_score(y_te, p),
                 "Macro F1": f1_score(y_te, p, average="macro"), "Train time (s)": round(time.time()-t, 1)})
    print(rows[-1])
res = pd.DataFrame(rows).sort_values("Macro F1", ascending=False)
res.to_csv("models/model_comparison.csv", index=False)

# Deploy Logistic Regression (gives probabilities for the app) unless clearly worse
best_name = "Logistic Regression"
m, p = fitted[best_name]
rep = classification_report(y_te, p, output_dict=True)
open("models/classification_report.txt", "w").write(classification_report(y_te, p))
print(classification_report(y_te, p))
labels = sorted(df.label.unique())
cm = confusion_matrix(y_te, p, labels=labels)
plt.figure(figsize=(8, 6.5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title(f"Confusion matrix - {best_name}")
plt.xticks(rotation=40, ha="right"); plt.tight_layout(); plt.savefig("figures/confusion_matrix.png", dpi=130); plt.close()

res.set_index("Model")[["Accuracy","Macro F1"]].plot.bar(figsize=(7,4), rot=15); plt.ylim(0.5,1)
plt.title("Model comparison"); plt.tight_layout(); plt.savefig("figures/model_comparison.png", dpi=130); plt.close()

# top features per class
fn = np.array(vec.get_feature_names_out())
top = {c: list(fn[np.argsort(m.coef_[i])[-10:][::-1]]) for i, c in enumerate(m.classes_)}
json.dump(top, open("models/top_features.json", "w"), indent=1)
json.dump({"model": best_name, "accuracy": accuracy_score(y_te, p), "macro_f1": f1_score(y_te, p, average="macro"),
           "n_train": int(len(y_tr)), "n_test": int(len(y_te)), "classes": list(m.classes_)}, open("models/metrics.json","w"), indent=1)
joblib.dump({"vectorizer": vec, "model": m}, "models/news_classifier.joblib", compress=3)
print(top)
