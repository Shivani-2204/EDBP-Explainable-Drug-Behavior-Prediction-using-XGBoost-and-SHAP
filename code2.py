import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------
# 1. Load Dataset
# -------------------------------
df = pd.read_csv(r"better_youth_dataset.csv")

# -------------------------------
# 2. Data Cleaning
# -------------------------------
df = df.drop_duplicates()

# -------------------------------
# 3. Encoding
# -------------------------------
from sklearn.preprocessing import LabelEncoder

cat_cols = df.select_dtypes(include=['object']).columns
le = LabelEncoder()

for col in cat_cols:
    df[col] = le.fit_transform(df[col])

# -------------------------------
# 4. Train-Test Split
# -------------------------------
from sklearn.model_selection import train_test_split

X = df.drop(columns=['Drug_Experimentation'])
y = df['Drug_Experimentation']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# 5. XGBoost Model
# -------------------------------
from xgboost import XGBClassifier

xgb_model = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.1,
    random_state=42
)

xgb_model.fit(X_train, y_train)
y_pred = xgb_model.predict(X_test)

# -------------------------------
# 6. Evaluation
# -------------------------------
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix

print("\nEvaluation Metrics (XGBoost):")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))

print("\nClassification Report:\n", classification_report(y_test, y_pred))

# -------------------------------
# ✅ VISUAL 1: Feature Importance
# -------------------------------
from xgboost import plot_importance

plot_importance(xgb_model)
plt.title("Feature Importance")
plt.show()

# -------------------------------
# ✅ VISUAL 2: Confusion Matrix
# -------------------------------
cm = confusion_matrix(y_test, y_pred)

plt.figure()
sns.heatmap(cm, annot=True, fmt='d')
plt.title("Confusion Matrix Heatmap")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# -------------------------------
# 7. Model Comparison
# -------------------------------
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC(),
    "KNN": KNeighborsClassifier(),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
}

results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    specificity = tn / (tn + fp)

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred),
        "Recall": recall_score(y_test, pred),
        "F1-Score": f1_score(y_test, pred),
        "Specificity": specificity
    })

results_df = pd.DataFrame(results)
print("\nModel Comparison:\n", results_df)

# -------------------------------
# ✅ VISUAL 3: Combined Boxplot
# -------------------------------
melted = results_df.melt(
    id_vars="Model",
    value_vars=["Accuracy", "Precision", "Recall", "F1-Score", "Specificity"],
    var_name="Metric",
    value_name="Score"
)

plt.figure(figsize=(10,6))
sns.boxplot(x="Metric", y="Score", data=melted)
plt.title("Model Performance Comparison (All Metrics)")
plt.show()