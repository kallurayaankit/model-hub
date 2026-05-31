import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import pandas as pd

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("model-hub")

# Load hidden test data to train dummy models
df = pd.read_csv('data/hidden_test.csv')
X = df.drop('target', axis=1)
y = df['target']

# Model 1: Random Forest
with mlflow.start_run(run_name="random_forest_v1") as run:
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    mlflow.sklearn.log_model(model, "model")
    mlflow.set_tag("owner", "ankit")
    mlflow.set_tag("description", "Random Forest baseline")
    # Register in registry
    mlflow.register_model(f"runs:/{run.info.run_id}/model", "churn-model")

# Model 2: Logistic Regression
with mlflow.start_run(run_name="logistic_v1") as run:
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    mlflow.sklearn.log_model(model, "model")
    mlflow.set_tag("owner", "ankit")
    mlflow.set_tag("description", "Logistic Regression baseline")
    mlflow.register_model(f"runs:/{run.info.run_id}/model", "churn-model")

print("Models registered.")