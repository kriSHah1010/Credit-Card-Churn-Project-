import pandas as pd
from pathlib import Path
from joblib import dump

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from src.utils import make_dirs, load_processed_data, save_model
from src.features import add_features

RAW_PROCESSED_FILE = "credit_card_churn_cleaned.csv"


def build_pipeline(X: pd.DataFrame, model):
    categorical_cols = X.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    numerical_cols = X.select_dtypes(exclude=["object", "string", "category"]).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numerical_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )

    clf = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    return clf


def main():
    make_dirs()

    df = load_processed_data(RAW_PROCESSED_FILE)
    df = add_features(df)

    if "churned" not in df.columns:
        raise ValueError("Target column 'churned' not found. Check cleaning step.")

    X = df.drop(columns=["churned"])
    y = df["churned"]

    # Drop columns that should not be modeled
    for col in ["CLIENTNUM", "Attrition_Flag"]:
        if col in X.columns:
            X = X.drop(columns=[col])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "logreg": LogisticRegression(max_iter=1000),
        "rf": RandomForestClassifier(n_estimators=200, random_state=42)
    }

    best_model_name = None
    best_score = -1
    best_pipeline = None

    for name, model in models.items():
        pipeline = build_pipeline(X_train, model)
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]

        auc = roc_auc_score(y_test, y_prob)
        print(f"\n=== {name.upper()} ===")
        print("ROC AUC:", round(auc, 4))
        print(confusion_matrix(y_test, y_pred))
        print(classification_report(y_test, y_pred))

        if auc > best_score:
            best_score = auc
            best_model_name = name
            best_pipeline = pipeline

    save_model(best_pipeline, "best_churn_model.pkl")
    print(f"\nBest model: {best_model_name} | ROC AUC: {round(best_score, 4)}")
    print("Saved model to outputs/models/best_churn_model.pkl")


if __name__ == "__main__":
    main()