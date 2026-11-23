
import os
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


def train_churn_model(csv_path):

    # Read dataset
    data = pd.read_csv(csv_path)

    # Check required columns
    required = ["Customer ID", "Order Date", "Sales"]

    for col in required:
        if col not in data.columns:
            raise ValueError(f"Missing column: {col}")

    # Convert columns
    data["Order Date"] = pd.to_datetime(
        data["Order Date"],
        errors="coerce"
    )

    data["Sales"] = pd.to_numeric(
        data["Sales"],
        errors="coerce"
    )

    data = data.dropna(
        subset=["Customer ID", "Order Date", "Sales"]
    )

    # Customer-level analysis
    latest_date = data["Order Date"].max()

    customer_data = data.groupby("Customer ID").agg(
        total_spend=("Sales", "sum"),
        purchase_frequency=("Order Date", "nunique"),
        last_purchase=("Order Date", "max")
    ).reset_index()

    customer_data["days_inactive"] = (
        latest_date - customer_data["last_purchase"]
    ).dt.days

    # Create proxy churn label:
    # 1 = inactive for more than 90 days
    # 0 = purchased within the last 90 days
    customer_data["churn"] = (
        customer_data["days_inactive"] > 90
    ).astype(int)

    # Features and target
    features = [
        "total_spend",
        "purchase_frequency",
        "days_inactive"
    ]

    X = customer_data[features]
    y = customer_data["churn"]

    # Check if both classes exist
    if y.nunique() < 2:
        raise ValueError(
            "Dataset does not contain both churn classes. "
            "Cannot train a classifier."
        )

    # Train Random Forest
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced"
    )

    model.fit(X_scaled, y)

    # Save model and scaler
    os.makedirs("models", exist_ok=True)

    joblib.dump(
        model,
        "models/churn_model.pkl"
    )

    joblib.dump(
        scaler,
        "models/churn_scaler.pkl"
    )

    # Predict customer churn risk
    customer_data["churn_probability"] = (
        model.predict_proba(X_scaled)[:, 1] * 100
    ).round(2)

    customer_data["churn_status"] = (
        customer_data["churn"].map({
            1: "At Risk",
            0: "Active"
        })
    )

    return customer_data


if __name__ == "__main__":

    result = train_churn_model("train.csv")

    print("\nCustomer Churn Prediction Results:")

    print(result.head(10))

    print("\nChurn Summary:")

    print(result["churn_status"].value_counts())
