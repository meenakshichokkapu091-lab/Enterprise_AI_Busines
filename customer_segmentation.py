
import pandas as pd
import numpy as np

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def find_column(data, keywords):

    for column in data.columns:

        normalized = (
            str(column).lower()
            .replace("_", " ")
            .strip()
        )

        if normalized in keywords:
            return column

    return None


def segment_customers(data):

    data = data.copy()

    # Identify customer column
    customer_column = find_column(
        data,
        ["customer id", "customer", "client id", "client"]
    )

    if customer_column is None:
        raise ValueError(
            "Customer segmentation requires a customer ID column."
        )

    # Identify useful numerical columns
    numeric_columns = data.select_dtypes(
        include=np.number
    ).columns.tolist()

    if not numeric_columns:
        raise ValueError(
            "No numerical customer information is available."
        )

    # Remove customer identifiers from clustering features
    feature_columns = [
        column for column in numeric_columns
        if column != customer_column
    ]

    if not feature_columns:
        raise ValueError(
            "No suitable numerical features for segmentation."
        )

    # Aggregate records for each customer
    customer_data = data.groupby(
        customer_column
    )[feature_columns].mean().reset_index()

    customer_data = customer_data.replace(
        [np.inf, -np.inf], np.nan
    )

    customer_data[feature_columns] = (
        customer_data[feature_columns].fillna(
            customer_data[feature_columns].median()
        )
    )

    # Remove features with no variation
    useful_features = [
        column for column in feature_columns
        if customer_data[column].nunique() > 1
    ]

    if not useful_features:
        raise ValueError(
            "Customer data does not contain enough variation."
        )

    # At least two customers are required
    number_of_customers = len(customer_data)

    if number_of_customers < 2:
        raise ValueError(
            "At least two customers are required for segmentation."
        )

    # Select cluster count based on available customers
    cluster_count = min(3, number_of_customers)

    scaler = StandardScaler()

    scaled_data = scaler.fit_transform(
        customer_data[useful_features]
    )

    model = KMeans(
        n_clusters=cluster_count,
        random_state=42,
        n_init=10
    )

    customer_data["customer_segment"] = (
        model.fit_predict(scaled_data)
    )

    # Identify spending feature if available
    spending_column = next(
        (
            column for column in useful_features
            if column.lower() in [
                "sales", "revenue", "amount",
                "total spend", "total spending",
                "purchase amount"
            ]
        ),
        None
    )

    if spending_column:

        average_spend = customer_data.groupby(
            "customer_segment"
        )[spending_column].mean()

        segment_order = average_spend.sort_values().index.tolist()

        labels = [
            "Low Value Customers",
            "Medium Value Customers",
            "High Value Customers"
        ]

        segment_names = {
            segment: labels[index]
            for index, segment in enumerate(segment_order)
        }

    else:

        segment_names = {
            segment: f"Customer Group {index + 1}"
            for index, segment in enumerate(
                sorted(customer_data["customer_segment"].unique())
            )
        }

    customer_data["segment_name"] = (
        customer_data["customer_segment"].map(segment_names)
    )

    return customer_data


if __name__ == "__main__":

    print(
        "Customer segmentation module is ready."
    )

