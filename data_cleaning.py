
import pandas as pd
import numpy as np


def clean_dataset(data):

    data = data.copy()

    original_rows = len(data)
    original_columns = len(data.columns)

    # 1. Clean column names
    data.columns = (
        data.columns.astype(str)
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )

    # 2. Remove empty rows and columns
    data = data.dropna(how="all")
    data = data.dropna(axis=1, how="all")

    # 3. Remove duplicate rows
    duplicates_removed = int(data.duplicated().sum())
    data = data.drop_duplicates().copy()

    # 4. Convert numeric columns
    numeric_filled = 0

    for column in data.columns:

        if pd.api.types.is_numeric_dtype(data[column]):
            continue

        converted = pd.to_numeric(
            data[column].astype(str)
            .str.replace(r"[$₹€£,%]", "", regex=True)
            .str.replace(",", "", regex=False),
            errors="coerce"
        )

        if (
            converted.notna().sum() > 0
            and converted.notna().mean() >= 0.8
        ):
            data[column] = converted

    # 5. Recognize sales column
    sales_names = [
        "sales",
        "amount",
        "revenue",
        "total sales",
        "sales amount",
        "order amount"
    ]

    column_lookup = {
        col.lower().strip(): col
        for col in data.columns
    }

    for name in sales_names:

        if name in column_lookup:

            actual_column = column_lookup[name]

            if actual_column != "Sales":
                data.rename(
                    columns={actual_column: "Sales"},
                    inplace=True
                )

            break

    # 6. Fill missing numeric values
    for column in data.select_dtypes(
        include=np.number
    ).columns:

        missing_count = int(data[column].isna().sum())

        if missing_count > 0:

            median_value = data[column].median()

            if pd.notna(median_value):
                data[column] = data[column].fillna(
                    median_value
                )
                numeric_filled += missing_count

    # 7. Detect date columns
    date_filled = 0

    date_keywords = [
        "date", "time", "day",
        "month", "year", "timestamp"
    ]

    for column in data.columns:

        if not any(
            word in column.lower()
            for word in date_keywords
        ):
            continue

        if pd.api.types.is_datetime64_any_dtype(
            data[column]
        ):
            continue

        converted_dates = pd.to_datetime(
            data[column],
            errors="coerce",
            format="mixed"
        )

        if converted_dates.notna().mean() >= 0.5:
            data[column] = converted_dates

    # 8. Fill missing text values
    text_filled = 0

    for column in data.select_dtypes(
        include=["object", "string"]
    ).columns:

        missing_count = int(data[column].isna().sum())

        if missing_count > 0:
            data[column] = data[column].fillna(
                "Unknown"
            )
            text_filled += missing_count

    # 9. Remove completely empty columns
    data = data.dropna(axis=1, how="all")

    # 10. Cleaning report
    cleaning_report = {

        "original_rows": original_rows,
        "cleaned_rows": len(data),

        "duplicates_removed": duplicates_removed,

        "missing_numeric_filled": numeric_filled,
        "missing_text_filled": text_filled,

        "original_columns": original_columns,
        "cleaned_columns": len(data.columns),

        "sales_column_detected": (
            "Sales" in data.columns
        )
    }

    return data, cleaning_report


