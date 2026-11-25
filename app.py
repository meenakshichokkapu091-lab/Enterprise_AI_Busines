
    if getattr(churn_scaler, "n_features_in_", 3) != len(features):
        raise ValueError(
            "The churn model's expected features do not match."
        )

    scaled_input = churn_scaler.transform(churn_input)

    probabilities = churn_model.predict_proba(
        scaled_input
    )

    classes = list(churn_model.classes_)

    if 1 not in classes:
        raise ValueError(
            "Expected positive class not found in churn model."
        )

    positive_index = classes.index(1)

    customer_data["churn_probability"] = (
        probabilities[:, positive_index] * 100
    ).round(2)

    customer_data["churn_status"] = np.where(
        customer_data["churn_probability"] >= 50,
        "At Risk",
        "Active"
    )

    return customer_data


# --------------------------------------------------
# MAIN DATA ANALYSIS
# --------------------------------------------------

def analyse_dataset(dataframe, cleaning_report, cleaned_filename):

    data = dataframe.copy()

    if data.empty:
        raise ValueError(
            "No records available for analysis."
        )

    sales_column = find_column(data, "sales")
    profit_column = find_column(data, "profit")
    customer_column = find_column(data, "customer")
    date_column = find_column(data, "date")
    category_column = find_column(data, "category")
    region_column = find_column(data, "region")

    numeric_columns = data.select_dtypes(
        include=np.number
    ).columns.tolist()

    metric_column = sales_column

    if metric_column is None and numeric_columns:
        metric_column = numeric_columns[0]

    if metric_column:

        metric_values = get_numeric_series(
            data, metric_column
        ).dropna()

        metric_total = format_currency(
            metric_values.sum()
        )

        metric_average = format_currency(
            metric_values.mean()
        )

    else:
        metric_total = "Not Available"
        metric_average = "Not Available"

    if profit_column:

        profit_values = get_numeric_series(
            data, profit_column
        ).dropna()

        total_profit = (
            format_currency(profit_values.sum())
            if not profit_values.empty
            else "Not Available"
        )

    else:
        total_profit = "Not Available"

    total_customers = (
        int(data[customer_column].nunique())
        if customer_column else 0
    )

    # General chart
    chart_labels = []
    chart_values = []

    categorical_columns = data.select_dtypes(
        include=["object", "category", "string"]
    ).columns.tolist()

    excluded_names = [
        "id", "customer id", "order id",
        "transaction id", "invoice id"
    ]

    chart_category = next(
        (
            column for column in categorical_columns
            if normalize_name(column) not in excluded_names
            and data[column].nunique() <= 100
        ),
        None
    )

    if metric_column and chart_category:

        temp = data[
            [chart_category, metric_column]
        ].copy()

        temp[metric_column] = get_numeric_series(
            temp, metric_column
        )

        grouped = (
            temp.dropna()
            .groupby(chart_category)[metric_column]
            .sum()
            .sort_values(ascending=False)
            .head(10)
        )

        chart_labels = [
            str(value) for value in grouped.index
        ]

        chart_values = [
            round(float(value), 2)
            for value in grouped.values
        ]

    # Category chart
    category_labels = []
    category_values = []

    if metric_column and category_column:

        temp = data[
            [category_column, metric_column]
        ].copy()
