
import pandas as pd
import re


def normalize_name(name):
    name = str(name).strip().lower()
    name = re.sub(r"[_\-]+", " ", name)
    name = re.sub(r"\s+", " ", name)
    return name


def find_column(df, possible_names):

    aliases = [normalize_name(x) for x in possible_names]

    for column in df.columns:
        if normalize_name(column) in aliases:
            return column

    return None


def generate_recommendations(df):

    recommendations = []

    if df.empty:
        return ["The uploaded dataset contains no records to analyse."]

    sales_column = find_column(
        df,
        [
            "sales", "revenue", "amount",
            "total sales", "total revenue",
            "purchase amount", "transaction amount",
            "order value", "total amount", "net sales",
            "price", "total price", "selling price",
            "payment amount", "purchase value"
        ]
    )

    category_column = find_column(
        df,
        [
            "category", "product category",
            "product", "product name",
            "item", "item name", "department",
            "type", "product type"
        ]
    )

    region_column = find_column(
        df,
        [
            "region", "city", "state",
            "country", "location", "area",
            "territory", "branch"
        ]
    )

    customer_column = find_column(
        df,
        [
            "customer id", "customer",
            "customer name", "client id",
            "client", "client name",
            "user id", "user"
        ]
    )

    # Sales / revenue insights
    if sales_column:

        sales = pd.to_numeric(
            df[sales_column].astype(str)
            .str.replace(r"[₹$€£,%]", "", regex=True)
            .str.replace(",", "", regex=False),
            errors="coerce"
        ).dropna()

        if not sales.empty:

            recommendations.append(
                f"Total {sales_column}: {sales.sum():,.2f}. "
                "Monitor this metric to understand overall performance."
            )

            recommendations.append(
                f"Average {sales_column}: {sales.mean():,.2f}. "
                "Look for ways to improve the average transaction value."
            )

            if len(sales) > 1:

                if sales.iloc[-1] > sales.iloc[0]:
                    recommendations.append(
                        "The last recorded value is higher than the first. "
                        "Check whether this improvement continues."
                    )

                elif sales.iloc[-1] < sales.iloc[0]:
                    recommendations.append(
                        "The last recorded value is lower than the first. "
                        "Investigate possible reasons for the decline."
                    )

    else:
        recommendations.append(
            "No sales or revenue column was detected. "
            "General dataset analysis is available, but "
            "sales-based insights cannot be generated."
        )

    # Category insights
    if category_column and sales_column:

        temp = df[[category_column, sales_column]].copy()

        temp[sales_column] = pd.to_numeric(
            temp[sales_column].astype(str)
            .str.replace(r"[₹$€£,%]", "", regex=True)
            .str.replace(",", "", regex=False),
            errors="coerce"
        )

        grouped = (
            temp.dropna()
            .groupby(category_column)[sales_column]
            .sum()
            .sort_values(ascending=False)
        )

        if not grouped.empty:

            recommendations.append(
                f"{grouped.idxmax()} has the highest "
                f"{sales_column} among the categories."
            )

            if len(grouped) > 1:
                recommendations.append(
                    f"{grouped.idxmin()} has the lowest "
                    f"{sales_column}. Review its performance."
                )

    # Location insights
    if region_column and sales_column:

        temp = df[[region_column, sales_column]].copy()

        temp[sales_column] = pd.to_numeric(
            temp[sales_column].astype(str)
            .str.replace(r"[₹$€£,%]", "", regex=True)
            .str.replace(",", "", regex=False),
            errors="coerce"
        )

        grouped = (
            temp.dropna()
            .groupby(region_column)[sales_column]
            .sum()
            .sort_values(ascending=False)
        )

        if not grouped.empty:
            recommendations.append(
                f"{grouped.idxmax()} has the highest "
                "recorded value among the locations."
            )

    # Customer insights
    if customer_column:

        count = df[customer_column].nunique()

        recommendations.append(
            f"The dataset contains {count:,} unique customers. "
            "Study their behaviour to identify customer groups."
        )

    # General dataset insights
    recommendations.append(
        f"The dataset contains {len(df):,} records "
        f"and {len(df.columns)} columns."
    )

    return recommendations


