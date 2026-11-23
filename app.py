
import os
import re
import uuid
import joblib
import pandas as pd
import numpy as np
import xgboost as xgb

from datetime import timedelta
from flask import (
    Flask, render_template, request, redirect,
    url_for, session, send_from_directory, flash
)
from werkzeug.utils import secure_filename

from data_cleaning import clean_dataset
from customer_segmentation import segment_customers
from business_recommendations import generate_recommendations


# --------------------------------------------------
# FLASK CONFIGURATION
# --------------------------------------------------

app = Flask(__name__)

app.secret_key = os.environ.get(
    "SECRET_KEY",
    "enterprise-ai-local-secret-key"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
MODEL_FOLDER = os.path.join(BASE_DIR, "models")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024

ALLOWED_EXTENSIONS = {"csv", "xlsx"}

analysis_store = {}


# --------------------------------------------------
# LOAD ML MODELS
# --------------------------------------------------

sales_model = None
churn_model = None
churn_scaler = None

SALES_MODEL_PATH = os.path.join(
    MODEL_FOLDER, "sales_forecasting.json"
)

CHURN_MODEL_PATH = os.path.join(
    MODEL_FOLDER, "churn_model.pkl"
)

CHURN_SCALER_PATH = os.path.join(
    MODEL_FOLDER, "churn_scaler.pkl"
)


def load_ml_models():

    global sales_model, churn_model, churn_scaler

    # Sales forecasting model
    try:
        if os.path.isfile(SALES_MODEL_PATH):

            model = xgb.XGBRegressor()
            model.load_model(SALES_MODEL_PATH)

            sales_model = model

            print("SUCCESS: Sales forecasting model loaded.")

        else:
            sales_model = None
            print("WARNING: Sales forecasting model file not found.")
            print("Expected location:", SALES_MODEL_PATH)

    except Exception as error:
        sales_model = None
        print("Sales model loading error:", error)

    # Churn model
    try:
        if os.path.isfile(CHURN_MODEL_PATH):
            churn_model = joblib.load(CHURN_MODEL_PATH)
            print("SUCCESS: Churn model loaded.")
        else:
            churn_model = None
            print("WARNING: Churn model file not found.")

    except Exception as error:
        churn_model = None
        print("Churn model loading error:", error)

    # Churn scaler
    try:
        if os.path.isfile(CHURN_SCALER_PATH):
            churn_scaler = joblib.load(CHURN_SCALER_PATH)
            print("SUCCESS: Churn scaler loaded.")
        else:
            churn_scaler = None
            print("WARNING: Churn scaler file not found.")

    except Exception as error:
        churn_scaler = None
        print("Churn scaler loading error:", error)


load_ml_models()


# --------------------------------------------------
# COLUMN DETECTION
# --------------------------------------------------

COLUMN_ALIASES = {

    "sales": [
        "sales", "revenue", "amount",
        "total sales", "total revenue",
        "purchase amount", "transaction amount",
        "order value", "total amount", "net sales",
        "total price", "selling price",
        "payment amount", "purchase value",
        "sales amount", "sales revenue",
        "sale amount", "price", "grand total",
        "invoice amount", "subtotal",
        "total purchase", "order amount",
        "final amount", "net amount"
    ],

    "profit": [
        "profit", "net profit", "gross profit",
        "earnings", "income", "margin"
    ],

    "customer": [
        "customer id", "customer", "customer name",
        "client id", "client", "client name",
        "user id", "user", "buyer id",
        "buyer name", "consumer id"
    ],

    "date": [
        "order date", "date", "transaction date",
        "purchase date", "sale date", "invoice date",
        "created date", "timestamp", "order timestamp",
        "payment date", "billing date", "purchase timestamp"
    ],

    "order": [
        "order id", "transaction id", "invoice id",
        "purchase id", "bill id", "receipt id",
        "order number", "transaction number"
    ],

    "category": [
        "category", "product category",
        "product", "product name", "item", "item name",
        "department", "product type", "type"
    ],

    "region": [
        "region", "city", "state",
        "country", "location", "area",
        "branch", "territory"
    ]
}


def normalize_name(name):

    name = str(name).strip().lower()
    name = re.sub(r"[_\-]+", " ", name)
    name = re.sub(r"\s+", " ", name)

    return name


def find_column(data, column_type):

    aliases = COLUMN_ALIASES.get(column_type, [])

    normalized_aliases = {
        normalize_name(alias) for alias in aliases
    }

    for column in data.columns:
        if normalize_name(column) in normalized_aliases:
            return column

    return None


# --------------------------------------------------
# GENERAL HELPERS
# --------------------------------------------------

def allowed_file(filename):

    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower()
        in ALLOWED_EXTENSIONS
    )


def make_json_safe(value):

    if value is None:
        return ""

    if isinstance(value, (pd.Timestamp,)):
        return value.strftime("%Y-%m-%d")

    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.floating,)):
        value = float(value)

    if isinstance(value, (np.bool_,)):
        return bool(value)

    if isinstance(value, float) and not np.isfinite(value):
        return ""

    if pd.isna(value):
        return ""

    return value


def dataframe_records(dataframe):

    if dataframe is None or dataframe.empty:
        return []

    records = dataframe.to_dict(orient="records")

    return [
        {
            key: make_json_safe(value)
            for key, value in record.items()
        }
        for record in records
    ]


def format_currency(value):

    try:
        return "₹{:,.2f}".format(float(value))
    except (ValueError, TypeError):
        return "Not Available"


def get_numeric_series(data, column):

    values = data[column]

    if (
        values.dtype == "object"
        or pd.api.types.is_string_dtype(values)
    ):

        values = (
            values.astype(str)
            .str.replace(r"[₹$€£,%]", "", regex=True)
            .str.replace(",", "", regex=False)
            .str.strip()
        )

    return pd.to_numeric(
        values, errors="coerce"
    ).replace([np.inf, -np.inf], np.nan)


# --------------------------------------------------
# FLEXIBLE CSV / EXCEL READER
# --------------------------------------------------

def read_uploaded_dataset(file_path, extension):

    if extension == "xlsx":
        return pd.read_excel(file_path)

    encodings = [
        "utf-8-sig",
        "utf-8",
        "cp1252",
        "latin1"
    ]

    separators = [
        ",",
        ";",
        "\t",
        None
    ]

    last_error = None

    for encoding in encodings:

        for separator in separators:

            try:

                data = pd.read_csv(
                    file_path,
                    encoding=encoding,
                    sep=separator,
                    engine="python"
                )

                if data.empty and len(data.columns) == 0:
                    continue

                if len(data.columns) == 1:
                    header = str(data.columns[0])

                    if any(
                        delimiter in header
                        for delimiter in [",", ";", "\t"]
                    ):
                        continue

                print(
                    "CSV loaded:",
                    encoding,
                    separator,
                    data.shape
                )

                return data

            except (
                UnicodeDecodeError,
                pd.errors.ParserError,
                pd.errors.EmptyDataError
            ) as error:

                last_error = error

    raise ValueError(
        f"Unable to read CSV file: {last_error}"
    )


# --------------------------------------------------
# SALES FORECASTING
# --------------------------------------------------

def generate_sales_forecast(data, date_column, value_column):

    if sales_model is None:
        raise ValueError(
            "Sales forecasting model is not loaded. "
            "Check models/sales_forecasting.json."
        )

    if date_column is None or value_column is None:
        raise ValueError(
            "A suitable date and sales column are required."
        )

    working = data[[date_column, value_column]].copy()

    working[date_column] = pd.to_datetime(
        working[date_column], errors="coerce"
    )

    working[value_column] = get_numeric_series(
        working, value_column
    )

    working = working.dropna()

    if working.empty:
        raise ValueError(
            "No valid sales and date records available."
        )

    latest_date = working[date_column].max()

    forecast_dates = [
        latest_date + timedelta(days=day)
        for day in range(1, 31)
    ]

    forecast_input = pd.DataFrame({
        "Day": [date.day for date in forecast_dates],
        "Month": [date.month for date in forecast_dates],
        "Year": [date.year for date in forecast_dates],
        "DayOfWeek": [date.dayofweek for date in forecast_dates]
    })

    predictions = sales_model.predict(forecast_input)
    predictions = np.maximum(predictions, 0)

    labels = [
        date.strftime("%d %b %Y")
        for date in forecast_dates
    ]

    values = [
        round(float(value), 2)
        for value in predictions
    ]

    return labels, values


# --------------------------------------------------
# CUSTOMER SEGMENTATION
# --------------------------------------------------

def generate_customer_segments(data):

    segmented_data = segment_customers(data)

    segment_distribution = (
        segmented_data["segment_name"].value_counts()
    )

    return segmented_data, segment_distribution


# --------------------------------------------------
# CHURN PREDICTION
# --------------------------------------------------

def generate_churn_predictions(data):

    if churn_model is None or churn_scaler is None:
        raise ValueError(
            "The trained churn model or scaler is not available."
        )

    customer_column = find_column(data, "customer")
    sales_column = find_column(data, "sales")
    date_column = find_column(data, "date")
    order_column = find_column(data, "order")

    if not all([
        customer_column,
        sales_column,
        date_column,
        order_column
    ]):
        raise ValueError(
            "Churn prediction requires customer, sales, "
            "date and order ID columns."
        )

    working = data.copy()

    working[sales_column] = get_numeric_series(
        working, sales_column
    ).fillna(0)

    working[date_column] = pd.to_datetime(
        working[date_column], errors="coerce"
    )

    latest_date = working[date_column].max()

    if pd.isna(latest_date):
        raise ValueError(
            "Valid customer purchase dates are required."
        )

    customer_data = working.groupby(customer_column).agg(
        total_spend=(sales_column, "sum"),
        purchase_frequency=(order_column, "nunique"),
        last_purchase=(date_column, "max")
    ).reset_index()

    customer_data["average_order_value"] = (
        customer_data["total_spend"]
        / customer_data["purchase_frequency"].replace(0, 1)
    )

    features = [
        "total_spend",
        "purchase_frequency",
        "average_order_value"
    ]

    churn_input = customer_data[features].replace(
        [np.inf, -np.inf], np.nan
    ).fillna(0)

    if getattr(churn_scaler, "n_features_in_", 3) != len(features):
        raise ValueError(
            "The churn model's expected features do not match."
        )

    scaled_input = churn_scaler.transform(churn_input)

    probabilities = churn_model.predict_proba(scaled_input)
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
        raise ValueError("No records available for analysis.")

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

        metric_total = format_currency(metric_values.sum())
        metric_average = format_currency(metric_values.mean())

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

        temp = data[[chart_category, metric_column]].copy()

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

        chart_labels = [str(x) for x in grouped.index]
        chart_values = [round(float(x), 2) for x in grouped.values]

    # Category chart
    category_labels = []
    category_values = []

    if metric_column and category_column:

        temp = data[[category_column, metric_column]].copy()

        temp[metric_column] = get_numeric_series(
            temp, metric_column
        )

        grouped = (
            temp.dropna()
            .groupby(category_column)[metric_column]
            .sum()
            .sort_values(ascending=False)
        )

        category_labels = [str(x) for x in grouped.index]
        category_values = [round(float(x), 2) for x in grouped.values]

    # Region chart
    region_labels = []
    region_values = []

    if metric_column and region_column:

        temp = data[[region_column, metric_column]].copy()

        temp[metric_column] = get_numeric_series(
            temp, metric_column
        )

        grouped = (
            temp.dropna()
            .groupby(region_column)[metric_column]
            .sum()
            .sort_values(ascending=False)
        )

        region_labels = [str(x) for x in grouped.index]
        region_values = [round(float(x), 2) for x in grouped.values]

    # Segmentation
    segmentation_records = []
    customer_segments = []
    segment_counts = []
    segmentation_error = None

    try:

        if customer_column is None:
            raise ValueError(
                "No customer ID or customer name column detected."
            )

        customer_data, segment_distribution = (
            generate_customer_segments(data)
        )

        customer_segments = [
            str(value) for value in segment_distribution.index
        ]

        segment_counts = [
            int(value) for value in segment_distribution.values
        ]

        segmentation_records = dataframe_records(customer_data)

    except Exception as error:

        segmentation_error = str(error)
        print("Segmentation error:", error)

    # Forecasting
    forecast_labels = []
    forecast_values = []
    forecast_error = None

    try:

        if sales_column is None:
            raise ValueError(
                "No sales or revenue column detected. "
                "General analytics are still available."
            )

        if date_column is None:
            raise ValueError(
                "No suitable date column detected."
            )

        forecast_labels, forecast_values = generate_sales_forecast(
            data, date_column, sales_column
        )

    except Exception as error:

        forecast_error = str(error)
        print("Forecast error:", error)

    # Churn
    churn_customers = []
    at_risk_customers = 0
    active_customers = 0
    churn_percentage = 0
    churn_error = None

    try:

        churn_results = generate_churn_predictions(data)

        at_risk_customers = int(
            (churn_results["churn_status"] == "At Risk").sum()
        )

        active_customers = int(
            (churn_results["churn_status"] == "Active").sum()
        )

        if len(churn_results) > 0:
            churn_percentage = round(
                at_risk_customers / len(churn_results) * 100, 2
            )

        churn_results = churn_results.sort_values(
            "churn_probability", ascending=False
        )

        churn_customers = dataframe_records(churn_results)

    except Exception as error:

        churn_error = str(error)
        print("Churn prediction error:", error)

    # Recommendations
    recommendation_error = None

    try:
        business_recommendations = generate_recommendations(data)

    except Exception as error:
        business_recommendations = []
        recommendation_error = str(error)
        print("Recommendation error:", error)

    columns = data.columns.tolist()
    records = dataframe_records(data.head(10))

    return {

        "total_sales": metric_total,
        "total_profit": total_profit,
        "average_sales": metric_average,
        "total_customers": total_customers,

        "rows": int(len(data)),
        "column_count": int(len(data.columns)),

        "chart_labels": chart_labels,
        "chart_values": chart_values,

        "category_labels": category_labels,
        "category_values": category_values,

        "region_labels": region_labels,
        "region_values": region_values,

        "customer_segments": customer_segments,
        "segment_counts": segment_counts,
        "segmentation_records": segmentation_records,
        "segmentation_error": segmentation_error,

        "forecast_labels": forecast_labels,
        "forecast_values": forecast_values,
        "forecast_error": forecast_error,

        "churn_customers": churn_customers,
        "at_risk_customers": at_risk_customers,
        "active_customers": active_customers,
        "churn_percentage": churn_percentage,
        "churn_error": churn_error,

        "business_recommendations": business_recommendations,
        "recommendation_error": recommendation_error,

        "cleaning_report": cleaning_report,
        "cleaned_filename": cleaned_filename,

        "columns": columns,
        "records": records,

        "metric_column": metric_column,
        "sales_column": sales_column,
        "date_column": date_column
    }


# --------------------------------------------------
# CURRENT ANALYSIS
# --------------------------------------------------

def get_current_analysis():

    analysis_id = session.get("analysis_id")

    if not analysis_id:
        return None

    return analysis_store.get(analysis_id)


def render_analysis_page(template_name):

    analysis = get_current_analysis()

    if analysis is None:
        flash("Please upload your dataset first.")
        return redirect(url_for("home"))

    return render_template(template_name, **analysis)


# --------------------------------------------------
# HOME
# --------------------------------------------------

@app.route("/")
def home():
    return render_template("upload.html")


# --------------------------------------------------
# DATASET UPLOAD
# --------------------------------------------------

@app.route("/upload", methods=["POST"])
def upload_file():

    if "file" not in request.files:
        flash("Please select a dataset.")
        return redirect(url_for("home"))

    file = request.files["file"]

    if file.filename == "":
        flash("No file selected.")
        return redirect(url_for("home"))

    if not allowed_file(file.filename):
        flash("Please upload a CSV or Excel file.")
        return redirect(url_for("home"))

    try:

        original_filename = secure_filename(file.filename)

        if not original_filename:
            raise ValueError("Invalid filename.")

        extension = original_filename.rsplit(".", 1)[1].lower()

        unique_name = str(uuid.uuid4()) + "." + extension

        file_path = os.path.join(
            UPLOAD_FOLDER, unique_name
        )

        file.save(file_path)

        original_data = read_uploaded_dataset(
            file_path, extension
        )

        if original_data.empty:
            raise ValueError("The uploaded dataset is empty.")

        if len(original_data.columns) < 1:
            raise ValueError("No columns found in the dataset.")

        print("Original columns:", original_data.columns.tolist())

        cleaned_data, cleaning_report = clean_dataset(
            original_data.copy()
        )

        if cleaned_data.empty:
            raise ValueError(
                "No records available after data cleaning."
            )

        print("Cleaned columns:", cleaned_data.columns.tolist())

        cleaned_filename = (
            "cleaned_"
            + unique_name.rsplit(".", 1)[0]
            + ".csv"
        )

        cleaned_path = os.path.join(
            UPLOAD_FOLDER, cleaned_filename
        )

        cleaned_data.to_csv(
            cleaned_path,
            index=False,
            encoding="utf-8-sig"
        )

        analysis = analyse_dataset(
            cleaned_data,
            cleaning_report,
            cleaned_filename
        )

        analysis_id = str(uuid.uuid4())

        analysis_store[analysis_id] = analysis
        session["analysis_id"] = analysis_id

        print("Dataset uploaded and analysed successfully.")

        return redirect(url_for("overview"))

    except Exception as error:

        print("Upload error:", repr(error))

        flash(
            "Dataset processing failed: " + str(error)
        )

        return redirect(url_for("home"))


# --------------------------------------------------
# SEPARATE WEBPAGES
# --------------------------------------------------

@app.route("/overview")
def overview():
    return render_analysis_page("overview.html")


@app.route("/data-cleaning")
def data_cleaning():
    return render_analysis_page("data_cleaning.html")


@app.route("/sales-analytics")
def sales_analytics():
    return render_analysis_page("sales_analytics.html")


@app.route("/forecasting")
def forecasting():
    return render_analysis_page("forecasting.html")


@app.route("/segmentation")
def segmentation():
    return render_analysis_page("segmentation.html")


@app.route("/churn")
def churn():
    return render_analysis_page("churn.html")


@app.route("/recommendations")
def recommendations():
    return render_analysis_page("recommendations.html")


# --------------------------------------------------
# DOWNLOAD CLEANED DATASET
# --------------------------------------------------

@app.route("/download-cleaned/<filename>")
def download_cleaned(filename):

    safe_filename = secure_filename(filename)

    analysis = get_current_analysis()

    if analysis is None:
        flash("Please upload a dataset first.")
        return redirect(url_for("home"))

    if safe_filename != analysis.get("cleaned_filename"):
        flash("Invalid download request.")
        return redirect(url_for("data_cleaning"))

    return send_from_directory(
        UPLOAD_FOLDER,
        safe_filename,
        as_attachment=True
    )


# --------------------------------------------------
# ERROR HANDLERS
# --------------------------------------------------

@app.errorhandler(413)
def file_too_large(error):

    flash(
        "File is too large. Maximum upload size is 50 MB."
    )

    return redirect(url_for("home"))


@app.errorhandler(404)
def page_not_found(error):
    return "Page not found.", 404


# --------------------------------------------------
# APPLICATION START
# --------------------------------------------------

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 5000))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )









