import os
import requests
import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta


st.set_page_config(layout="wide")

# Read API URL from environment variable, default to docker-compose service name 'api' or 127.0.0.1 locally
API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000")

st.title("Enefit Energy Forecast")

VALID_COMBINATIONS = [
    {"county": 0, "is_business": False, "product_type": 1},
    {"county": 0, "is_business": False, "product_type": 2},
    {"county": 0, "is_business": False, "product_type": 3},
    {"county": 0, "is_business": True, "product_type": 0},
    {"county": 0, "is_business": True, "product_type": 1},
    {"county": 0, "is_business": True, "product_type": 2},
    {"county": 0, "is_business": True, "product_type": 3},
    {"county": 1, "is_business": False, "product_type": 1},
    {"county": 1, "is_business": False, "product_type": 3},
    {"county": 1, "is_business": True, "product_type": 3},
    {"county": 2, "is_business": False, "product_type": 1},
    {"county": 2, "is_business": False, "product_type": 3},
    {"county": 2, "is_business": True, "product_type": 1},
    {"county": 2, "is_business": True, "product_type": 3},
    {"county": 3, "is_business": False, "product_type": 1},
    {"county": 3, "is_business": False, "product_type": 3},
    {"county": 3, "is_business": True, "product_type": 1},
    {"county": 3, "is_business": True, "product_type": 3},
    {"county": 4, "is_business": False, "product_type": 1},
    {"county": 4, "is_business": False, "product_type": 3},
    {"county": 4, "is_business": True, "product_type": 0},
    {"county": 4, "is_business": True, "product_type": 1},
    {"county": 4, "is_business": True, "product_type": 3},
    {"county": 5, "is_business": False, "product_type": 1},
    {"county": 5, "is_business": False, "product_type": 3},
    {"county": 5, "is_business": True, "product_type": 0},
    {"county": 5, "is_business": True, "product_type": 1},
    {"county": 5, "is_business": True, "product_type": 3},
    {"county": 6, "is_business": True, "product_type": 3},
    {"county": 7, "is_business": False, "product_type": 1},
    {"county": 7, "is_business": False, "product_type": 2},
    {"county": 7, "is_business": False, "product_type": 3},
    {"county": 7, "is_business": True, "product_type": 0},
    {"county": 7, "is_business": True, "product_type": 1},
    {"county": 7, "is_business": True, "product_type": 3},
    {"county": 8, "is_business": False, "product_type": 1},
    {"county": 8, "is_business": False, "product_type": 3},
    {"county": 8, "is_business": True, "product_type": 3},
    {"county": 9, "is_business": False, "product_type": 1},
    {"county": 9, "is_business": False, "product_type": 3},
    {"county": 9, "is_business": True, "product_type": 1},
    {"county": 9, "is_business": True, "product_type": 3},
    {"county": 10, "is_business": False, "product_type": 1},
    {"county": 10, "is_business": False, "product_type": 3},
    {"county": 10, "is_business": True, "product_type": 1},
    {"county": 10, "is_business": True, "product_type": 2},
    {"county": 10, "is_business": True, "product_type": 3},
    {"county": 11, "is_business": False, "product_type": 1},
    {"county": 11, "is_business": False, "product_type": 2},
    {"county": 11, "is_business": False, "product_type": 3},
    {"county": 11, "is_business": True, "product_type": 0},
    {"county": 11, "is_business": True, "product_type": 1},
    {"county": 11, "is_business": True, "product_type": 2},
    {"county": 11, "is_business": True, "product_type": 3},
    {"county": 12, "is_business": True, "product_type": 3},
    {"county": 13, "is_business": False, "product_type": 1},
    {"county": 13, "is_business": False, "product_type": 3},
    {"county": 13, "is_business": True, "product_type": 1},
    {"county": 13, "is_business": True, "product_type": 3},
    {"county": 14, "is_business": False, "product_type": 1},
    {"county": 14, "is_business": False, "product_type": 3},
    {"county": 14, "is_business": True, "product_type": 1},
    {"county": 14, "is_business": True, "product_type": 2},
    {"county": 14, "is_business": True, "product_type": 3},
    {"county": 15, "is_business": False, "product_type": 1},
    {"county": 15, "is_business": False, "product_type": 3},
    {"county": 15, "is_business": True, "product_type": 0},
    {"county": 15, "is_business": True, "product_type": 1},
    {"county": 15, "is_business": True, "product_type": 3},
]

COUNTY_MAPPING = {
    0: "Harjumaa",
    1: "Hiiumaa",
    2: "Ida-Virumaa",
    3: "Järvamaa",
    4: "Jõgevamaa",
    5: "Lääne-Virumaa",
    6: "Läänemaa",
    7: "Pärnumaa",
    8: "Põlvamaa",
    9: "Raplamaa",
    10: "Saaremaa",
    11: "Tartumaa",
    12: "Unknown",
    13: "Valgamaa",
    14: "Viljandimaa",
    15: "Võrumaa",
}
PRODUCT_MAPPING = {
    0: "Combined contract",
    1: "Fixed contract",
    2: "General service contract",
    3: "Spot contract",
}

with st.sidebar:
    st.header("Parameters")

    # 1. Select County
    valid_counties = sorted(list(set(c["county"] for c in VALID_COMBINATIONS)))
    county = st.selectbox(
        "County",
        valid_counties,
        format_func=lambda x: COUNTY_MAPPING.get(x, f"County {x}"),
    )

    # 2. Select Client Type constrained by County
    valid_business_flags = sorted(
        list(
            set(
                c["is_business"]
                for c in VALID_COMBINATIONS
                if c["county"] == county
            )
        ),
        reverse=True,
    )
    is_business = st.selectbox(
        "Client Type",
        valid_business_flags,
        format_func=lambda x: (
            "Business clients" if x else "Non-business clients"
        ),
    )

    # 3. Select Product Type constrained by County and Client Type
    valid_product_types = sorted(
        list(
            set(
                c["product_type"]
                for c in VALID_COMBINATIONS
                if c["county"] == county and c["is_business"] == is_business
            )
        )
    )
    product_type = st.selectbox(
        "Product Type",
        valid_product_types,
        format_func=lambda x: PRODUCT_MAPPING.get(x, "Unknown"),
    )

    # Both Consumption and Production are always available for any
    # valid unit segment
    is_consumption = st.selectbox(
        "Target Type",
        [True, False],
        format_func=lambda x: "Consumption" if x else "Production",
    )

    st.divider()

    st.header("Date Range (Max 30 Days)")
    # Defaulting strictly in the range of the test set data history
    # for a reliable example
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start", value=datetime(2023, 4, 1))
        start_hour = st.number_input("Start hour", 0, 23, 0)
    with col2:
        end_date = st.date_input("End", value=datetime(2023, 4, 7))
        end_hour = st.number_input("End hour", 0, 23, 23)

    run = st.button("Get Forecast", type="primary", width="stretch")

if run:
    start_dt = datetime.combine(start_date, datetime.min.time()).replace(
        hour=start_hour
    )
    end_dt = datetime.combine(end_date, datetime.min.time()).replace(
        hour=end_hour
    )

    if end_dt <= start_dt:
        st.error("End date must be after start date.")
    elif (end_dt - start_dt).days > 30:
        st.error("Range cannot exceed 30 days.")
    else:
        with st.spinner("Fetching predictions..."):
            try:
                resp = requests.post(
                    f"{API_URL}/predict/range",
                    json={
                        "county": county,
                        "is_business": is_business,
                        "product_type": product_type,
                        "is_consumption": is_consumption,
                        "start_datetime": start_dt.isoformat(),
                        "end_datetime": end_dt.isoformat(),
                    },
                )
            except Exception as e:
                st.error(f"Failed to connect to API: {e}")
                resp = None

        if resp is not None:
            if resp.status_code == 200:
                data = resp.json()["predictions"]
                df = pd.DataFrame(data)
                df["timestamp"] = pd.to_datetime(df["timestamp"])

                # Check for errors in specific rows (prediction must exist)
                valid_df = df.dropna(subset=["prediction"])

                if valid_df.empty:
                    st.warning(
                        "No predictions could be generated. This typically means there is no historical data available in the service range for this configuration."
                    )
                else:
                    client_type = (
                        "Business clients"
                        if is_business
                        else "Non-business clients"
                    )
                    target_type = (
                        "Consumption" if is_consumption else "Production"
                    )
                    county_name = COUNTY_MAPPING.get(
                        county, f"County {county}"
                    )
                    prod_name = PRODUCT_MAPPING.get(
                        product_type, f"Product {product_type}"
                    )

                    actual_valid = pd.DataFrame()
                    mae_val = None
                    if "actual" in valid_df.columns:
                        actual_valid = valid_df.dropna(
                            subset=["actual", "prediction"]
                        )
                        if not actual_valid.empty:
                            mae_val = (
                                (
                                    actual_valid["prediction"]
                                    - actual_valid["actual"]
                                )
                                .abs()
                                .mean()
                            )

                    plot_cols = ["prediction"]
                    if not actual_valid.empty:
                        plot_cols.append("actual")

                    fig = px.line(
                        valid_df,
                        x="timestamp",
                        y=plot_cols,
                        title=f"{county_name} | {client_type} | {prod_name} | {target_type}",
                        labels={
                            "value": "KWh",
                            "timestamp": "Date & Time",
                            "variable": "Series",
                        },
                        template="plotly_white",
                    )

                    test_start = pd.Timestamp("2023-04-01")
                    if (
                        valid_df["timestamp"].min()
                        <= test_start
                        <= valid_df["timestamp"].max()
                    ):
                        fig.add_vline(
                            x=test_start.timestamp() * 1000,
                            line_dash="dash",
                            line_color="black",
                        )
                        fig.add_annotation(
                            x=test_start.timestamp() * 1000,
                            y=1.05,
                            yref="paper",
                            text="Test Period Begins",
                            showarrow=False,
                            font=dict(color="black", size=11),
                        )

                    fig.update_layout(
                        hovermode="x unified",
                        xaxis_title="",
                        yaxis_title="Energy (KWh)",
                        legend_title=None,
                        height=700,
                        hoverlabel=dict(font_size=20),
                    )

                    st.plotly_chart(fig, width="stretch")

                    m_cols = (
                        st.columns(4) if mae_val is not None else st.columns(3)
                    )
                    m_cols[0].metric(
                        "Total Hours Predicted", f"{len(valid_df)}"
                    )
                    m_cols[1].metric(
                        "Max value (KWh)",
                        f"{valid_df['prediction'].max():.2f}",
                    )
                    m_cols[2].metric(
                        "Min value (KWh)",
                        f"{valid_df['prediction'].min():.2f}",
                    )
                    if mae_val is not None:
                        m_cols[3].metric("MAE (Test Set)", f"{mae_val:.2f}")

            else:
                detail = resp.json().get("detail", "Unknown Error")
                st.error(
                    f"API returned an error (Status {resp.status_code}): {detail}"
                )
