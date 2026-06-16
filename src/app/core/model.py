import xgboost as xgb
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional
import threading
from app.utils.feature_engineering import get_lag, get_moving_average

TEST_PERIOD_START = pd.Timestamp("2023-04-01")


class ForecastingModel:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(ForecastingModel, cls).__new__(cls)
                    cls._instance.model = None
                    cls._instance.history = None
                    cls._instance.initialized = False
        return cls._instance

    def load(self):
        with self._lock:
            if self.initialized:
                return

            base_path = Path(__file__).resolve().parent.parent.parent
            model_path = base_path / "models" / "optuna_split_0.ubj"
            history_path = base_path / "data" / "service_history.parquet"

            print(f"Loading model from {model_path}...")
            self.model = xgb.Booster()
            self.model.load_model(model_path)

            print(f"Loading history from {history_path}...")
            self.history = pd.read_parquet(history_path)
            self.history["datetime"] = pd.to_datetime(self.history["datetime"])
            self.history = self.history.sort_values("datetime")

            self.initialized = True

    def predict(self, input_data: dict) -> float:
        """
        Generate prediction for a single data point.
        input_data must contain:
        - timestamp (datetime)
        - county (int)
        - is_business (bool/int)
        - product_type (int)
        - is_consumption (bool/int)
        - eic_count (int) - from Client
        - installed_capacity (float) - from Client
        """
        if not self.initialized:
            self.load()

        ts = pd.to_datetime(input_data["timestamp"])
        county = int(input_data["county"])
        is_business = int(input_data["is_business"])
        product_type = int(input_data["product_type"])
        is_consumption = int(input_data["is_consumption"])

        # Filter history for this specific unit
        # Max window is 14 days + 48h shift = 16 days
        # 30 days prior

        start_bound = ts - pd.Timedelta(days=30)

        mask = (
            (self.history["county"] == county)
            & (self.history["is_business"] == is_business)
            & (self.history["product_type"] == product_type)
            & (self.history["is_consumption"] == is_consumption)
            & (self.history["datetime"] >= start_bound)
            & (self.history["datetime"] <= ts)
        )

        unit_df = self.history.loc[mask].copy()

        row_mask = unit_df["datetime"] == ts

        if not row_mask.any():
            raise ValueError(
                f"No history/weather data found for timestamp {ts}"
            )

        idx = unit_df.index[row_mask][0]
        unit_df.loc[idx, "eic_count"] = input_data["eic_count"]
        unit_df.loc[idx, "installed_capacity"] = input_data[
            "installed_capacity"
        ]

        # lags - 2, 3, 7
        TARGET_C = [
            "county",
            "product_type",
            "is_business",
            "is_consumption",
            "datetime",
        ]

        unit_df = unit_df.sort_values("datetime")

        for lag in [2, 3, 7]:
            lag_df = get_lag(
                unit_df[TARGET_C + ["target"]], lag=lag, columns=["target"]
            )
            unit_df = unit_df.merge(lag_df, how="left", on=TARGET_C)

        CATEGORICAL_C = [
            "county",
            "product_type",
            "is_business",
            "is_consumption",
        ]

        dfgb = (
            unit_df.set_index("datetime")
            .sort_index()
            .groupby(CATEGORICAL_C, observed=True, as_index=False)
        )

        for window in [24, 24 * 3, 24 * 7, 24 * 14]:
            ma_df = get_moving_average(dfgb, columns=["target"], window=window)
            unit_df = unit_df.merge(ma_df, how="left", on=TARGET_C)

        target_row = unit_df[unit_df["datetime"] == ts].iloc[0].copy()

        target_row["t_over_cap"] = (
            target_row["2d_lag_target"] / target_row["installed_capacity"]
        )
        target_row["t_over_eic"] = (
            target_row["2d_lag_target"] / target_row["eic_count"]
        )
        target_row["cap_per_eic"] = (
            target_row["installed_capacity"] / target_row["eic_count"]
        )

        target_row = target_row.fillna(0).replace([np.inf, -np.inf], 0)

        drop_labels = [
            c
            for c in ["datetime", "data_block_id", "date", "target"]
            if c in target_row.index
        ]
        features = target_row.drop(drop_labels)
        # features = target_row.drop(
        #     ["datetime", "data_block_id", "date", "target"]
        # )

        X = pd.DataFrame([features])

        int_cols = ["is_business", "is_consumption", "dst"]
        for c in int_cols:
            if c in X.columns:
                X[c] = X[c].astype(int)

        cat_cols = [
            "county",
            "product_type",
            "national_holiday",
            "observance_day",
            "season_event",
        ]
        for c in cat_cols:
            if c in X.columns:
                X[c] = X[c].astype(int).astype("category")

        dtest = xgb.DMatrix(X, enable_categorical=True)

        prediction = self.model.predict(dtest)

        return float(prediction[0])

    def predict_range(
        self, params: dict, timestamps: list[pd.Timestamp]
    ) -> list[dict]:
        if not self.initialized:
            self.load()

        county = int(params["county"])
        is_business = int(params["is_business"])
        product_type = int(params["product_type"])
        is_consumption = int(params["is_consumption"])

        ts_min = min(timestamps)
        ts_max = max(timestamps)

        start_bound = ts_min - pd.Timedelta(days=30)

        mask = (
            (self.history["county"] == county)
            & (self.history["is_business"] == is_business)
            & (self.history["product_type"] == product_type)
            & (self.history["is_consumption"] == is_consumption)
            & (self.history["datetime"] >= start_bound)
            & (self.history["datetime"] <= ts_max)
        )
        unit_df = self.history.loc[mask].copy().sort_values("datetime")

        TARGET_C = [
            "county",
            "product_type",
            "is_business",
            "is_consumption",
            "datetime",
        ]
        CATEGORICAL_C = [
            "county",
            "product_type",
            "is_business",
            "is_consumption",
        ]

        for lag in [2, 3, 7]:
            lag_df = get_lag(
                unit_df[TARGET_C + ["target"]], lag=lag, columns=["target"]
            )
            unit_df = unit_df.merge(lag_df, how="left", on=TARGET_C)

        dfgb = (
            unit_df.set_index("datetime")
            .sort_index()
            .groupby(CATEGORICAL_C, observed=True, as_index=False)
        )
        for window in [24, 24 * 3, 24 * 7, 24 * 14]:
            ma_df = get_moving_average(dfgb, columns=["target"], window=window)
            unit_df = unit_df.merge(ma_df, how="left", on=TARGET_C)

        result_df = unit_df[unit_df["datetime"].isin(timestamps)].copy()

        result_df["t_over_cap"] = (
            result_df["2d_lag_target"] / result_df["installed_capacity"]
        )
        result_df["t_over_eic"] = (
            result_df["2d_lag_target"] / result_df["eic_count"]
        )
        result_df["cap_per_eic"] = (
            result_df["installed_capacity"] / result_df["eic_count"]
        )
        result_df = result_df.fillna(0).replace([np.inf, -np.inf], 0)

        drop_cols = [
            c
            for c in ["datetime", "data_block_id", "date", "target"]
            if c in result_df.columns
        ]
        X = result_df.drop(columns=drop_cols)

        int_cols = ["is_business", "is_consumption", "dst"]
        for c in int_cols:
            if c in X.columns:
                X[c] = X[c].astype(int)

        cat_cols = [
            "county",
            "product_type",
            "national_holiday",
            "observance_day",
            "season_event",
        ]
        for c in cat_cols:
            if c in X.columns:
                X[c] = X[c].astype(int).astype("category")

        if X.empty:
            return [
                {
                    "timestamp": ts.isoformat(),
                    "prediction": None,
                    "error": "no history data",
                }
                for ts in timestamps
            ]

        dtest = xgb.DMatrix(X, enable_categorical=True)
        predictions = self.model.predict(dtest)

        # Map predictions back to timestamps
        result_timestamps = result_df["datetime"].tolist()
        pred_map = {
            ts: float(p) for ts, p in zip(result_timestamps, predictions)
        }
        target_map = {
            ts: float(t) if not pd.isna(t) else None
            for ts, t in zip(result_timestamps, result_df["target"])
        }

        # Build output, marking missing timestamps
        output = []
        for ts in timestamps:
            item = {
                "timestamp": ts.isoformat(),
                "prediction": None,
                "actual": None,
            }
            if ts in pred_map:
                item["prediction"] = pred_map[ts]
                if ts >= TEST_PERIOD_START:
                    item["actual"] = target_map.get(ts)
            else:
                item["error"] = "no history data"
            output.append(item)

        return output


model_instance = ForecastingModel()
