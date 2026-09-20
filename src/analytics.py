from __future__ import annotations

import logging

import pandas as pd

from src.config import DATASET_PATH

logger = logging.getLogger(__name__)

_CATEGORICAL_FEATURES = [
    "Internet_Service", "Contract", "Payment_Method",
    "Online_Security", "Online_Backup", "Device_Protection",
    "Tech_Support", "Streaming_TV", "Streaming_Movies",
    "Phone_Service", "Paperless_Billing", "Is_Married",
    "Dependents",
]


class DatasetAnalytics:
    def __init__(self, path=DATASET_PATH):
        self.df = pd.read_csv(path)
        self.df.columns = self.df.columns.str.strip()
        self.df["Total_Charges"] = pd.to_numeric(
            self.df["Total_Charges"], errors="coerce"
        ).fillna(0.0)
        self.df["Senior_Citizen"] = self.df["Senior_Citizen"].map({0: "No", 1: "Yes"})
        logger.info("Dataset loaded: %d rows", len(self.df))

    def overall_stats(self) -> str:
        df = self.df
        total = len(df)
        churned = (df["Churn"] == "Yes").sum()
        rate = churned / total * 100

        avg_tenure = df["tenure"].mean()
        avg_monthly = df["Monthly_Charges"].mean()
        avg_total = df["Total_Charges"].mean()

        churner_tenure = df.loc[df["Churn"] == "Yes", "tenure"].mean()
        stayer_tenure = df.loc[df["Churn"] == "No", "tenure"].mean()

        return (
            f"Dataset Overview:\n"
            f"  Total customers: {total}\n"
            f"  Churned: {churned} ({rate:.1f}%)\n"
            f"  Stayed: {total - churned} ({100 - rate:.1f}%)\n\n"
            f"Averages:\n"
            f"  Tenure: {avg_tenure:.1f} months\n"
            f"  Monthly charges: ${avg_monthly:.2f}\n"
            f"  Total charges: ${avg_total:.2f}\n\n"
            f"Tenure by group:\n"
            f"  Churners avg tenure: {churner_tenure:.1f} months\n"
            f"  Stayers avg tenure: {stayer_tenure:.1f} months"
        )

    def churn_by_feature(self, feature: str) -> str:
        col_map = {k.lower().replace("_", ""): k for k in self.df.columns}
        col_map.update({k.lower(): k for k in self.df.columns})
        clean_key = feature.lower().replace("_", "").replace(" ", "")
        col = col_map.get(clean_key) or col_map.get(feature.lower())

        if col is None or col not in self.df.columns:
            return f"Feature '{feature}' not found. Available: {', '.join(_CATEGORICAL_FEATURES)}"

        grouped = self.df.groupby(col)["Churn"].apply(
            lambda x: f"{(x == 'Yes').sum()}/{len(x)} ({(x == 'Yes').mean() * 100:.1f}%)"
        )
        lines = [f"Churn rate by {col}:"]
        for val, stat in grouped.items():
            lines.append(f"  {val}: {stat}")
        return "\n".join(lines)

    def high_risk_segments(self) -> str:
        df = self.df
        segments = []

        fiber_mtm = df[(df["Internet_Service"] == "Fiber optic") & (df["Contract"] == "Month-to-month")]
        if len(fiber_mtm) > 0:
            rate = (fiber_mtm["Churn"] == "Yes").mean() * 100
            segments.append(f"  Fiber optic + Month-to-month: {rate:.1f}% churn ({len(fiber_mtm)} customers)")

        echeck = df[df["Payment_Method"] == "Electronic check"]
        if len(echeck) > 0:
            rate = (echeck["Churn"] == "Yes").mean() * 100
            segments.append(f"  Electronic check payers: {rate:.1f}% churn ({len(echeck)} customers)")

        no_support = df[(df["Tech_Support"] == "No") & (df["Online_Security"] == "No")]
        if len(no_support) > 0:
            rate = (no_support["Churn"] == "Yes").mean() * 100
            segments.append(f"  No tech support + No online security: {rate:.1f}% churn ({len(no_support)} customers)")

        short_tenure = df[df["tenure"] <= 6]
        if len(short_tenure) > 0:
            rate = (short_tenure["Churn"] == "Yes").mean() * 100
            segments.append(f"  Tenure <= 6 months: {rate:.1f}% churn ({len(short_tenure)} customers)")

        senior = df[df["Senior_Citizen"] == "Yes"]
        if len(senior) > 0:
            rate = (senior["Churn"] == "Yes").mean() * 100
            segments.append(f"  Senior citizens: {rate:.1f}% churn ({len(senior)} customers)")

        return "High-Risk Customer Segments:\n" + "\n".join(segments)

    def query(self, topic: str) -> str:
        topic_lower = topic.lower()

        if any(w in topic_lower for w in ["overview", "summary", "overall", "general", "total"]):
            return self.overall_stats()

        if any(w in topic_lower for w in ["risk", "segment", "high risk", "dangerous", "worst"]):
            return self.high_risk_segments()

        for feat in _CATEGORICAL_FEATURES:
            if feat.lower().replace("_", " ") in topic_lower or feat.lower() in topic_lower:
                return self.churn_by_feature(feat)

        if any(w in topic_lower for w in ["internet", "fiber", "dsl"]):
            return self.churn_by_feature("Internet_Service")
        if any(w in topic_lower for w in ["contract", "month", "year"]):
            return self.churn_by_feature("Contract")
        if any(w in topic_lower for w in ["payment", "check", "credit", "bank"]):
            return self.churn_by_feature("Payment_Method")
        if any(w in topic_lower for w in ["senior", "age", "old"]):
            return self.churn_by_feature("Senior_Citizen")
        if any(w in topic_lower for w in ["married", "marital", "spouse"]):
            return self.churn_by_feature("Is_Married")
        if any(w in topic_lower for w in ["security"]):
            return self.churn_by_feature("Online_Security")
        if any(w in topic_lower for w in ["support", "tech"]):
            return self.churn_by_feature("Tech_Support")
        if any(w in topic_lower for w in ["streaming", "tv", "movie"]):
            return self.churn_by_feature("Streaming_TV") + "\n\n" + self.churn_by_feature("Streaming_Movies")
        if any(w in topic_lower for w in ["billing", "paperless"]):
            return self.churn_by_feature("Paperless_Billing")

        return self.overall_stats() + "\n\n" + self.high_risk_segments()
