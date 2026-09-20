from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class CustomerFeatures(BaseModel):
    gender: Literal["Male", "Female"]
    senior_citizen: Literal[0, 1] = Field(alias="Senior_Citizen")
    is_married: Literal["Yes", "No"] = Field(alias="Is_Married")
    dependents: Literal["Yes", "No"] = Field(alias="Dependents")
    tenure: int = Field(ge=0, le=72)
    phone_service: Literal["Yes", "No"] = Field(alias="Phone_Service")
    dual: Literal["Yes", "No", "No phone service"] = Field(alias="Dual")
    internet_service: Literal["DSL", "Fiber optic", "No"] = Field(alias="Internet_Service")
    online_security: Literal["Yes", "No", "No internet service"] = Field(alias="Online_Security")
    online_backup: Literal["Yes", "No", "No internet service"] = Field(alias="Online_Backup")
    device_protection: Literal["Yes", "No", "No internet service"] = Field(alias="Device_Protection")
    tech_support: Literal["Yes", "No", "No internet service"] = Field(alias="Tech_Support")
    streaming_tv: Literal["Yes", "No", "No internet service"] = Field(alias="Streaming_TV")
    streaming_movies: Literal["Yes", "No", "No internet service"] = Field(alias="Streaming_Movies")
    contract: Literal["Month-to-month", "One year", "Two year"] = Field(alias="Contract")
    paperless_billing: Literal["Yes", "No"] = Field(alias="Paperless_Billing")
    payment_method: Literal[
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)",
    ] = Field(alias="Payment_Method")
    monthly_charges: float = Field(ge=0, alias="Monthly_Charges")
    total_charges: float = Field(ge=0, alias="Total_Charges")

    model_config = {"populate_by_name": True}

    def to_dataframe_row(self) -> dict:
        return {
            "gender": self.gender,
            "Senior_Citizen": self.senior_citizen,
            "Is_Married": self.is_married,
            "Dependents": self.dependents,
            "tenure": self.tenure,
            "Phone_Service": self.phone_service,
            "Dual": self.dual,
            "Internet_Service": self.internet_service,
            "Online_Security": self.online_security,
            "Online_Backup": self.online_backup,
            "Device_Protection": self.device_protection,
            "Tech_Support": self.tech_support,
            "Streaming_TV": self.streaming_tv,
            "Streaming_Movies": self.streaming_movies,
            "Contract": self.contract,
            "Paperless_Billing": self.paperless_billing,
            "Payment_Method": self.payment_method,
            "Monthly_Charges": self.monthly_charges,
            "Total_Charges": self.total_charges,
        }


class PartialCustomerFeatures(BaseModel):
    senior_citizen: Optional[Literal[0, 1]] = None
    is_married: Optional[Literal["Yes", "No"]] = None
    dependents: Optional[Literal["Yes", "No"]] = None
    tenure: Optional[int] = None
    phone_service: Optional[Literal["Yes", "No"]] = None
    internet_service: Optional[Literal["DSL", "Fiber optic", "No"]] = None
    online_security: Optional[Literal["Yes", "No", "No internet service"]] = None
    online_backup: Optional[Literal["Yes", "No", "No internet service"]] = None
    device_protection: Optional[Literal["Yes", "No", "No internet service"]] = None
    tech_support: Optional[Literal["Yes", "No", "No internet service"]] = None
    streaming_tv: Optional[Literal["Yes", "No", "No internet service"]] = None
    streaming_movies: Optional[Literal["Yes", "No", "No internet service"]] = None
    contract: Optional[Literal["Month-to-month", "One year", "Two year"]] = None
    paperless_billing: Optional[Literal["Yes", "No"]] = None
    payment_method: Optional[Literal[
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)",
    ]] = None
    monthly_charges: Optional[float] = None
    total_charges: Optional[float] = None

    def apply_logical_defaults(self) -> None:
        if self.internet_service == "No":
            for field in (
                "online_security", "online_backup", "device_protection",
                "tech_support", "streaming_tv", "streaming_movies",
            ):
                if getattr(self, field) is None:
                    setattr(self, field, "No internet service")

    def missing_fields(self) -> list[str]:
        self.apply_logical_defaults()
        return [name for name in type(self).model_fields if getattr(self, name) is None]

    def is_complete(self) -> bool:
        return len(self.missing_fields()) == 0

    def to_customer_features(self) -> CustomerFeatures:
        self.apply_logical_defaults()
        data = self.model_dump()
        data["gender"] = "Male"
        data["dual"] = "Yes"
        return CustomerFeatures(**data)


class PredictionResult(BaseModel):
    churn_probability: float
    churn_prediction: bool
    top_factors: list[dict]
    decision_tree_path: str


class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatResponse(BaseModel):
    session_id: str
    response: str
    prediction: PredictionResult | None = None
