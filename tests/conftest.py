import pytest

from src.predict import ChurnPredictor
from src.schemas import CustomerFeatures


@pytest.fixture(scope="session")
def predictor():
    return ChurnPredictor()


@pytest.fixture
def high_risk_customer():
    return CustomerFeatures(
        gender="Female", Senior_Citizen=0, Is_Married="No", Dependents="No",
        tenure=2, Phone_Service="Yes", Dual="No",
        Internet_Service="Fiber optic", Online_Security="No", Online_Backup="No",
        Device_Protection="No", Tech_Support="No", Streaming_TV="No",
        Streaming_Movies="No", Contract="Month-to-month", Paperless_Billing="Yes",
        Payment_Method="Electronic check", Monthly_Charges=70.7, Total_Charges=151.65,
    )


@pytest.fixture
def low_risk_customer():
    return CustomerFeatures(
        gender="Male", Senior_Citizen=0, Is_Married="Yes", Dependents="Yes",
        tenure=60, Phone_Service="Yes", Dual="Yes",
        Internet_Service="DSL", Online_Security="Yes", Online_Backup="Yes",
        Device_Protection="Yes", Tech_Support="Yes", Streaming_TV="Yes",
        Streaming_Movies="Yes", Contract="Two year", Paperless_Billing="No",
        Payment_Method="Credit card (automatic)", Monthly_Charges=100.0, Total_Charges=6000.0,
    )
