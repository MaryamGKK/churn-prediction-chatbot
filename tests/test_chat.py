import pytest

from src.chatbot.session import ChatSession, SessionStore
from src.schemas import PartialCustomerFeatures


def test_session_store_creates_session():
    store = SessionStore(max_sessions=10)
    session = store.get_or_create("test-1")
    assert session.session_id == "test-1"
    assert session.language == "en"


def test_session_store_returns_same_session():
    store = SessionStore(max_sessions=10)
    s1 = store.get_or_create("test-1")
    s1.add_message("user", "hello")
    s2 = store.get_or_create("test-1")
    assert len(s2.history) == 1


def test_session_store_evicts_oldest():
    store = SessionStore(max_sessions=2)
    store.get_or_create("a")
    store.get_or_create("b")
    store.get_or_create("c")
    assert "a" not in store._sessions
    assert "b" in store._sessions
    assert "c" in store._sessions


def test_partial_features_missing_fields():
    p = PartialCustomerFeatures()
    assert len(p.missing_fields()) == 17


def test_partial_features_logical_defaults_internet():
    p = PartialCustomerFeatures(internet_service="No")
    p.apply_logical_defaults()
    assert p.online_security == "No internet service"
    assert p.online_backup == "No internet service"
    assert p.device_protection == "No internet service"
    assert p.tech_support == "No internet service"
    assert p.streaming_tv == "No internet service"
    assert p.streaming_movies == "No internet service"


def test_partial_features_logical_defaults_phone():
    p = PartialCustomerFeatures(phone_service="No")
    p.apply_logical_defaults()
    assert p.phone_service == "No"


def test_partial_features_complete():
    p = PartialCustomerFeatures(
        senior_citizen=0, is_married="No", dependents="No",
        tenure=5, phone_service="No", internet_service="No",
        contract="Month-to-month", paperless_billing="Yes",
        payment_method="Electronic check", monthly_charges=50.0, total_charges=250.0,
    )
    assert p.is_complete()


def test_partial_to_customer_features():
    p = PartialCustomerFeatures(
        senior_citizen=0, is_married="No", dependents="No",
        tenure=5, phone_service="No", internet_service="No",
        contract="Month-to-month", paperless_billing="Yes",
        payment_method="Electronic check", monthly_charges=50.0, total_charges=250.0,
    )
    cf = p.to_customer_features()
    assert cf.gender == "Male"
    assert cf.dual == "Yes"
    assert cf.online_security == "No internet service"


def test_chat_session_history():
    session = ChatSession("test")
    session.add_message("user", "hello")
    session.add_message("assistant", "hi there")
    text = session.get_conversation_text()
    assert "user: hello" in text
    assert "assistant: hi there" in text
