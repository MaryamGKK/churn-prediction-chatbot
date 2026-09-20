import pytest

from src.analytics import DatasetAnalytics
from src.chatbot.chain import _build_tools, _detect_language, _parse_tool_call, _ToolContext
from src.chatbot.session import ChatSession
from src.predict import ChurnPredictor


@pytest.fixture(scope="module")
def predictor():
    return ChurnPredictor()


@pytest.fixture(scope="module")
def analytics():
    return DatasetAnalytics()


@pytest.fixture
def session():
    return ChatSession("test-agent")


@pytest.fixture
def ctx(session, predictor, analytics):
    return _ToolContext(session, predictor, analytics)


@pytest.fixture
def tools(ctx):
    return {t.name: t for t in _build_tools(ctx)}


def test_detect_language_english():
    assert _detect_language("Hello, I have a customer") == "en"


def test_detect_language_arabic():
    assert _detect_language("مرحبا، لدي عميل") == "ar"


def test_detect_language_mixed():
    assert _detect_language("The customer name is احمد") == "ar"


def test_parse_tool_call_json():
    result = _parse_tool_call('{"tool": "predict_churn", "args": {}}')
    assert result == ("predict_churn", {})


def test_parse_tool_call_markdown_fenced():
    text = '```json\n{"tool": "get_collection_status", "args": {}}\n```'
    result = _parse_tool_call(text)
    assert result == ("get_collection_status", {})


def test_parse_tool_call_plain_text():
    result = _parse_tool_call("Sure, let me ask about the contract type.")
    assert result is None


def test_update_features_saves_values(tools, ctx):
    result = tools["update_customer_features"].invoke(
        {"features": {"tenure": 24, "contract": "Month-to-month"}}
    )
    assert "Updated: tenure, contract" in result
    assert ctx.session.partial_features.tenure == 24
    assert ctx.session.partial_features.contract == "Month-to-month"


def test_update_features_converts_types(tools, ctx):
    tools["update_customer_features"].invoke(
        {"features": {"monthly_charges": "70.5", "senior_citizen": "0"}}
    )
    assert ctx.session.partial_features.monthly_charges == 70.5
    assert ctx.session.partial_features.senior_citizen == 0


def test_update_features_ignores_invalid(tools, ctx):
    result = tools["update_customer_features"].invoke(
        {"features": {"nonexistent_field": "value"}}
    )
    assert "No fields updated" in result


def test_update_features_applies_logical_defaults(tools, ctx):
    tools["update_customer_features"].invoke(
        {"features": {"internet_service": "No"}}
    )
    assert ctx.session.partial_features.online_security == "No internet service"


def test_collection_status_shows_missing(tools):
    result = tools["get_collection_status"].invoke({})
    assert "Missing" in result


def test_predict_churn_rejects_incomplete(tools):
    result = tools["predict_churn"].invoke({})
    assert "Cannot predict" in result


def test_predict_churn_runs_when_complete(tools, ctx):
    ctx.session.partial_features.senior_citizen = 0
    ctx.session.partial_features.is_married = "No"
    ctx.session.partial_features.dependents = "No"
    ctx.session.partial_features.tenure = 2
    ctx.session.partial_features.phone_service = "Yes"
    ctx.session.partial_features.internet_service = "Fiber optic"
    ctx.session.partial_features.online_security = "No"
    ctx.session.partial_features.online_backup = "No"
    ctx.session.partial_features.device_protection = "No"
    ctx.session.partial_features.tech_support = "No"
    ctx.session.partial_features.streaming_tv = "No"
    ctx.session.partial_features.streaming_movies = "No"
    ctx.session.partial_features.contract = "Month-to-month"
    ctx.session.partial_features.paperless_billing = "Yes"
    ctx.session.partial_features.payment_method = "Electronic check"
    ctx.session.partial_features.monthly_charges = 70.7
    ctx.session.partial_features.total_charges = 151.65

    result = tools["predict_churn"].invoke({})
    assert "LIKELY TO CHURN" in result
    assert "Churn Probability" in result
    assert "Top Factors" in result
    assert ctx.prediction_result is not None
    assert ctx.prediction_result.churn_probability > 0.5


def test_fallback_response_english():
    from src.chatbot.chain import ChurnChatbot

    class FakePredictor:
        pass

    bot = ChurnChatbot.__new__(ChurnChatbot)
    bot.predictor = FakePredictor()
    session = ChatSession("test")
    resp = bot._fallback_response(session)
    assert "I need more information" in resp


def test_fallback_response_arabic():
    from src.chatbot.chain import ChurnChatbot

    class FakePredictor:
        pass

    bot = ChurnChatbot.__new__(ChurnChatbot)
    bot.predictor = FakePredictor()
    session = ChatSession("test")
    session.language = "ar"
    resp = bot._fallback_response(session)
    assert "معلومات" in resp


def test_analyze_data_overview(tools):
    result = tools["analyze_data"].invoke({"topic": "overview"})
    assert "Total customers" in result
    assert "Churned" in result


def test_analyze_data_contract(tools):
    result = tools["analyze_data"].invoke({"topic": "churn by contract"})
    assert "Month-to-month" in result


def test_analyze_data_high_risk(tools):
    result = tools["analyze_data"].invoke({"topic": "high risk segments"})
    assert "High-Risk" in result
