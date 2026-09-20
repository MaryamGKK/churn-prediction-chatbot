import pytest

from src.schemas import PredictionResult


def test_high_risk_predicts_churn(predictor, high_risk_customer):
    result = predictor.predict(high_risk_customer)
    assert isinstance(result, PredictionResult)
    assert result.churn_probability > 0.5
    assert result.churn_prediction is True


def test_low_risk_predicts_stay(predictor, low_risk_customer):
    result = predictor.predict(low_risk_customer)
    assert isinstance(result, PredictionResult)
    assert result.churn_probability < 0.5
    assert result.churn_prediction is False


def test_prediction_has_top_factors(predictor, high_risk_customer):
    result = predictor.predict(high_risk_customer)
    assert len(result.top_factors) == 5
    for factor in result.top_factors:
        assert "feature" in factor
        assert "impact" in factor
        assert "value" in factor


def test_prediction_has_tree_path(predictor, high_risk_customer):
    result = predictor.predict(high_risk_customer)
    assert len(result.decision_tree_path) > 0
    assert "confidence" in result.decision_tree_path


def test_probability_in_range(predictor, high_risk_customer):
    result = predictor.predict(high_risk_customer)
    assert 0.0 <= result.churn_probability <= 1.0
