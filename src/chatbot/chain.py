from __future__ import annotations

import json
import logging
import re

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama

from src.chatbot.prompts import (
    AGENT_SYSTEM_PROMPT_EN,
    FIELD_LABELS,
    FIELD_LABELS_AR,
)
from src.chatbot.session import ChatSession
from src.analytics import DatasetAnalytics
from src.config import OLLAMA_BASE_URL, OLLAMA_FALLBACK_MODEL, OLLAMA_MODEL
from src.predict import ChurnPredictor
from src.schemas import PredictionResult

logger = logging.getLogger(__name__)

_FIELD_CONVERTERS = {
    "senior_citizen": lambda v: int(v) if isinstance(v, (str, float)) else v,
    "tenure": lambda v: int(v),
    "monthly_charges": lambda v: float(v),
    "total_charges": lambda v: float(v),
}


def _detect_language(text: str) -> str:
    if re.search(r"[؀-ۿݐ-ݿࢠ-ࣿ]", text):
        return "ar"
    return "en"


def _format_factors(factors: list[dict]) -> str:
    lines = []
    for f in factors:
        direction = "increases" if f["impact"] > 0 else "decreases"
        lines.append(f"- {f['feature']}: {direction} churn risk (impact: {f['impact']:.3f})")
    return "\n".join(lines)


def _labels(lang: str) -> dict[str, str]:
    return FIELD_LABELS_AR if lang == "ar" else FIELD_LABELS


def _collection_summary(partial: PartialCustomerFeatures, lang: str) -> str:
    labels = _labels(lang)
    collected = {k: v for k, v in partial.model_dump().items() if v is not None}
    missing = partial.missing_fields()
    total = len(collected) + len(missing)

    parts = [f"Collected ({len(collected)}/{total}):"]
    parts.extend(f"  {labels.get(k, k)}: {v}" for k, v in collected.items())
    parts.append(f"\nMissing ({len(missing)}):")
    parts.extend(f"  {labels.get(f, f)}" for f in missing)

    if not missing:
        parts.append("\nAll fields collected! Call predict_churn to make the prediction.")
    return "\n".join(parts)


_VALID_TOOLS = {"update_customer_features", "get_collection_status", "predict_churn", "analyze_data"}


def _parse_tool_call(text: str) -> tuple[str, dict] | None:
    text = text.strip()

    # 1. Try code-fenced JSON
    json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if json_match:
        candidate = json_match.group(1)
        try:
            data = json.loads(candidate)
            if isinstance(data, dict) and "tool" in data:
                return data["tool"], data.get("args", {})
        except (json.JSONDecodeError, TypeError):
            pass

    # 2. Try the whole response as JSON
    try:
        data = json.loads(text)
        if isinstance(data, dict) and "tool" in data:
            return data["tool"], data.get("args", {})
    except (json.JSONDecodeError, TypeError):
        pass

    # 3. Find {"tool": "name" ...} embedded anywhere
    match = re.search(r'\{[^{}]*"tool"\s*:\s*"(\w+)"[^{}]*\}', text)
    if match:
        try:
            data = json.loads(match.group())
            if isinstance(data, dict) and "tool" in data:
                return data["tool"], data.get("args", {})
        except (json.JSONDecodeError, TypeError):
            pass

    # 4. Catch "Tool: tool_name" pattern (model describes instead of JSON)
    tool_match = re.search(r"(?:Tool|tool|TOOL)\s*:\s*(\w+)", text)
    if tool_match:
        tool_name = tool_match.group(1)
        if tool_name in _VALID_TOOLS:
            args = {}
            args_match = re.search(r"(?:Arguments?|args?|ARGS?)\s*:\s*(\{[^}]*\})", text)
            if args_match:
                try:
                    args = json.loads(args_match.group(1))
                except (json.JSONDecodeError, TypeError):
                    pass
            return tool_name, args

    return None


class _ToolContext:
    def __init__(self, session: ChatSession, predictor: ChurnPredictor, analytics: DatasetAnalytics):
        self.session = session
        self.predictor = predictor
        self.analytics = analytics
        self.prediction_result: PredictionResult | None = None


def _build_tools(ctx: _ToolContext) -> list:

    @tool
    def update_customer_features(features: dict) -> str:
        """Save customer information extracted from the conversation.

        Args:
            features: dict of field_name:value pairs. Valid fields:
                senior_citizen (0/1), is_married ("Yes"/"No"),
                dependents ("Yes"/"No"), tenure (int months 0-72),
                phone_service ("Yes"/"No"),
                internet_service ("DSL"/"Fiber optic"/"No"),
                online_security ("Yes"/"No"), online_backup ("Yes"/"No"),
                device_protection ("Yes"/"No"), tech_support ("Yes"/"No"),
                streaming_tv ("Yes"/"No"), streaming_movies ("Yes"/"No"),
                contract ("Month-to-month"/"One year"/"Two year"),
                paperless_billing ("Yes"/"No"),
                payment_method ("Electronic check"/"Mailed check"/
                    "Bank transfer (automatic)"/"Credit card (automatic)"),
                monthly_charges (float), total_charges (float).
        """
        partial = ctx.session.partial_features
        updated = []

        for key, value in features.items():
            if value is None or not hasattr(partial, key):
                continue
            converter = _FIELD_CONVERTERS.get(key)
            if converter:
                try:
                    value = converter(value)
                except (ValueError, TypeError):
                    continue
            try:
                setattr(partial, key, value)
                updated.append(key)
            except Exception:
                continue

        partial.apply_logical_defaults()
        result = f"Updated: {', '.join(updated)}.\n" if updated else "No fields updated.\n"
        return result + _collection_summary(partial, ctx.session.language)

    @tool
    def get_collection_status() -> str:
        """Check which customer fields have been collected and which are still missing."""
        return _collection_summary(ctx.session.partial_features, ctx.session.language)

    @tool
    def predict_churn() -> str:
        """Run the churn prediction model. Call this ONLY when all customer fields are collected."""
        partial = ctx.session.partial_features
        if not partial.is_complete():
            missing = partial.missing_fields()
            labels = _labels(ctx.session.language)
            return f"Cannot predict yet. Missing: {', '.join(labels.get(f, f) for f in missing)}"

        customer = partial.to_customer_features()
        result = ctx.predictor.predict(customer)
        ctx.prediction_result = result

        verdict = "LIKELY TO CHURN" if result.churn_prediction else "LIKELY TO STAY"
        return (
            f"Prediction: {verdict}\n"
            f"Churn Probability: {result.churn_probability:.1%}\n\n"
            f"Top Factors:\n{_format_factors(result.top_factors)}\n\n"
            f"Decision Tree Path: {result.decision_tree_path}"
        )

    @tool
    def analyze_data(topic: str) -> str:
        """Query the customer churn dataset for analytics and insights.

        Use this to answer analytical questions about churn patterns, trends,
        customer segments, and feature breakdowns from the historical data.

        Args:
            topic: What to analyze. Examples: "overview", "churn by contract",
                "high risk segments", "internet service", "payment method",
                "senior citizens", "tech support", "streaming".
        """
        return ctx.analytics.query(topic)

    return [update_customer_features, get_collection_status, predict_churn, analyze_data]


class ChurnChatbot:
    def __init__(
        self,
        predictor: ChurnPredictor,
        analytics: DatasetAnalytics | None = None,
        model_name: str = OLLAMA_MODEL,
        fallback_model: str = OLLAMA_FALLBACK_MODEL,
    ):
        self.predictor = predictor
        self.analytics = analytics or DatasetAnalytics()
        self.model_name = model_name
        self.fallback_model = fallback_model

    async def handle_message(
        self, session: ChatSession, message: str,
    ) -> tuple[str, PredictionResult | None]:
        session.add_message("user", message)

        if len(session.history) == 1:
            session.language = _detect_language(message)

        # Primary: local open-source model with manual JSON tool calling
        try:
            return await self._run_manual_agent(session, self.model_name)
        except Exception as e:
            logger.warning("Primary model (%s) failed: %s", self.model_name, e)

        # Fallback: smaller local model
        if self.fallback_model and self.fallback_model != self.model_name:
            try:
                return await self._run_manual_agent(session, self.fallback_model)
            except Exception as e:
                logger.warning("Fallback model (%s) failed: %s", self.fallback_model, e)

        logger.error("All models failed, returning fallback response")
        fallback = self._fallback_response(session)
        session.add_message("assistant", fallback)
        return fallback, None

    async def _run_native_agent(
        self, session: ChatSession, llm: object,
    ) -> tuple[str, PredictionResult | None]:
        """Native tool calling via Ollama (bind_tools)."""
        ctx = _ToolContext(session, self.predictor, self.analytics)
        tools = _build_tools(ctx)
        tool_map = {t.name: t for t in tools}

        llm_with_tools = llm.bind_tools(tools)
        messages = self._build_messages(session)

        max_steps = 6
        response = None
        for _ in range(max_steps):
            response = await llm_with_tools.ainvoke(messages)
            messages.append(response)

            if not response.tool_calls:
                break

            for tc in response.tool_calls:
                tool_fn = tool_map.get(tc["name"])
                if tool_fn is None:
                    result_str = f"Unknown tool: {tc['name']}"
                else:
                    result_str = str(tool_fn.invoke(tc["args"]))
                messages.append(ToolMessage(
                    content=result_str, tool_call_id=tc.get("id", ""),
                ))

        text = (response.content if response else "") or ""
        if not text:
            text = self._fallback_response(session)
        session.add_message("assistant", text)
        return text, ctx.prediction_result

    async def _run_manual_agent(
        self, session: ChatSession, model_name: str,
    ) -> tuple[str, PredictionResult | None]:
        """Manual JSON-based tool calling for reliability fallback."""
        ctx = _ToolContext(session, self.predictor, self.analytics)
        tools = _build_tools(ctx)
        tool_map = {t.name: t for t in tools}

        llm = ChatOllama(
            model=model_name, base_url=OLLAMA_BASE_URL,
            temperature=0.3, num_predict=512, num_ctx=4096,
        )

        messages = self._build_messages(session)

        max_steps = 6
        final_text = ""
        for _ in range(max_steps):
            response = await llm.ainvoke(messages)
            text = response.content or ""

            tool_call = _parse_tool_call(text)
            if tool_call is None:
                logger.info("Model replied with text (no tool call): %s", text[:200])
                final_text = text
                break

            tool_name, args = tool_call
            logger.info("Tool call: %s args=%s", tool_name, str(args)[:200])
            tool_fn = tool_map.get(tool_name)
            if tool_fn is None:
                result_str = f"Unknown tool: {tool_name}"
            elif tool_name == "update_customer_features":
                features = args.get("features", args)
                result_str = str(tool_fn.invoke({"features": features}))
            elif tool_name == "analyze_data":
                topic = args.get("topic", "overview")
                result_str = str(tool_fn.invoke({"topic": topic}))
            else:
                result_str = str(tool_fn.invoke({}))

            messages.append(AIMessage(content=text))
            messages.append(HumanMessage(
                content=f"[Tool result for {tool_name}]\n{result_str}",
            ))

            # Auto-predict when all fields are collected
            if "All fields collected" in result_str and tool_name != "predict_churn":
                predict_fn = tool_map.get("predict_churn")
                if predict_fn:
                    pred_result = str(predict_fn.invoke({}))
                    messages.append(AIMessage(content='{"tool": "predict_churn", "args": {}}'))
                    messages.append(HumanMessage(
                        content=f"[Tool result for predict_churn]\n{pred_result}",
                    ))

        if not final_text:
            final_text = self._fallback_response(session)
        session.add_message("assistant", final_text)
        return final_text, ctx.prediction_result

    def _build_messages(self, session: ChatSession) -> list:
        messages: list = [SystemMessage(content=AGENT_SYSTEM_PROMPT_EN)]
        for msg in session.history:
            cls = HumanMessage if msg["role"] == "user" else AIMessage
            messages.append(cls(content=msg["content"]))
        return messages

    def _fallback_response(self, session: ChatSession) -> str:
        partial = session.partial_features
        missing = partial.missing_fields()
        labels = _labels(session.language)

        if session.language == "ar":
            if not missing:
                return "تم جمع جميع المعلومات. يرجى المحاولة مرة اخرى للحصول على التنبؤ."
            fields = ", ".join(labels.get(f, f) for f in missing[:3])
            return f"احتاج الى مزيد من المعلومات. يرجى تقديم: {fields}"

        if not missing:
            return "All information collected. Please try again for the prediction."
        fields = ", ".join(labels.get(f, f) for f in missing[:3])
        return f"I need more information. Please provide: {fields}"
