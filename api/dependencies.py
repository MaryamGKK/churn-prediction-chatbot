from src.analytics import DatasetAnalytics
from src.chatbot.chain import ChurnChatbot
from src.chatbot.session import SessionStore
from src.predict import ChurnPredictor

predictor: ChurnPredictor | None = None
chatbot: ChurnChatbot | None = None
session_store: SessionStore | None = None


def init_services() -> None:
    global predictor, chatbot, session_store
    predictor = ChurnPredictor()
    analytics = DatasetAnalytics()
    chatbot = ChurnChatbot(predictor, analytics=analytics)
    session_store = SessionStore()


def get_predictor() -> ChurnPredictor:
    assert predictor is not None
    return predictor


def get_chatbot() -> ChurnChatbot:
    assert chatbot is not None
    return chatbot


def get_session_store() -> SessionStore:
    assert session_store is not None
    return session_store
