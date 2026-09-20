from __future__ import annotations

from collections import OrderedDict
from datetime import datetime

from src.config import MAX_SESSIONS, SESSION_TTL_MINUTES
from src.schemas import PartialCustomerFeatures


class ChatSession:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.language: str = "en"
        self.partial_features = PartialCustomerFeatures()
        self.history: list[dict[str, str]] = []
        self.created_at = datetime.now()

    def add_message(self, role: str, content: str) -> None:
        self.history.append({"role": role, "content": content})

    def get_conversation_text(self) -> str:
        return "\n".join(f"{m['role']}: {m['content']}" for m in self.history)


class SessionStore:
    def __init__(self, max_sessions: int = MAX_SESSIONS):
        self._sessions: OrderedDict[str, ChatSession] = OrderedDict()
        self._max = max_sessions

    def get_or_create(self, session_id: str) -> ChatSession:
        if session_id in self._sessions:
            self._sessions.move_to_end(session_id)
            return self._sessions[session_id]

        if len(self._sessions) >= self._max:
            self._sessions.popitem(last=False)

        session = ChatSession(session_id)
        self._sessions[session_id] = session
        return session
