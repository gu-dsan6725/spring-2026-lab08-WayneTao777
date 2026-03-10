"""In-memory conversation session manager with circular buffer."""

import logging
from collections import deque
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class Message:
    """Represents a single conversation message."""

    role: str
    content: str
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class Session:
    """Represents a conversation session with circular message buffer."""

    session_id: str
    messages: deque = field(default_factory=deque)
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_accessed: datetime = field(default_factory=datetime.utcnow)
    max_size: int = 100

    def add_message(
        self,
        role: str,
        content: str
    ) -> None:
        """Add a message to the session history.

        Uses circular buffer - oldest messages are automatically removed
        when max_size is reached.
        """
        message = Message(role=role, content=content)

        if len(self.messages) >= self.max_size:
            self.messages.popleft()

        self.messages.append(message)
        self.last_accessed = datetime.utcnow()

        logger.debug(
            f"Added {role} message to session {self.session_id}. "
            f"History size: {len(self.messages)}/{self.max_size}"
        )

    def get_history(self) -> List[Dict[str, str]]:
        """Get conversation history as list of dicts for LLM."""
        self.last_accessed = datetime.utcnow()
        return [
            {"role": msg.role, "content": msg.content}
            for msg in self.messages
        ]

    def clear(self) -> None:
        """Clear all messages from session."""
        self.messages.clear()
        logger.info(f"Cleared session {self.session_id}")


class SessionManager:
    """Manages multiple conversation sessions in memory."""

    def __init__(
        self,
        max_history_size: int = 100
    ):
        self.sessions: Dict[str, Session] = {}
        self.max_history_size = max_history_size
        logger.info(f"SessionManager initialized with max_history_size={max_history_size}")

    def get_or_create_session(
        self,
        session_id: str
    ) -> Session:
        """Get existing session or create new one."""
        if session_id not in self.sessions:
            self.sessions[session_id] = Session(
                session_id=session_id,
                max_size=self.max_history_size
            )
            logger.info(f"Created new session: {session_id}")

        return self.sessions[session_id]

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str
    ) -> None:
        """Add a message to a session's history."""
        session = self.get_or_create_session(session_id)
        session.add_message(role, content)

    def get_history(
        self,
        session_id: str
    ) -> List[Dict[str, str]]:
        """Get conversation history for a session."""
        session = self.get_or_create_session(session_id)
        return session.get_history()

    def clear_session(
        self,
        session_id: str
    ) -> None:
        """Clear a specific session."""
        if session_id in self.sessions:
            self.sessions[session_id].clear()

    def delete_session(
        self,
        session_id: str
    ) -> bool:
        """Delete a session entirely."""
        if session_id in self.sessions:
            del self.sessions[session_id]
            logger.info(f"Deleted session: {session_id}")
            return True
        return False

    def get_session_count(self) -> int:
        """Get total number of active sessions."""
        return len(self.sessions)

    def get_session_info(
        self,
        session_id: str
    ) -> Optional[Dict]:
        """Get metadata about a session."""
        if session_id not in self.sessions:
            return None

        session = self.sessions[session_id]
        return {
            "session_id": session.session_id,
            "message_count": len(session.messages),
            "max_size": session.max_size,
            "created_at": session.created_at.isoformat(),
            "last_accessed": session.last_accessed.isoformat()
        }
