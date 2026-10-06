"""Agent/environment transcript item."""

from pydantic import BaseModel


class SessionItem(BaseModel):
    role: str  # system / user / assistant
    content: str
