from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class ToolCall(BaseModel):
    name: str
    arguments: Dict[str, Any]

class Message(BaseModel):
    role: str = Field(description="system, user, assistant, or tool")
    content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None

class Trajectory(BaseModel):
    task_id: str
    seed: int
    split: str
    messages: List[Message]
    final_answer: List[str]
    is_correct: bool