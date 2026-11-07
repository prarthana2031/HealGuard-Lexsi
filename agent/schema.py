from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional

# One trajectory contract. This file previously disagreed with
# agent/trajectory_schema.md (task_id vs id, is_correct vs success, no tool
# counts), which is why every consumer needed its own adapter. The markdown file
# is the source of truth; this model mirrors it and a test keeps them aligned.


class ToolCall(BaseModel):
    name: str
    arguments: Dict[str, Any]
    valid: Optional[bool] = None


class Message(BaseModel):
    role: str = Field(description="system, user, assistant, or tool")
    content: Optional[str] = None


class Trajectory(BaseModel):
    id: str
    seed: int
    split: str
    task: str
    messages: List[Message]
    tool_calls: List[ToolCall] = Field(default_factory=list)
    required_tools: List[str] = Field(default_factory=list)
    final_answer: List[str] = Field(default_factory=list)
    reward: float = 0.0
    success: bool = False
    num_steps: int = 0
    template_id: str = "custom_tool_call_v1"
    error: Optional[str] = None
    valid_tool_calls: int = 0
    total_tool_calls: int = 0
    redundant_calls: int = 0
    finished: bool = False
    drift: Optional[str] = None
    family_success: Optional[Dict[str, bool]] = None
