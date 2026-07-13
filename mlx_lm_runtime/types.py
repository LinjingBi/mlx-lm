"""Transport-neutral request and response types for the local runtime."""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class GenerateRequest:
    """A generation request for the daemon's single resident model."""

    prompt: Optional[str] = None
    messages: Optional[List[Dict[str, Any]]] = None
    max_tokens: int = 256
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0
    min_p: float = 0.0
    stop: Tuple[str, ...] = field(default_factory=tuple)
    seed: Optional[int] = None
    chat_template_kwargs: Dict[str, Any] = field(default_factory=dict)

    def validate(self):
        def is_number(value):
            return isinstance(value, (int, float)) and not isinstance(value, bool)

        if (self.prompt is None) == (self.messages is None):
            raise ValueError("Exactly one of 'prompt' or 'messages' must be provided.")
        if self.prompt is not None and not isinstance(self.prompt, str):
            raise ValueError("'prompt' must be a string.")
        if self.messages is not None:
            if not isinstance(self.messages, list) or not self.messages:
                raise ValueError("'messages' must be a non-empty list.")
            for message in self.messages:
                if not isinstance(message, dict):
                    raise ValueError("Each message must be an object.")
                if not isinstance(message.get("role"), str):
                    raise ValueError("Each message must have a string 'role'.")
                if "content" not in message:
                    raise ValueError("Each message must have a 'content' field.")
        if not isinstance(self.max_tokens, int) or isinstance(self.max_tokens, bool):
            raise ValueError("'max_tokens' must be an integer.")
        if self.max_tokens <= 0:
            raise ValueError("'max_tokens' must be greater than zero.")
        if not is_number(self.temperature) or self.temperature < 0:
            raise ValueError("'temperature' cannot be negative.")
        if not is_number(self.top_p) or not 0 <= self.top_p <= 1:
            raise ValueError("'top_p' must be between zero and one.")
        if (
            not isinstance(self.top_k, int)
            or isinstance(self.top_k, bool)
            or self.top_k < 0
        ):
            raise ValueError("'top_k' must be a non-negative integer.")
        if not is_number(self.min_p) or not 0 <= self.min_p <= 1:
            raise ValueError("'min_p' must be between zero and one.")
        if not all(isinstance(item, str) and item for item in self.stop):
            raise ValueError("Every stop sequence must be a non-empty string.")
        if self.seed is not None and (
            not isinstance(self.seed, int) or isinstance(self.seed, bool)
        ):
            raise ValueError("'seed' must be an integer or null.")
        if not isinstance(self.chat_template_kwargs, dict):
            raise ValueError("'chat_template_kwargs' must be an object.")

    def to_dict(self) -> Dict[str, Any]:
        self.validate()
        result = asdict(self)
        result["stop"] = list(self.stop)
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        if not isinstance(data, dict):
            raise ValueError("Generation request data must be an object.")
        allowed = {
            "prompt",
            "messages",
            "max_tokens",
            "temperature",
            "top_p",
            "top_k",
            "min_p",
            "stop",
            "seed",
            "chat_template_kwargs",
        }
        unknown = set(data) - allowed
        if unknown:
            raise ValueError(
                "Unknown generation request fields: " + ", ".join(sorted(unknown))
            )
        values = dict(data)
        stop = values.get("stop", ())
        if isinstance(stop, str):
            stop = (stop,)
        elif isinstance(stop, list):
            stop = tuple(stop)
        elif not isinstance(stop, tuple):
            raise ValueError("'stop' must be a string or a list of strings.")
        values["stop"] = stop
        request = cls(**values)
        request.validate()
        return request


@dataclass
class GenerationStarted:
    model: str
    prompt_tokens: int
    cached_tokens: int


@dataclass
class GenerationDelta:
    text: str
    token: int
    finish_reason: Optional[str] = None


@dataclass
class GenerationFinished:
    finish_reason: str
    prompt_tokens: int
    cached_tokens: int
    generation_tokens: int
    prompt_tps: float
    generation_tps: float
    peak_memory_gb: float
    ttft_seconds: float
    total_seconds: float


@dataclass
class GenerationResult:
    text: str
    started: GenerationStarted
    finished: GenerationFinished
