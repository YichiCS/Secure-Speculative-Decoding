from dataclasses import dataclass


@dataclass
class SamplingParams:
    temperature: float = 0.0
    draft_temperature: float = 0.0
    max_new_tokens: int = 256
    ignore_eos: bool = False

    def __post_init__(self):
        if self.temperature < 0:
            raise ValueError(f"temperature must be >= 0, got {self.temperature}")
        if self.draft_temperature < 0:
            raise ValueError(
                f"draft_temperature must be >= 0, got {self.draft_temperature}"
            )
