from dataclasses import dataclass

@dataclass
class Step:
    before: str
    after: str
    explanation: str = ""