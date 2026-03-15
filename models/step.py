from dataclasses import dataclass

@dataclass
class EquationStep:
    """
    Data model representing a single step in the equation solving process.

    before : str
        The equation before the transformation step.

    after : str
        The equation after the transformation step.

    explanation : str
        Optional human-readable explanation of the transformation.
        Example:
            "Subtract 3 from both sides"
            "Apply the quadratic formula"

    Notes:
        This class is used by the solver to communicate steps to the renderer.
        The renderer should only care about displaying these transitions,
        not about how the math was computed.
    """
    before: str
    after: str
    explanation: str | None = None               # There may not be an explanation.
    