# Math Animator

A Python tool that takes a linear equation and generates an animated video showing every single step of the solution, with plain-language explanations alongside the math.

## Features

- **Solve mode:** resolves a linear equation step by step, showing every transformation until `x = value`
- **Guess mode:** the user inputs a value for x and the tool shows the full substitution and verification process
- **Natural language input:** type equations in plain English and the AI converts them to math notation
- **Decimal and fraction support:** handles decimals, unreduced fractions, rational coefficients, and division by zero
- **MP4 output:** every animation is rendered as a video file using Manim

## AI Component

Math Animator uses the [Groq API](https://console.groq.com) running **Llama 3.3 70B** to parse natural language input into standard mathematical notation.

Example:
```
"half x plus eight equals three x minus four"  >>  "1/2x + 8 = 3x - 4"
```

## Project Structure

```
math_animator/
├── main.py                        # Entry point
├── controller/
│   └── animation_controller.py   # Routes equation to solver or guess checker
├── animation/
│   └── equation_renderer.py      # Manim animation logic
├── solvers/
│   ├── linear_solver.py          # Main solving pipeline
│   ├── rational_coef_steps.py    # Steps for fractional coefficients
│   └── integer_coef_steps.py     # Steps for integer coefficients
├── parser/
│   └── equation_parser.py        # Parses and normalises equation strings
├── utils/
│   ├── ai_parser.py              # Groq API, natural language to math
│   ├── guess_checker.py          # Substitution and verification logic
│   ├── latex_helpers.py          # LaTeX formatting utilities
│   ├── term_combiner.py          # Stepwise term combination
│   ├── term_rearranger.py        # Moves terms to correct sides
│   ├── simplification_steps.py   # Decimal and fraction simplification
│   └── impossible_solution.py    # Division by zero detection
├── models/
│   └── step.py                   # Step model (before, after, explanation)
└── math_utils/
    └── gcd_utils.py              # GCD helpers
```

## Requirements

- Python 3.11+
- [Manim Community](https://www.manim.community/) v0.20+
- [Groq Python SDK](https://github.com/groq/groq-python)
- SymPy

Install dependencies:
```bash
pip install manim groq sympy
```

## Setup

Set your Groq API key as an environment variable:
```bash
export GROQ_API_KEY="your-key-here"
```

Get a free API key at [console.groq.com/keys](https://console.groq.com/keys).

## Usage

```bash
python -m manim main.py SolveScene -pqh
```

You will be prompted for:
1. **Input mode:** `math` or `natural`
2. **Equation:** e.g. `1/2x + 8 = 3x - 4`
3. **Mode:** `solve` or `guess`
4. **x value:** (guess mode only) e.g. `4`

The animation is saved as an MP4 in `media/videos/`.

## Examples

**Solve mode:**
```
Equation: 1/2x + 8 = 3x - 4
Mode: solve
```

**Guess mode:**
```
Equation: 2x + 3 = 7
Mode: guess
x = 2
```

**Natural language:**
```
Input mode: natural
Describe the equation: half x plus eight equals three x minus four
```

## Design Thinking

Built for two users:
- A **student** who gets lost with fractions and needs step-by-step visual explanations
- A **teacher** who needs ready-to-play animations for class

Core question: *How might we help students understand why each step works, not just what the answer is?*

## Author

Francisco Ponte Pimentel, Solo Project

