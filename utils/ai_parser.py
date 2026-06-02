from groq import Groq
import os

client = Groq()

'''
This function takes a natural language description of a mathematical equation
and returns it in standard format using the Groq API.

Example: "two x plus three equals seven"  >>  "2x + 3 = 7"
         "half x minus four is ten"        >>  "1/2x - 4 = 10"
'''
def parse_natural_language(user_input: str) -> str:
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a math equation parser. "
                    "Convert the user's natural language description of a linear equation "
                    "into standard mathematical notation. "
                    "Return ONLY the equation string, nothing else. "
                    "Use 'x' as the variable. "
                    "Examples: "
                    "'two x plus three equals seven' -> '2x + 3 = 7', "
                    "'half x minus four is ten' -> '1/2x - 4 = 10'"
                )
            },
            {
                "role": "user",
                "content": user_input
            }
        ],
        max_tokens=50,
        temperature=0
    )
    return response.choices[0].message.content.strip()

