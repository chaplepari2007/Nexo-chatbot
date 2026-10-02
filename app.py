from flask import Flask, render_template, request, jsonify
import requests
import base64
import time
import re
import urllib.parse
import subprocess
import tempfile
import os

app = Flask(__name__)

# =========================
# OLLAMA SETTINGS
# =========================
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"

TEXT_MODEL = "qwen2.5:0.5b"
VISION_MODEL = "qwen2.5vl:3b"


# =========================
# ASK OLLAMA
# =========================
def ask_ollama(prompt, model=TEXT_MODEL, image=None):
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }

    if image:
        payload["images"] = [image]

    try:
        start = time.time()

        response = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=180
        )

        elapsed = round(time.time() - start, 2)

        response.raise_for_status()

        return {
            "response": response.json().get("response", ""),
            "time": elapsed,
            "error": None
        }

    except requests.exceptions.ConnectionError:
        return {
            "response": "",
            "time": 0,
            "error": "Ollama is not running. Start Ollama first."
        }

    except requests.exceptions.Timeout:
        return {
            "response": "",
            "time": 0,
            "error": "Ollama took too long to respond."
        }

    except Exception as e:
        return {
            "response": "",
            "time": 0,
            "error": str(e)
        }


# =========================
# QUESTION ANALYSIS
# =========================
def analyze_question(question):
    q = question.lower()

    programming = any(x in q for x in [
        "code",
        "program",
        "python",
        "java",
        "c language",
        "c++",
        "javascript",
        "implement",
        "write a program",
        "algorithm",
        "debug",
        "sort",
        "search",
        "stack",
        "queue",
        "linked list",
        "tree",
        "graph",
        "recursion"
    ])

    numerical = any(x in q for x in [
        "solve",
        "calculate",
        "find",
        "equation",
        "numerical",
        "compute",
        "value of",
        "determine"
    ])

    diagram = any(x in q for x in [
        "diagram",
        "architecture",
        "circuit",
        "flowchart",
        "motor",
        "engine",
        "beam",
        "truss",
        "transformer",
        "osi",
        "tcp",
        "network",
        "process",
        "working",
        "structure",
        "block diagram"
    ])

    complexity = programming or any(x in q for x in [
        "complexity",
        "big o",
        "time complexity",
        "space complexity"
    ])

    topics = [
        "binary search",
        "linear search",
        "bubble sort",
        "selection sort",
        "insertion sort",
        "merge sort",
        "quick sort",
        "stack",
        "queue",
        "linked list",
        "osi model",
        "tcp handshake",
        "ohm law",
        "projectile",
        "pendulum",
        "four stroke engine"
    ]

    interactive = next(
        (x for x in topics if x in q),
        None
    )

    return {
        "theory": True,
        "steps": True,
        "example": True,
        "diagram": diagram,
        "video": True,
        "interactive": interactive is not None,
        "interactiveTopic": interactive,
        "code": programming,
        "run": programming,
        "output": programming,
        "complexity": complexity,
        "applications": True,
        "summary": True,
        "numerical": numerical
    }


# =========================
# BUILD AI PROMPT
# =========================
def build_prompt(question):
    return f"""
You are NEXO, a general educational assistant for college students.

You can answer questions from:

CSE
Computer Science
IT
Civil Engineering
Mechanical Engineering
Electrical Engineering
Electronics
Mathematics
Physics
Chemistry
Biology
Architecture
Science
and other academic subjects.

Question:
{question}

Create a clear, accurate and beginner-friendly educational solution.

IMPORTANT RULES:

1. Do not invent facts.
2. Explain the concept clearly.
3. Use simple language.
4. Give step-by-step explanation where useful.
5. For numerical questions show:
   Given
   Formula
   Substitution
   Calculation
   Final Answer
6. For programming questions provide correct code.
7. For programming questions provide expected output.
8. For programming/DSA questions provide time and space complexity.
9. For engineering/science questions explain components and working.
10. If a diagram is useful, provide Mermaid diagram syntax.
11. If a diagram is not useful, write NONE.
12. Give a useful YouTube educational search phrase.
13. Do not force code for non-programming questions.
14. Do not force complexity for topics where it is irrelevant.
15. Give practical applications whenever appropriate.
16. Keep the headings EXACTLY as shown below.

Use this exact format:

[TITLE]
topic title

[THEORY]
clear theory explanation

[STEPS]
1. step
2. step
3. step

[ALGORITHM]
1. algorithm step
2. algorithm step

[EXAMPLE]
clear example

[DIAGRAM]
```mermaid
flowchart TD
A[Start] --> B[Process]
B --> C[End]
