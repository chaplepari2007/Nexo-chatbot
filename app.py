````python
from flask import Flask, render_template, request, jsonify
import requests
import base64
import time
import re
import urllib.parse
import subprocess
import tempfile
import os
import random

app = Flask(__name__)

# ============================================================
# GEMINI CONFIGURATION
# ============================================================

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

GEMINI_MODELS = [
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
]

GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta/interactions"
)

MAX_RETRIES_PER_MODEL = 2


# ============================================================
# GEMINI RESPONSE PARSER
# ============================================================

def extract_gemini_text(data):
    """Extract text from Gemini Interactions API response."""

    output_text = data.get("output_text")

    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    steps = data.get("steps", [])

    if not isinstance(steps, list):
        return ""

    pieces = []

    for step in steps:
        if not isinstance(step, dict):
            continue

        if step.get("type") != "model_output":
            continue

        content = step.get("content", [])

        if not isinstance(content, list):
            continue

        for block in content:
            if not isinstance(block, dict):
                continue

            if block.get("type") == "text":
                text = block.get("text", "")

                if isinstance(text, str) and text.strip():
                    pieces.append(text.strip())

    return "\n".join(pieces).strip()


# ============================================================
# ASK GEMINI
# ============================================================

def ask_gemini(prompt, image_data=None, mime_type=None):

    if not GEMINI_API_KEY:
        return {
            "response": "",
            "time": 0,
            "error": "GEMINI_API_KEY is not configured on the server."
        }

    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": GEMINI_API_KEY
    }

    # Text-only request
    if image_data:
        input_data = [
            {
                "type": "image",
                "data": image_data,
                "mime_type": mime_type or "image/jpeg"
            },
            {
                "type": "text",
                "text": prompt
            }
        ]
    else:
        input_data = prompt

    start_total = time.time()
    errors = []

    for model in GEMINI_MODELS:

        for attempt in range(MAX_RETRIES_PER_MODEL):

            try:

                payload = {
                    "model": model,
                    "input": input_data
                }

                response = requests.post(
                    GEMINI_URL,
                    headers=headers,
                    json=payload,
                    timeout=90
                )

                status = response.status_code

                # ------------------------------------------------
                # SUCCESS
                # ------------------------------------------------

                if response.ok:

                    try:
                        data = response.json()
                    except Exception:
                        return {
                            "response": "",
                            "time": round(
                                time.time() - start_total,
                                2
                            ),
                            "error":
                                "Gemini returned an invalid response."
                        }

                    text = extract_gemini_text(data)

                    if text:

                        return {
                            "response": text,
                            "time": round(
                                time.time() - start_total,
                                2
                            ),
                            "error": None,
                            "model": model
                        }

                    errors.append(
                        f"{model}: empty response"
                    )

                    break

                # ------------------------------------------------
                # TEMPORARY ERROR
                # ------------------------------------------------

                if (
                    status == 408
                    or status == 429
                    or status >= 500
                ):

                    try:
                        error_json = response.json()

                        error_message = (
                            error_json
                            .get("error", {})
                            .get("message", "")
                        )

                    except Exception:
                        error_message = response.text[:500]

                    errors.append(
                        f"{model} ({status}): "
                        f"{error_message}"
                    )

                    delay = (
                        (2 ** attempt)
                        + random.uniform(0.2, 0.8)
                    )

                    time.sleep(delay)

                    continue

                # ------------------------------------------------
                # PERMANENT ERROR
                # ------------------------------------------------

                try:
                    error_json = response.json()

                    error_message = (
                        error_json
                        .get("error", {})
                        .get("message", "")
                    )

                except Exception:
                    error_message = response.text[:1000]

                return {
                    "response": "",
                    "time": round(
                        time.time() - start_total,
                        2
                    ),
                    "error": (
                        f"Gemini API error ({status}): "
                        f"{error_message}"
                    )
                }

            except requests.exceptions.Timeout:

                errors.append(
                    f"{model}: request timed out"
                )

                delay = (
                    (2 ** attempt)
                    + random.uniform(0.2, 0.8)
                )

                time.sleep(delay)

            except requests.exceptions.ConnectionError:

                errors.append(
                    f"{model}: connection error"
                )

                delay = (
                    (2 ** attempt)
                    + random.uniform(0.2, 0.8)
                )

                time.sleep(delay)

            except Exception as e:

                errors.append(
                    f"{model}: {str(e)}"
                )

                break

    total_time = round(
        time.time() - start_total,
        2
    )

    return {
        "response": "",
        "time": total_time,
        "error": (
            "Gemini is temporarily unavailable after "
            "automatic retries and fallback models."
            "\n\n"
            + "\n".join(errors[-8:])
        )
    }


# ============================================================
# QUESTION ANALYSIS
# ============================================================

def analyze_question(question):

    q = question.lower()

    programming = any(
        x in q
        for x in [
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
        ]
    )

    numerical = any(
        x in q
        for x in [
            "solve",
            "calculate",
            "find",
            "equation",
            "numerical",
            "compute",
            "value of",
            "determine"
        ]
    )

    diagram = any(
        x in q
        for x in [
            "diagram",
            "architecture",
            "circuit",
            "flowchart",
            "motor",
            "engine",
            "beam",
            "truss",
            "osi",
            "tcp",
            "network",
            "process",
            "working",
            "structure",
            "block diagram"
        ]
    )

    complexity = (
        programming
        or any(
            x in q
            for x in [
                "complexity",
                "big o",
                "time complexity",
                "space complexity"
            ]
        )
    )

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
        (
            x
            for x in topics
            if x in q
        ),
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


# ============================================================
# EDUCATIONAL PROMPT
# ============================================================

def build_prompt(question):

    return f"""
You are NEXO, a general educational assistant for college students.

You can answer questions from:

Computer Science
CSE
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
16. Keep answers educational and well structured.
17. For mathematical problems, show actual calculation.
18. For engineering topics, explain the working principle.
19. For programming questions, make code runnable when practical.

Keep the headings EXACTLY as shown below.

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
Mermaid diagram if useful.
If no diagram is useful, write NONE.

[CODE]
Python/code if applicable.
If code is not applicable, write NONE.

[OUTPUT]
expected output or result.
If not applicable, write NONE.

[COMPLEXITY]
Time: ...
Space: ...
If not applicable, write NONE.

[APPLICATIONS]
- application
- application

[SUMMARY]
short revision summary

[VIDEO_QUERY]
short YouTube educational search phrase

[END]
"""


# ============================================================
# RESPONSE PARSING
# ============================================================

def clean_fence(value):

    value = value.strip()

    value = re.sub(
        r"^```[a-zA-Z0-9_+-]*\s*",
        "",
        value
    )

    value = re.sub(
        r"\s*```$",
        "",
        value
    )

    return value.strip()


def extract_section(text, name):

    pattern = (
        rf"\[{re.escape(name)}\]\s*"
        rf"(.*?)(?=\n\[[A-Z_]+\]|\Z)"
    )

    match = re.search(
        pattern,
        text,
        re.S | re.I
    )

    return (
        match.group(1).strip()
        if match
        else ""
    )


def parse_answer(raw, question):

    title = (
        extract_section(raw, "TITLE")
        or question[:80]
    )

    theory = extract_section(
        raw,
        "THEORY"
    )

    steps = extract_section(
        raw,
        "STEPS"
    )

    algorithm = extract_section(
        raw,
        "ALGORITHM"
    )

    example = extract_section(
        raw,
        "EXAMPLE"
    )

    diagram = extract_section(
        raw,
        "DIAGRAM"
    )

    code = extract_section(
        raw,
        "CODE"
    )

    output = extract_section(
        raw,
        "OUTPUT"
    )

    complexity = extract_section(
        raw,
        "COMPLEXITY"
    )

    applications = extract_section(
        raw,
        "APPLICATIONS"
    )

    summary = extract_section(
        raw,
        "SUMMARY"
    )

    video_query = extract_section(
        raw,
        "VIDEO_QUERY"
    )

    if diagram.upper() == "NONE":
        diagram = ""
    else:
        diagram = clean_fence(diagram)

    if code.upper() == "NONE":
        code = ""
    else:
        code = clean_fence(code)

    if output.upper() == "NONE":
        output = ""

    if complexity.upper() == "NONE":
        complexity = ""

    if not theory:
        theory = raw

    step_list = []

    for line in steps.splitlines():

        line = line.strip()

        if not line:
            continue

        line = re.sub(
            r"^\s*(?:[-*]|\d+[.)])\s*",
            "",
            line
        )

        if line:
            step_list.append(line)

    algorithm_list = []

    for line in algorithm.splitlines():

        line = line.strip()

        if not line:
            continue

        line = re.sub(
            r"^\s*(?:[-*]|\d+[.)])\s*",
            "",
            line
        )

        if line:
            algorithm_list.append(line)

    app_list = []

    for line in applications.splitlines():

        line = line.strip()

        if not line:
            continue

        line = re.sub(
            r"^\s*[-*]\s*",
            "",
            line
        )

        if line:
            app_list.append(line)

    analysis = analyze_question(
        question
    )

    analysis["code"] = (
        bool(code)
        or analysis["code"]
    )

    analysis["run"] = bool(code)

    analysis["output"] = bool(output)

    analysis["diagram"] = (
        bool(diagram)
        or analysis["diagram"]
    )

    analysis["complexity"] = (
        bool(complexity)
        or analysis["complexity"]
    )

    analysis["interactive"] = bool(
        analysis["interactiveTopic"]
    )

    return {
        "title": title,
        "theory": theory,
        "steps": step_list,
        "algorithm": algorithm_list,
        "example": example,
        "diagram": diagram,
        "code": code,
        "output": output,
        "complexity": complexity,
        "applications": app_list,
        "summary": summary,
        "videoQuery": (
            video_query
            or title
        ),
        "analysis": analysis
    }


# ============================================================
# HOME
# ============================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# ============================================================
# CHAT
# ============================================================

@app.route(
    "/chat",
    methods=["POST"]
)
def chat():

    data = (
        request.get_json(
            silent=True
        )
        or {}
    )

    question = (
        data.get(
            "message",
            ""
        )
        .strip()
    )

    if not question:

        return jsonify({
            "error":
                "Please enter a question."
        }), 400

    result = ask_gemini(
        build_prompt(question)
    )

    if result["error"]:

        return jsonify({
            "error":
                result["error"]
        }), 503

    answer = parse_answer(
        result["response"],
        question
    )

    query = urllib.parse.quote_plus(
        answer["videoQuery"]
    )

    answer["videoUrl"] = (
        "https://www.youtube.com/results"
        "?search_query="
        + query
    )

    answer["responseTime"] = (
        result["time"]
    )

    answer["model"] = result.get(
        "model",
        GEMINI_MODELS[0]
    )

    # IMPORTANT:
    # The existing frontend expects a top-level
    # "response" property.
    answer["response"] = answer["theory"]

    return jsonify(answer)


# ============================================================
# IMAGE / VISION
# ============================================================

@app.route(
    "/vision",
    methods=["POST"]
)
def vision():

    image_file = request.files.get(
        "image"
    )

    question = (
        request.form.get(
            "question",
            ""
        )
        .strip()
    )

    if not image_file:

        return jsonify({
            "error":
                "Please select an image."
        }), 400

    allowed = [
        "image/png",
        "image/jpeg",
        "image/jpg",
        "image/webp"
    ]

    if image_file.content_type not in allowed:

        return jsonify({
            "error":
                "Please use PNG, JPG or WEBP."
        }), 400

    image_data = image_file.read()

    if len(image_data) > 8 * 1024 * 1024:

        return jsonify({
            "error":
                "Image must be smaller than 8 MB."
        }), 400

    encoded = base64.b64encode(
        image_data
    ).decode("utf-8")

    prompt = f"""
You are NEXO, an educational assistant.

Analyze the uploaded educational image.

User question:
{
    question
    or
    "Explain and solve what is shown in the image."
}

Give a proper educational solution.

If it is mathematics:
- identify the problem
- solve step-by-step
- show formulas
- show calculation
- show final answer

If it is programming:
- identify the code
- explain it
- correct it if necessary
- provide output

If it is engineering/science:
- identify components
- explain working
- explain important concepts

If something is unreadable,
clearly say that instead of guessing.

Use clear educational language.
"""

    result = ask_gemini(
        prompt,
        image_data=encoded,
        mime_type=image_file.content_type
    )

    if result["error"]:

        return jsonify({
            "error":
                result["error"]
        }), 503

    return jsonify({
        "response":
            result["response"],

        "responseTime":
            result["time"],

        "model":
            result.get(
                "model",
                GEMINI_MODELS[0]
            ),

        "analysis":
            analyze_question(
                question
                or
                "image question"
            )
    })


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route("/health")
def health():

    return jsonify({
        "gemini":
            bool(GEMINI_API_KEY),

        "primaryModel":
            GEMINI_MODELS[0],

        "fallbackModels":
            GEMINI_MODELS[1:],

        "api":
            "Interactions API",

        "status":
            "ready"
            if GEMINI_API_KEY
            else "missing API key"
    })


# ============================================================
# SAFE PYTHON RUNNER
# ============================================================

@app.route(
    "/run-python",
    methods=["POST"]
)
def run_python():

    data = (
        request.get_json(
            silent=True
        )
        or {}
    )

    code = data.get(
        "code",
        ""
    )

    if not code.strip():

        return jsonify({
            "output":
                "No code provided."
        })

    blocked = [
        "import os",
        "from os",
        "import sys",
        "from sys",
        "subprocess",
        "socket",
        "shutil",
        "pathlib",
        "__import__",
        "eval(",
        "exec(",
        "open(",
        "requests",
        "urllib",
        "http.client"
    ]

    low = code.lower()

    if any(
        item in low
        for item in blocked
    ):

        return jsonify({
            "output":
                "Execution blocked: "
                "unsafe operation detected."
        })

    try:

        with tempfile.TemporaryDirectory() as d:

            path = os.path.join(
                d,
                "main.py"
            )

            with open(
                path,
                "w",
                encoding="utf-8"
            ) as f:

                f.write(code)

            p = subprocess.run(
                [
                    "python",
                    "-I",
                    path
                ],
                capture_output=True,
                text=True,
                timeout=5,
                cwd=d
            )

            output = (
                p.stdout or ""
            ) + (
                p.stderr or ""
            )

            if not output.strip():

                output = (
                    "Program finished "
                    "without printed output."
                )

            return jsonify({
                "output":
                    output[:12000]
            })

    except subprocess.TimeoutExpired:

        return jsonify({
            "output":
                "Execution stopped: "
                "time limit exceeded."
        })

    except Exception as e:

        return jsonify({
            "output":
                "Runner error: "
                + str(e)
        })


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            5000
        )
    )

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )
````


