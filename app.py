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

# ============================================================
# GEMINI CONFIGURATION
# ============================================================

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

GEMINI_MODEL = "gemini-3.8-flash"

GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta/interactions"
)


# ============================================================
# GEMINI API
# ============================================================

def ask_gemini(prompt, image_data=None, mime_type=None):

    if not GEMINI_API_KEY:
        return {
            "response": "",
            "time": 0,
            "error": (
                "GEMINI_API_KEY is not configured on Render."
            )
        }

    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": GEMINI_API_KEY
    }

    # Normal text request
    if not image_data:

        payload = {
            "model": GEMINI_MODEL,
            "input": prompt
        }

    # Image + text request
    else:

        payload = {
            "model": GEMINI_MODEL,
            "input": [
                {
                    "type": "text",
                    "text": prompt
                },
                {
                    "type": "image",
                    "data": image_data,
                    "mime_type": mime_type or "image/jpeg"
                }
            ]
        }

    try:

        start = time.time()

        response = requests.post(
            GEMINI_URL,
            headers=headers,
            json=payload,
            timeout=180
        )

        elapsed = round(
            time.time() - start,
            2
        )

        if not response.ok:

            try:
                error_data = response.json()

                error_message = (
                    error_data
                    .get("error", {})
                    .get("message", response.text)
                )

            except Exception:

                error_message = response.text

            return {
                "response": "",
                "time": elapsed,
                "error": (
                    f"Gemini API error "
                    f"({response.status_code}): "
                    f"{error_message}"
                )
            }

        data = response.json()

        # Current Interactions API response
        # contains model_output steps.
        output_text = ""

        for step in data.get("steps", []):

            if step.get("type") != "model_output":
                continue

            for content in step.get(
                "content",
                []
            ):

                if content.get("type") == "text":

                    output_text += content.get(
                        "text",
                        ""
                    )

        output_text = output_text.strip()

        if not output_text:

            return {
                "response": "",
                "time": elapsed,
                "error": (
                    "Gemini returned an empty answer."
                )
            }

        return {
            "response": output_text,
            "time": elapsed,
            "error": None
        }

    except requests.exceptions.Timeout:

        return {
            "response": "",
            "time": 0,
            "error": (
                "Gemini took too long to respond."
            )
        }

    except requests.exceptions.ConnectionError:

        return {
            "response": "",
            "time": 0,
            "error": (
                "Could not connect to Gemini API."
            )
        }

    except Exception as e:

        return {
            "response": "",
            "time": 0,
            "error": (
                "Gemini connection error: "
                + str(e)
            )
        }


# ============================================================
# QUESTION ANALYSIS
# ============================================================

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


# ============================================================
# NEXO PROMPT
# ============================================================

def build_prompt(question):

    prompt = """
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
QUESTION_PLACEHOLDER

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
16. Make Mermaid diagrams valid.
17. Keep the headings EXACTLY as shown below.

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

    return prompt.replace(
        "QUESTION_PLACEHOLDER",
        question
    )


# ============================================================
# TEXT CLEANING
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


# ============================================================
# PARSE ANSWER
# ============================================================

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
        diagram = clean_fence(
            diagram
        )

    if code.upper() == "NONE":
        code = ""
    else:
        code = clean_fence(
            code
        )

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

    data = request.get_json(
        silent=True
    ) or {}

    question = data.get(
        "message",
        ""
    ).strip()

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
        "https://www.youtube.com/results?search_query="
        + query
    )

    answer["responseTime"] = (
        result["time"]
    )

    return jsonify(answer)


# ============================================================
# IMAGE QUESTION
# ============================================================

@app.route(
    "/vision",
    methods=["POST"]
)
def vision():

    image_file = request.files.get(
        "image"
    )

    question = request.form.get(
        "question",
        ""
    ).strip()

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
{question or "Explain and solve what is shown in the image."}

Give a proper solution.

If it is mathematics:
- identify the problem
- solve step-by-step
- show formulas
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
        "analysis":
            analyze_question(
                question
                or "image question"
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
        "model":
            GEMINI_MODEL
    })


# ============================================================
# PYTHON CODE RUNNER
# ============================================================

@app.route(
    "/run-python",
    methods=["POST"]
)
def run_python():

    data = request.get_json(
        silent=True
    ) or {}

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

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )
