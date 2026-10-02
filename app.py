from flask import Flask, render_template, request, jsonify
import requests
import base64
import time
import re
import json
import urllib.parse
import subprocess
import tempfile
import os

app = Flask(__name__)

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
TEXT_MODEL = "qwen2.5:0.5b"
VISION_MODEL = "qwen2.5vl:3b"


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
        r = requests.post(OLLAMA_URL, json=payload, timeout=180)
        elapsed = round(time.time() - start, 2)
        r.raise_for_status()
        return {
            "response": r.json().get("response", ""),
            "time": elapsed,
            "error": None
        }
    except requests.exceptions.ConnectionError:
        return {"response": "", "time": 0,
                "error": "Ollama is not running. Start Ollama first."}
    except requests.exceptions.Timeout:
        return {"response": "", "time": 0,
                "error": "Ollama took too long to respond."}
    except Exception as e:
        return {"response": "", "time": 0, "error": str(e)}


def analyze_question(question):
    q = question.lower()

    programming = any(x in q for x in [
        "code", "program", "python", "java", "c language", "c++",
        "javascript", "javascript", "implement", "write a program",
        "algorithm", "debug", "sort", "search", "stack", "queue",
        "linked list", "tree", "graph", "recursion"
    ])

    numerical = any(x in q for x in [
        "solve", "calculate", "find", "equation", "numerical",
        "compute", "value of", "determine"
    ])

    diagram = any(x in q for x in [
        "diagram", "architecture", "circuit", "flowchart", "motor",
        "engine", "beam", "truss", "transformer", "osi", "tcp",
        "network", "process", "working", "structure", "block diagram"
    ])

    complexity = programming or any(x in q for x in [
        "complexity", "big o", "time complexity", "space complexity"
    ])

    topics = [
        "binary search", "linear search", "bubble sort", "selection sort",
        "insertion sort", "merge sort", "quick sort", "stack", "queue",
        "linked list", "osi model", "tcp handshake", "ohm law",
        "projectile", "pendulum", "four stroke engine"
    ]

    interactive = next((x for x in topics if x in q), None)

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


def build_prompt(question):
    return f"""
You are NOVA, a general educational assistant for college students.

You can answer CSE, Civil, Mechanical, Electrical, Electronics,
Mathematics, Physics, Chemistry, Biology, Architecture and other subjects.

Question:
{question}

Create a clear, accurate, beginner-friendly educational solution.

IMPORTANT:
- Do not invent facts.
- Do not force code when the question is not programming-related.
- For programming/DSA, include correct code, expected output, time complexity
  and space complexity.
- For numerical questions, show Given, Formula, Substitution, Calculation
  and Final Answer.
- For engineering/science questions, explain components, working and applications.
- If a diagram is useful, provide Mermaid flowchart/graph syntax.
- Give a useful YouTube search phrase for a related educational demonstration.
- Keep headings exactly as shown below.

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
```
If no diagram is useful, write NONE.

[CODE]
```python
# code only when applicable
```
If code is not applicable, write NONE.

[OUTPUT]
expected output or result; otherwise NONE

[COMPLEXITY]
Time: ...
Space: ...
or NONE

[APPLICATIONS]
- application
- application

[SUMMARY]
short revision summary

[VIDEO_QUERY]
short YouTube educational search phrase

[END]

Question again:
{question}
"""


def clean_fence(value):
    value = value.strip()
    value = re.sub(r"^```[a-zA-Z0-9_+-]*\s*", "", value)
    value = re.sub(r"\s*```$", "", value)
    return value.strip()


def extract_section(text, name):
    pattern = rf"\[{re.escape(name)}\]\s*(.*?)(?=\n\[[A-Z_]+\]|\Z)"
    m = re.search(pattern, text, re.S | re.I)
    return m.group(1).strip() if m else ""


def parse_answer(raw, question):
    title = extract_section(raw, "TITLE") or question[:80]
    theory = extract_section(raw, "THEORY")
    steps = extract_section(raw, "STEPS")
    algorithm = extract_section(raw, "ALGORITHM")
    example = extract_section(raw, "EXAMPLE")
    diagram = extract_section(raw, "DIAGRAM")
    code = extract_section(raw, "CODE")
    output = extract_section(raw, "OUTPUT")
    complexity = extract_section(raw, "COMPLEXITY")
    applications = extract_section(raw, "APPLICATIONS")
    summary = extract_section(raw, "SUMMARY")
    video_query = extract_section(raw, "VIDEO_QUERY")

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

    step_list = [
        re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", x).strip()
        for x in steps.splitlines()
        if x.strip()
    ]
    algorithm_list = [
        re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", x).strip()
        for x in algorithm.splitlines()
        if x.strip()
    ]
    app_list = [
        re.sub(r"^\s*[-*]\s*", "", x).strip()
        for x in applications.splitlines()
        if x.strip()
    ]

    analysis = analyze_question(question)

    # If the model supplied actual code/diagram/complexity, show them.
    analysis["code"] = bool(code) or analysis["code"]
    analysis["run"] = bool(code)
    analysis["output"] = bool(output)
    analysis["diagram"] = bool(diagram) or analysis["diagram"]
    analysis["complexity"] = bool(complexity) or analysis["complexity"]

    # Never show a fake interactive model for an unrelated topic.
    analysis["interactive"] = bool(analysis["interactiveTopic"])

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
        "videoQuery": video_query or title,
        "analysis": analysis
    }


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    question = data.get("message", "").strip()

    if not question:
        return jsonify({"error": "Please enter a question."}), 400

    result = ask_ollama(build_prompt(question))

    if result["error"]:
        return jsonify({"error": result["error"]}), 503

    answer = parse_answer(result["response"], question)

    query = urllib.parse.quote_plus(answer["videoQuery"])
    answer["videoUrl"] = "https://www.youtube.com/results?search_query=" + query
    answer["responseTime"] = result["time"]

    return jsonify(answer)


@app.route("/vision", methods=["POST"])
def vision():
    image_file = request.files.get("image")
    question = request.form.get("question", "").strip()

    if not image_file:
        return jsonify({"error": "Please select an image."}), 400

    allowed = ["image/png", "image/jpeg", "image/jpg", "image/webp"]
    if image_file.content_type not in allowed:
        return jsonify({"error": "Please use PNG, JPG or WEBP."}), 400

    image_data = image_file.read()
    if len(image_data) > 8 * 1024 * 1024:
        return jsonify({"error": "Image must be smaller than 8 MB."}), 400

    encoded = base64.b64encode(image_data).decode("utf-8")

    prompt = f"""
You are NOVA, an educational assistant.
Analyze the uploaded educational image.

User question:
{question or "Explain and solve what is shown in the image."}

Give a proper solution. If it is mathematics, solve step-by-step.
If it is programming, identify/correct code and give output.
If it is engineering/science, explain components, working and diagram.
Be clear about anything unreadable.
"""

    result = ask_ollama(prompt, model=VISION_MODEL, image=encoded)

    if result["error"]:
        return jsonify({
            "error": result["error"] +
            "\n\nFor image questions install:\n"
            "ollama pull qwen2.5vl:3b"
        }), 503

    return jsonify({
        "response": result["response"],
        "responseTime": result["time"],
        "analysis": analyze_question(question or "image question")
    })


@app.route("/health")
def health():
    result = {"ollama": False, "textModel": False, "visionModel": False}
    try:
        r = requests.get("http://127.0.0.1:11434/api/tags", timeout=5)
        if r.ok:
            result["ollama"] = True
            names = [m.get("name", "") for m in r.json().get("models", [])]
            result["textModel"] = any(
                n == TEXT_MODEL or n.startswith(TEXT_MODEL + ":") for n in names
            )
            result["visionModel"] = any(
                n == VISION_MODEL or n.startswith(VISION_MODEL + ":") for n in names
            )
    except Exception:
        pass
    return jsonify(result)


@app.route("/run-python", methods=["POST"])
def run_python():
    data = request.get_json(silent=True) or {}
    code = data.get("code", "")

    if not code.strip():
        return jsonify({"output": "No code provided."})

    # This endpoint is intended for your local demo.
    # It blocks common filesystem/network/process operations.
    blocked = [
        "import os", "from os", "import sys", "from sys",
        "subprocess", "socket", "shutil", "pathlib",
        "__import__", "eval(", "exec(", "open(",
        "requests", "urllib", "http.client"
    ]

    low = code.lower()
    if any(item in low for item in blocked):
        return jsonify({
            "output": "Execution blocked: unsafe operation detected."
        })

    try:
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "main.py")
            with open(path, "w", encoding="utf-8") as f:
                f.write(code)

            p = subprocess.run(
                ["python", "-I", path],
                capture_output=True,
                text=True,
                timeout=5,
                cwd=d
            )

            output = (p.stdout or "") + (p.stderr or "")
            if not output.strip():
                output = "Program finished without printed output."

            return jsonify({"output": output[:12000]})

    except subprocess.TimeoutExpired:
        return jsonify({"output": "Execution stopped: time limit exceeded."})
    except Exception as e:
        return jsonify({"output": "Runner error: " + str(e)})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
