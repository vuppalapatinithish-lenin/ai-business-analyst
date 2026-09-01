from flask import Flask, render_template, request, jsonify
import subprocess
import sys
import os


# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)


# ============================================================
# RUN AI BUSINESS ANALYST
# ============================================================

def run_business_analyst(question):

    try:

        # Project folder
        project_dir = os.path.dirname(
            os.path.abspath(__file__)
        )

        # ----------------------------------------------------
        # Force UTF-8
        # This fixes Windows emoji / cp1252 errors
        # ----------------------------------------------------

        env = os.environ.copy()

        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUTF8"] = "1"

        # ----------------------------------------------------
        # Run AI Business Analyst
        # ----------------------------------------------------

        process = subprocess.run(

            [
                sys.executable,
                "ai_business_analyst.py"
            ],

            input=question + "\n",

            text=True,

            capture_output=True,

            cwd=project_dir,

            env=env,

            encoding="utf-8",

            errors="replace",

            timeout=120
        )

        # ----------------------------------------------------
        # Get output
        # ----------------------------------------------------

        output = process.stdout

        # ----------------------------------------------------
        # Check for errors
        # ----------------------------------------------------

        if process.returncode != 0:

            error_output = process.stderr

            if not error_output:

                error_output = output

            return {
                "success": False,
                "answer": error_output
            }

        # ----------------------------------------------------
        # Extract final AI answer
        # ----------------------------------------------------

        marker = "AI ANSWER"

        if marker in output:

            answer_part = output.split(
                marker,
                1
            )[1]

            # Remove separator characters
            answer_part = answer_part.replace(
                "=",
                ""
            ).strip()

            if answer_part:

                return {
                    "success": True,
                    "answer": answer_part,
                    "full_output": output
                }

        # ----------------------------------------------------
        # Fallback
        # ----------------------------------------------------

        return {
            "success": True,
            "answer": output,
            "full_output": output
        }

    except subprocess.TimeoutExpired:

        return {
            "success": False,
            "answer": "Request timed out. Please try again."
        }

    except Exception as e:

        return {
            "success": False,
            "answer": str(e)
        }


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# ============================================================
# ASK AI BUSINESS ANALYST
# ============================================================

@app.route(
    "/ask",
    methods=["POST"]
)
def ask():

    try:

        data = request.get_json()

        if not data:

            return jsonify({
                "success": False,
                "answer": "No request data received."
            })

        question = data.get(
            "question",
            ""
        ).strip()

        # ----------------------------------------------------
        # Empty question
        # ----------------------------------------------------

        if not question:

            return jsonify({
                "success": False,
                "answer": "Please enter a question."
            })

        print(
            f"\nUser Question: {question}"
        )

        # ----------------------------------------------------
        # Run AI
        # ----------------------------------------------------

        result = run_business_analyst(
            question
        )

        print(
            "\nAI Result:"
        )

        print(
            result["answer"]
        )

        return jsonify(result)

    except Exception as e:

        return jsonify({
            "success": False,
            "answer": str(e)
        })


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route("/health")
def health():

    return jsonify({
        "status": "ok",
        "application": "AI Business Analyst"
    })


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    print()
    print("======================================")
    print("       AI BUSINESS ANALYST")
    print("======================================")
    print()
    print("Web application starting...")
    print()
    print("Open:")
    print("http://127.0.0.1:5000")
    print()
    print("Press CTRL+C to stop the server.")
    print()

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )