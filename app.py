
import os
import sys
import re
import traceback

from flask import Flask, jsonify, render_template_string, request

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "src")
sys.path.insert(0, SRC)

from models import SimplifiedLesk, NLTKLeskModel

app = Flask(__name__)

PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Lesk WSD Explorer</title>
<style>
body { margin:0; background:#f4f7fb; color:#182338;
       font-family:Arial,sans-serif; }
main { max-width:850px; margin:40px auto; padding:20px; }
header { background:#182b50; color:white; padding:28px;
         border-radius:16px; }
.panel { background:white; padding:22px; margin-top:18px;
         border-radius:14px; box-shadow:0 3px 15px #15264b0d; }
label { display:block; margin:15px 0 7px; font-weight:bold; }
textarea,input { box-sizing:border-box; width:100%; padding:12px;
                 border:1px solid #ccd4e0; border-radius:8px;
                 font:inherit; }
button { margin-top:18px; background:#2458d3; color:white;
         border:0; border-radius:8px; padding:12px 20px;
         cursor:pointer; font-weight:bold; }
button:disabled { opacity:.6; }
.result { background:#edf4ff; padding:16px; border-radius:10px;
          margin-top:12px; overflow-wrap:anywhere; }
.small { color:#637087; font-size:14px; }
</style>
</head>
<body>
<main>
<header>
<h1>Lesk WSD Explorer</h1>
<p>Word Sense Disambiguation using WordNet</p>
</header>
<section class="panel">
<h2>Analyze a sentence</h2>
<form id="form">
<label for="sentence">Sentence</label>
<textarea id="sentence" rows="3" required>The fisherman sat on the bank of the river.</textarea>
<label for="target">Target word</label>
<input id="target" value="bank" required>
<label for="window">Context window: <span id="size">7</span></label>
<input id="window" type="range" min="1" max="15" value="7">
<p class="small">Number of words considered on each side
of the target word.</p>
<button id="run" type="submit">Run disambiguation</button>
</form>
<div id="output" aria-live="polite"></div>
</section>
</main>

<script>
const slider = document.getElementById("window");

slider.addEventListener("input", () => {
  document.getElementById("size").textContent = slider.value;
});

document.getElementById("form").addEventListener("submit", async e => {
  e.preventDefault();

  const button = document.getElementById("run");
  const output = document.getElementById("output");

  button.disabled = true;
  output.textContent = "Analyzing sentence...";

  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({
        sentence: document.getElementById("sentence").value,
        target_word: document.getElementById("target").value,
        window_size: Number(slider.value)
      })
    });

    // Read the response as text first. Vercel can return
    // HTML or plain text when a serverless function fails.
    const raw = await response.text();
    let data;

    try {
      data = JSON.parse(raw);
    } catch {
      console.error("Non-JSON API response:", raw);
      throw new Error(
        "Server returned HTTP " + response.status +
        " with a non-JSON response: " + raw.slice(0, 300)
      );
    }

    if (!response.ok) {
      throw new Error(
        data.error || "Server returned HTTP " + response.status
      );
    }

    output.replaceChildren();

    const heading = document.createElement("h3");
    heading.textContent = "Predictions";
    output.appendChild(heading);

    const cards = [
      ["Custom Simplified Lesk", data.custom],
      ["NLTK Lesk baseline", data.nltk]
    ];

    for (const [title, result] of cards) {
      const card = document.createElement("div");
      card.className = "result";

      const h = document.createElement("h3");
      h.textContent = title;
      card.appendChild(h);

      const prediction = document.createElement("p");
      prediction.textContent = "Predicted synset: " +
        (result.predicted_synset || "No sense found");
      card.appendChild(prediction);

      if (result.definition) {
        const definition = document.createElement("p");
        definition.textContent = "Definition: " + result.definition;
        card.appendChild(definition);
      }

      if (result.score !== undefined && result.score !== null) {
        const score = document.createElement("p");
        score.textContent = "Score: " + result.score;
        card.appendChild(score);
      }

      if (result.context) {
        const context = document.createElement("p");
        context.textContent = "Context: " + result.context.join(", ");
        card.appendChild(context);
      }

      if (result.overlap_words) {
        const overlap = document.createElement("p");
        overlap.textContent = "Overlap words: " +
          result.overlap_words.join(", ");
        card.appendChild(overlap);
      }

      if (result.candidates) {
        const list = document.createElement("ol");

        for (const candidate of result.candidates.slice(0, 5)) {
          const item = document.createElement("li");
          item.textContent = candidate.name + " — " +
            candidate.score + " — " + candidate.definition;
          list.appendChild(item);
        }

        card.appendChild(list);
      }

      output.appendChild(card);
    }
  } catch (error) {
    console.error("Disambiguation request failed:", error);
    output.textContent = "Error: " + error.message;
  } finally {
    button.disabled = false;
  }
});
</script>
</body>
</html>
"""


@app.get("/")
def home():
    return render_template_string(PAGE)


@app.get("/api/health")
def health():
    return jsonify(status="ok", message="Flask API is running")


@app.post("/api/predict")
def predict():
    data = request.get_json(silent=True) or {}

    sentence = str(data.get("sentence", "")).strip()
    target = str(data.get("target_word", "")).strip().lower()

    try:
        window_size = int(data.get("window_size", 7))
    except (TypeError, ValueError):
        return jsonify(error="Window size must be an integer."), 400

    if not sentence or not target:
        return jsonify(
            error="Enter a sentence and target word."
        ), 400

    if not 1 <= window_size <= 15:
        return jsonify(
            error="Window size must be between 1 and 15."
        ), 400

    if not re.search(
        r"\b" + re.escape(target) + r"\b",
        sentence,
        flags=re.IGNORECASE
    ):
        return jsonify(
            error="Target word was not found in the sentence."
        ), 400

    try:
        custom = SimplifiedLesk(
            window_size=window_size
        ).disambiguate(sentence, target)

        nltk_result = NLTKLeskModel().disambiguate(
            sentence, target
        )

        custom_synset = custom.get("predicted_synset")
        nltk_synset = nltk_result.get("predicted_synset")

        return jsonify({
            "custom": {
                "predicted_synset": custom.get("predicted_name"),
                "definition": (
                    custom_synset.definition()
                    if custom_synset else None
                ),
                "score": custom.get("score"),
                "context": custom.get("context", []),
                "overlap_words": custom.get("overlap_words", []),
                "candidates": [
                    {
                        "name": item["name"],
                        "score": item["score"],
                        "definition": item["definition"]
                    }
                    for item in custom.get("candidates", [])[:5]
                ]
            },
            "nltk": {
                "predicted_synset": (
                    nltk_synset.name() if nltk_synset else None
                ),
                "definition": (
                    nltk_synset.definition()
                    if nltk_synset else None
                )
            }
        })

    except LookupError as error:
        app.logger.exception("Missing NLTK resource")

        # Return the specific LookupError for diagnosis.
        return jsonify(
            error="Missing NLTK resource: " + str(error)
        ), 500

    except Exception:
        app.logger.exception("Prediction failed")

        # Keep detailed tracebacks in server logs.
        return jsonify(
            error="Prediction failed. Check Vercel runtime logs."
        ), 500


if __name__ == "__main__":
    app.run(debug=False)
