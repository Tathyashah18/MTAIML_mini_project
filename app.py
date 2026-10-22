
import os
import sys
import re
import csv

from flask import Flask, jsonify, render_template_string, request

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "src")

if SRC not in sys.path:
    sys.path.insert(0, SRC)

from nltk.corpus import wordnet as wn
from models import SimplifiedLesk, NLTKLeskModel

app = Flask(__name__)

DATA_FILE = os.path.join(ROOT, "data", "annotated.csv")
DEFAULT_WINDOW_SIZE = 5


# =========================================================
# WEBPAGE
# =========================================================

PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Lesk WSD Explorer</title>

<style>
* { box-sizing: border-box; }

body {
    margin: 0;
    background: #f4f7fb;
    color: #182338;
    font-family: Arial, sans-serif;
}

main {
    max-width: 1150px;
    margin: 35px auto;
    padding: 20px;
}

header {
    background: #182b50;
    color: white;
    padding: 28px;
    border-radius: 16px;
}

header p { color: #dbe7ff; }

nav {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    margin-top: 22px;
}

nav button {
    margin: 0;
    background: white;
    color: #182b50;
}

nav button.active {
    background: #2458d3;
    color: white;
}

.panel {
    background: white;
    padding: 22px;
    margin-top: 18px;
    border-radius: 14px;
    box-shadow: 0 3px 15px #15264b0d;
}

.hidden { display: none !important; }

label {
    display: block;
    margin: 15px 0 7px;
    font-weight: bold;
}

textarea, input {
    width: 100%;
    padding: 12px;
    border: 1px solid #ccd4e0;
    border-radius: 8px;
    font: inherit;
}

button {
    margin-top: 18px;
    background: #2458d3;
    color: white;
    border: 0;
    border-radius: 8px;
    padding: 12px 20px;
    cursor: pointer;
    font-weight: bold;
}

button:disabled {
    opacity: .6;
    cursor: wait;
}

.small {
    color: #637087;
    font-size: 14px;
}

.result {
    background: #edf4ff;
    padding: 16px;
    border-radius: 10px;
    margin-top: 12px;
    overflow-wrap: anywhere;
}

.metrics {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 12px;
    margin: 18px 0;
}

.metric {
    background: #f1f5fb;
    padding: 17px;
    border-radius: 10px;
    min-width: 0;
}

.metric .value {
    font-size: 26px;
    font-weight: bold;
    margin-top: 8px;
    overflow-wrap: anywhere;
}

.metric .label {
    color: #637087;
    font-size: 13px;
}

.table-wrap {
    width: 100%;
    overflow-x: auto;
    margin-top: 16px;
}

table {
    width: 100%;
    border-collapse: collapse;
    font-size: 14px;
    min-width: 900px;
}

th, td {
    text-align: left;
    padding: 12px;
    border-bottom: 1px solid #e1e7f0;
    vertical-align: top;
    overflow-wrap: anywhere;
}

th {
    background: #edf2fa;
    position: sticky;
    top: 0;
}

.status-correct {
    color: #087443;
    font-weight: bold;
}

.status-wrong {
    color: #bd3030;
    font-weight: bold;
}

.notice {
    padding: 13px;
    border-radius: 8px;
    margin-top: 14px;
    background: #fff5dc;
    color: #76520a;
    overflow-wrap: anywhere;
}

.success {
    background: #e7f8ef;
    color: #11663c;
}

.error {
    background: #fff0f0;
    color: #a42323;
}

.bar-row {
    display: grid;
    grid-template-columns: 160px minmax(0, 1fr) 65px;
    gap: 10px;
    align-items: center;
    margin: 13px 0;
}

.bar-track {
    background: #e5eaf2;
    border-radius: 8px;
    height: 18px;
    overflow: hidden;
}

.bar {
    height: 100%;
    background: #2458d3;
    width: 0;
    transition: width .2s;
}

.bar.nltk { background: #15976b; }

@media (max-width: 700px) {
    main { margin: 10px auto; padding: 12px; }
    header, .panel { padding: 17px; }
    .metrics { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    .bar-row { grid-template-columns: 105px minmax(0, 1fr) 50px; }
}
</style>
</head>

<body>
<main>
<header>
    <h1>Lesk WSD Explorer</h1>
    <p>Word Sense Disambiguation using WordNet</p>
    <p>Compare a custom Simplified Lesk implementation with NLTK's
       built-in Lesk algorithm.</p>

    <nav>
        <button type="button" class="active" id="tabAnalyze"
                onclick="showTab('analyze')">
            Sentence Analyzer
        </button>
        <button type="button" id="tabEvaluate"
                onclick="showTab('evaluate')">
            Dataset Evaluation
        </button>
    </nav>
</header>

<section class="panel" id="analyzePanel">
    <h2>Analyze a sentence</h2>
    <p class="small">
        Enter a sentence and a target word to compare both models.
    </p>

    <form id="form">
        <label for="sentence">Sentence</label>
        <textarea id="sentence" rows="3" required>The fisherman sat on the bank of the river.</textarea>

        <label for="target">Target word</label>
        <input id="target" value="bank" required>

        <button id="run" type="submit">Run disambiguation</button>
    </form>

    <div id="output" aria-live="polite"></div>
</section>

<section class="panel hidden" id="evaluatePanel">
    <h2>Dataset Evaluation</h2>
    <p>
        Evaluate both models against the gold-standard synsets in
        your annotated CSV dataset.
    </p>

    <p class="small">
        Dataset path: data/annotated.csv
    </p>
    

    <button id="evaluateButton" type="button">
        Run dataset comparison
    </button>

    <div id="evaluationStatus" aria-live="polite"></div>
    <div id="evaluationResults"></div>
</section>
</main>

<script>
function showTab(tab) {
    const analyze = tab === "analyze";

    document.getElementById("analyzePanel").classList.toggle(
        "hidden", !analyze
    );
    document.getElementById("evaluatePanel").classList.toggle(
        "hidden", analyze
    );

    document.getElementById("tabAnalyze").classList.toggle(
        "active", analyze
    );
    document.getElementById("tabEvaluate").classList.toggle(
        "active", !analyze
    );
}

function addText(parent, tag, text, className = "") {
    const el = document.createElement(tag);
    el.textContent = text == null ? "" : String(text);
    if (className) el.className = className;
    parent.appendChild(el);
    return el;
}

async function postJSON(url, body) {
    const response = await fetch(url, {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(body)
    });

    const raw = await response.text();
    let data;

    try {
        data = JSON.parse(raw);
    } catch {
        throw new Error(
            "Server returned HTTP " + response.status +
            " with a non-JSON response: " + raw.slice(0, 250)
        );
    }

    if (!response.ok) {
        throw new Error(data.error || "HTTP " + response.status);
    }

    return data;
}


// =====================================================
// SINGLE SENTENCE PREDICTION
// =====================================================

document.getElementById("form").addEventListener("submit", async e => {
    e.preventDefault();

    const button = document.getElementById("run");
    const output = document.getElementById("output");

    button.disabled = true;
    output.replaceChildren();
    addText(output, "p", "Analyzing sentence...");

    try {
        const data = await postJSON("/api/predict", {
            sentence: document.getElementById("sentence").value,
            target_word: document.getElementById("target").value
        });

        output.replaceChildren();
        addText(output, "h3", "Predictions");

        const cards = [
            ["Custom Simplified Lesk", data.custom],
            ["NLTK Lesk baseline", data.nltk]
        ];

        for (const [title, result] of cards) {
            const card = document.createElement("div");
            card.className = "result";
            addText(card, "h3", title);

            addText(
                card, "p",
                "Predicted synset: " +
                (result.predicted_synset || "No sense found")
            );

            if (result.definition) {
                addText(card, "p", "Definition: " + result.definition);
            }

            if (result.score !== undefined && result.score !== null) {
                addText(card, "p", "Score: " + result.score);
            }

            if (result.context && result.context.length) {
                addText(
                    card, "p",
                    "Context: " + result.context.join(", ")
                );
            }

            if (result.overlap_words && result.overlap_words.length) {
                addText(
                    card, "p",
                    "Overlap words: " + result.overlap_words.join(", ")
                );
            }

            if (result.candidates && result.candidates.length) {
                addText(card, "h4", "Top custom candidates");
                const list = document.createElement("ol");

                for (const candidate of result.candidates.slice(0, 5)) {
                    addText(
                        list, "li",
                        candidate.name + " — " +
                        candidate.score + " — " +
                        candidate.definition
                    );
                }

                card.appendChild(list);
            }

            output.appendChild(card);
        }
    } catch (error) {
        output.replaceChildren();
        addText(output, "p", "Error: " + error.message, "notice error");
    } finally {
        button.disabled = false;
    }
});


// =====================================================
// DATASET EVALUATION
// =====================================================

document.getElementById("evaluateButton").addEventListener(
    "click", async () => {
        const button = document.getElementById("evaluateButton");
        const status = document.getElementById("evaluationStatus");
        const resultsContainer = document.getElementById("evaluationResults");

        button.disabled = true;
        status.replaceChildren();
        resultsContainer.replaceChildren();
        addText(status, "p", "Evaluating dataset. Please wait...");

        try {
            const data = await postJSON("/api/evaluate", {});

            status.replaceChildren();

            if (data.failed > 0) {
                addText(
                    status, "p",
                    "Completed with " + data.failed +
                    " failed row(s). Accuracy uses " +
                    data.metrics.total + " successfully evaluated rows.",
                    "notice"
                );
            } else {
                addText(
                    status, "p",
                    "Evaluation completed successfully.",
                    "notice success"
                );
            }

            const m = data.metrics;
            const metrics = document.createElement("div");
            metrics.className = "metrics";

            const metricItems = [
                ["Dataset rows", data.dataset_rows],
                ["Evaluated rows", m.total],
                ["Custom correct", m.custom_correct],
                ["NLTK correct", m.nltk_correct]
            ];

            for (const [label, value] of metricItems) {
                const box = document.createElement("div");
                box.className = "metric";
                addText(box, "div", label, "label");
                addText(box, "div", value, "value");
                metrics.appendChild(box);
            }

            resultsContainer.appendChild(metrics);

            const summary = document.createElement("div");
            summary.className = "result";
            addText(summary, "h3", "Accuracy comparison");

            const chartRows = [
                ["Custom Simplified Lesk", m.custom_accuracy, false],
                ["NLTK Lesk", m.nltk_accuracy, true]
            ];

            for (const [label, accuracy, isNltk] of chartRows) {
                const row = document.createElement("div");
                row.className = "bar-row";

                addText(row, "div", label);

                const track = document.createElement("div");
                track.className = "bar-track";

                const bar = document.createElement("div");
                bar.className = "bar" + (isNltk ? " nltk" : "");
                bar.style.width = Math.max(
                    0, Math.min(100, accuracy)
                ) + "%";

                track.appendChild(bar);
                row.appendChild(track);

                addText(row, "strong", accuracy.toFixed(2) + "%");
                summary.appendChild(row);
            }

            addText(
                summary, "p",
                "Accuracy difference (Custom − NLTK): " +
                (m.custom_accuracy - m.nltk_accuracy >= 0 ? "+" : "") +
                (m.custom_accuracy - m.nltk_accuracy).toFixed(2) + "%"
            );

            resultsContainer.appendChild(summary);

            // Five-metric comparison (macro-averaged per-synset metrics).
            const metricTableWrap = document.createElement("div");
            metricTableWrap.className = "table-wrap";
            const metricTable = document.createElement("table");
            metricTable.style.minWidth = "650px";
            const metricHead = document.createElement("thead");
            const metricHeadRow = document.createElement("tr");
            for (const heading of [
                "Model", "Accuracy", "Macro precision", "Macro recall",
                "Macro F1-score", "Macro specificity"
            ]) {
                addText(metricHeadRow, "th", heading);
            }
            metricHead.appendChild(metricHeadRow);
            metricTable.appendChild(metricHead);

            const metricBody = document.createElement("tbody");
            const metricRows = [
                ["Custom Simplified Lesk", "custom"],
                ["NLTK Lesk", "nltk"]
            ];
            for (const [modelName, prefix] of metricRows) {
                const row = document.createElement("tr");
                addText(row, "td", modelName);
                const values = [
                    m[prefix + "_accuracy"],
                    m[prefix + "_macro_precision"],
                    m[prefix + "_macro_recall"],
                    m[prefix + "_macro_f1"],
                    m[prefix + "_macro_specificity"]
                ];
                for (const value of values) {
                    addText(row, "td", Number(value).toFixed(2) + "%");
                }
                metricBody.appendChild(row);
            }
            metricTable.appendChild(metricBody);
            metricTableWrap.appendChild(metricTable);
            resultsContainer.appendChild(metricTableWrap);

            addText(resultsContainer, "p",
                "Precision, recall, F1-score, and specificity are macro-averaged " +
                "across the synset labels (one-vs-rest). Matrix rows are gold/actual " +
                "labels; columns are predicted labels. A missing prediction is shown " +
                "as '(no prediction).'", "small");

            function renderConfusionMatrix(title, matrix, labels) {
                const section = document.createElement("section");
                section.className = "result";
                addText(section, "h3", title);
                const wrap = document.createElement("div");
                wrap.className = "table-wrap";
                const table = document.createElement("table");
                table.style.minWidth = Math.max(900, (labels.length + 1) * 115) + "px";

                const thead = document.createElement("thead");
                const headerRow = document.createElement("tr");
                addText(headerRow, "th", "Actual \\ Predicted");
                for (const label of labels) addText(headerRow, "th", label);
                thead.appendChild(headerRow);
                table.appendChild(thead);

                const tbody = document.createElement("tbody");
                matrix.forEach((counts, rowIndex) => {
                    const tr = document.createElement("tr");
                    addText(tr, "th", labels[rowIndex]);
                    counts.forEach((count, colIndex) => {
                        const td = document.createElement("td");
                        td.textContent = String(count);
                        td.style.textAlign = "center";
                        if (rowIndex === colIndex) {
                            td.style.fontWeight = "bold";
                            td.style.background = "#e7f8ef";
                        }
                        tr.appendChild(td);
                    });
                    tbody.appendChild(tr);
                });
                table.appendChild(tbody);
                wrap.appendChild(table);
                section.appendChild(wrap);
                resultsContainer.appendChild(section);
            }

            renderConfusionMatrix(
                "Custom Simplified Lesk — confusion matrix",
                m.custom_confusion_matrix, m.labels
            );
            renderConfusionMatrix(
                "NLTK Lesk — confusion matrix",
                m.nltk_confusion_matrix, m.labels
            );

            addText(resultsContainer, "h3", "Detailed test results");

            const search = document.createElement("input");
            search.placeholder = "Filter by sentence, target word, or synset...";
            search.setAttribute("aria-label", "Filter evaluation results");
            resultsContainer.appendChild(search);

            const tableWrap = document.createElement("div");
            tableWrap.className = "table-wrap";

            const table = document.createElement("table");
            const thead = document.createElement("thead");
            const header = document.createElement("tr");

            const columns = [
                "No.", "Sentence", "Target", "Gold synset",
                "Custom prediction", "Custom score",
                "Custom result", "NLTK prediction", "NLTK result"
            ];

            for (const column of columns) {
                addText(header, "th", column);
            }

            thead.appendChild(header);
            table.appendChild(thead);

            const tbody = document.createElement("tbody");
            table.appendChild(tbody);
            tableWrap.appendChild(table);
            resultsContainer.appendChild(tableWrap);

            function renderRows(filter = "") {
                tbody.replaceChildren();
                const query = filter.toLowerCase();

                const filtered = data.results.filter(r => {
                    const searchable = [
                        r.sentence, r.target_word, r.gold_synset,
                        r.custom_prediction, r.nltk_prediction
                    ].join(" ").toLowerCase();

                    return searchable.includes(query);
                });

                for (const r of filtered) {
                    const tr = document.createElement("tr");

                    const values = [
                        r.index,
                        r.sentence,
                        r.target_word,
                        r.gold_synset,
                        r.custom_prediction || "No sense found",
                        r.custom_score ?? "",
                        r.custom_correct ? "CORRECT" : "INCORRECT",
                        r.nltk_prediction || "No sense found",
                        r.nltk_correct ? "CORRECT" : "INCORRECT"
                    ];

                    for (let i = 0; i < values.length; i++) {
                        const td = document.createElement("td");
                        td.textContent = values[i] == null
                            ? "" : String(values[i]);

                        if (i === 6 || i === 8) {
                            td.className = values[i] === "CORRECT"
                                ? "status-correct"
                                : "status-wrong";
                        }

                        tr.appendChild(td);
                    }

                    tbody.appendChild(tr);
                }

                if (!filtered.length) {
                    const tr = document.createElement("tr");
                    const td = document.createElement("td");
                    td.colSpan = columns.length;
                    td.textContent = "No matching rows.";
                    tr.appendChild(td);
                    tbody.appendChild(tr);
                }
            }

            renderRows();
            search.addEventListener("input", () => renderRows(search.value));

            if (data.errors && data.errors.length) {
                const errors = document.createElement("details");
                addText(
                    errors, "summary",
                    "View failed rows (" + data.errors.length + ")"
                );

                const list = document.createElement("ul");

                for (const item of data.errors) {
                    addText(
                        list, "li",
                        "Row " + item.index + ": " + item.error
                    );
                }

                errors.appendChild(list);
                resultsContainer.appendChild(errors);
            }

        } catch (error) {
            status.replaceChildren();
            addText(
                status, "p",
                "Evaluation failed: " + error.message,
                "notice error"
            );
        } finally {
            button.disabled = false;
        }
    }
);
</script>
</body>
</html>
"""


# =========================================================
# DATASET AND EVALUATION HELPERS
# =========================================================

def load_annotated_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError("Dataset not found: " + file_path)

    data = []

    with open(
        file_path, "r", encoding="utf-8-sig", newline=""
    ) as file:
        reader = csv.DictReader(file)

        required = {"sentence", "target_word", "gold_synset"}

        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(
                "CSV must contain columns: sentence, target_word, gold_synset"
            )

        for row in reader:
            sentence = (row.get("sentence") or "").strip()
            target = (row.get("target_word") or "").strip()
            gold = (row.get("gold_synset") or "").strip()

            if not sentence or not target or not gold:
                continue

            data.append({
                "sentence": sentence,
                "target_word": target,
                "gold_synset": gold,
            })

    return data


def get_synset_name(synset):
    if synset is None:
        return None

    if isinstance(synset, str):
        return synset

    return synset.name()


def evaluate_sentence(sentence, target_word, gold_synset_name, window_size):
    gold_synset_name = str(gold_synset_name).strip()

    try:
        gold_synset = wn.synset(gold_synset_name)
    except Exception as error:
        raise ValueError(
            "Invalid WordNet gold synset: " + gold_synset_name
        ) from error

    custom_result = SimplifiedLesk(
        window_size=window_size
    ).disambiguate(sentence, target_word)

    nltk_result = NLTKLeskModel().disambiguate(
        sentence, target_word
    )

    custom_synset = custom_result.get("predicted_synset")
    nltk_synset = nltk_result.get("predicted_synset")

    custom_name = get_synset_name(
        custom_result.get("predicted_name") or custom_synset
    )
    nltk_name = get_synset_name(nltk_synset)
    gold_name = gold_synset.name()

    return {
        "sentence": sentence,
        "target_word": target_word,
        "gold_synset": gold_name,
        "custom_prediction": custom_name,
        "custom_score": custom_result.get("score", 0),
        "custom_overlap": custom_result.get("overlap_words", []),
        "custom_correct": custom_name == gold_name,
        "nltk_prediction": nltk_name,
        "nltk_correct": nltk_name == gold_name,
    }


def calculate_metrics(results):
    """Calculate accuracy plus multiclass confusion matrices and macro metrics.

    Confusion-matrix rows are gold/actual labels; columns are predictions.
    Metrics other than accuracy are computed one-vs-rest for each label and
    then macro-averaged. A missing prediction is counted as its own predicted
    label so it appears as an error in the matrix.
    """
    total = len(results)
    custom_correct = sum(1 for result in results if result["custom_correct"])
    nltk_correct = sum(1 for result in results if result["nltk_correct"])

    no_prediction = "(no prediction)"
    gold_labels = [str(r["gold_synset"]) for r in results]
    custom_preds = [
        str(r["custom_prediction"]) if r["custom_prediction"] else no_prediction
        for r in results
    ]
    nltk_preds = [
        str(r["nltk_prediction"]) if r["nltk_prediction"] else no_prediction
        for r in results
    ]

    # Include every observed gold and predicted label so no prediction is hidden.
    labels = sorted(set(gold_labels + custom_preds + nltk_preds))
    label_index = {label: i for i, label in enumerate(labels)}

    def build_matrix(predictions):
        matrix = [[0 for _ in labels] for _ in labels]
        for gold, pred in zip(gold_labels, predictions):
            matrix[label_index[gold]][label_index[pred]] += 1
        return matrix

    def summarize_matrix(matrix):
        per_class = []
        for i, label in enumerate(labels):
            tp = matrix[i][i]
            fp = sum(matrix[row][i] for row in range(len(labels))) - tp
            fn = sum(matrix[i]) - tp
            tn = total - tp - fp - fn

            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            f1 = (
                2 * precision * recall / (precision + recall)
                if precision + recall else 0.0
            )
            specificity = tn / (tn + fp) if tn + fp else 0.0

            per_class.append({
                "label": label,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "precision": precision * 100,
                "recall": recall * 100,
                "f1": f1 * 100,
                "specificity": specificity * 100,
            })

        # Macro average: each label receives equal weight.
        n_labels = len(per_class)
        macro = {
            metric: (
                sum(item[metric] for item in per_class) / n_labels
                if n_labels else 0.0
            )
            for metric in ("precision", "recall", "f1", "specificity")
        }
        return per_class, macro

    custom_matrix = build_matrix(custom_preds)
    nltk_matrix = build_matrix(nltk_preds)
    custom_per_class, custom_macro = summarize_matrix(custom_matrix)
    nltk_per_class, nltk_macro = summarize_matrix(nltk_matrix)

    return {
        "total": total,
        "custom_correct": custom_correct,
        "nltk_correct": nltk_correct,
        "custom_accuracy": custom_correct / total * 100 if total else 0.0,
        "nltk_accuracy": nltk_correct / total * 100 if total else 0.0,
        "labels": labels,
        "custom_confusion_matrix": custom_matrix,
        "nltk_confusion_matrix": nltk_matrix,
        "custom_class_metrics": custom_per_class,
        "nltk_class_metrics": nltk_per_class,
        "custom_macro_precision": custom_macro["precision"],
        "custom_macro_recall": custom_macro["recall"],
        "custom_macro_f1": custom_macro["f1"],
        "custom_macro_specificity": custom_macro["specificity"],
        "nltk_macro_precision": nltk_macro["precision"],
        "nltk_macro_recall": nltk_macro["recall"],
        "nltk_macro_f1": nltk_macro["f1"],
        "nltk_macro_specificity": nltk_macro["specificity"],
    }



# =========================================================
# ROUTES
# =========================================================

@app.get("/")
def home():
    return render_template_string(PAGE)


@app.get("/api/health")
def health():
    return jsonify(
        status="ok",
        message="Flask API is running"
    )


@app.post("/api/predict")
def predict():
    data = request.get_json(silent=True) or {}

    sentence = str(data.get("sentence", "")).strip()
    target = str(data.get("target_word", "")).strip().lower()

    if not sentence or not target:
        return jsonify(
            error="Enter a sentence and target word."
        ), 400

    if not re.search(
        r"\b" + re.escape(target) + r"\b",
        sentence,
        flags=re.IGNORECASE,
    ):
        return jsonify(
            error="Target word was not found in the sentence."
        ), 400

    try:
        custom = SimplifiedLesk(
            window_size=DEFAULT_WINDOW_SIZE
        ).disambiguate(sentence, target)

        nltk_result = NLTKLeskModel().disambiguate(
            sentence, target
        )

        custom_synset = custom.get("predicted_synset")
        nltk_synset = nltk_result.get("predicted_synset")

        return jsonify({
            "custom": {
                "predicted_synset": (
                    custom.get("predicted_name")
                    or get_synset_name(custom_synset)
                ),
                "definition": (
                    custom_synset.definition()
                    if hasattr(custom_synset, "definition")
                    else None
                ),
                "score": custom.get("score"),
                "context": custom.get("context", []),
                "overlap_words": custom.get("overlap_words", []),
                "candidates": [
                    {
                        "name": item["name"],
                        "score": item["score"],
                        "definition": item["definition"],
                    }
                    for item in custom.get("candidates", [])[:5]
                ],
            },
            "nltk": {
                "predicted_synset": get_synset_name(nltk_synset),
                "definition": (
                    nltk_synset.definition()
                    if hasattr(nltk_synset, "definition")
                    else None
                ),
            },
        })

    except LookupError as error:
        app.logger.exception("Missing NLTK resource")
        return jsonify(
            error="Missing NLTK resource: " + str(error)
        ), 500

    except Exception:
        app.logger.exception("Prediction failed")
        return jsonify(
            error="Prediction failed. Check Vercel runtime logs."
        ), 500


@app.post("/api/evaluate")
def evaluate():
    # Fixed context window: no slider or request parameter.
    window_size = DEFAULT_WINDOW_SIZE

    try:
        dataset = load_annotated_data(DATA_FILE)
    except Exception as error:
        app.logger.exception("Could not load annotated dataset")
        return jsonify(
            error="Could not load dataset: " + str(error)
        ), 500

    results = []
    errors = []

    for index, item in enumerate(dataset, start=1):
        try:
            result = evaluate_sentence(
                item["sentence"],
                item["target_word"],
                item["gold_synset"],
                window_size,
            )

            score = result.get("custom_score")
            try:
                result["custom_score"] = (
                    float(score) if score is not None else 0.0
                )
            except (TypeError, ValueError):
                result["custom_score"] = 0.0

            result["index"] = index
            results.append(result)

        except Exception as error:
            app.logger.warning(
                "Evaluation failed for CSV row %s: %s",
                index,
                error,
            )

            errors.append({
                "index": index,
                "sentence": item["sentence"],
                "error": str(error),
            })

    metrics = calculate_metrics(results)

    return jsonify({
        "dataset_rows": len(dataset),
        "evaluated": len(results),
        "failed": len(errors),
        "window_size": window_size,
        "metrics": metrics,
        "results": results,
        "errors": errors,
    })


if __name__ == "__main__":
    app.run(debug=False)
