
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

PAGE = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Lesk WSD Explorer</title>
<style>
*{box-sizing:border-box}
body{margin:0;background:#f4f7fb;color:#182338;font-family:Arial,sans-serif}
main{max-width:1150px;margin:35px auto;padding:20px}
header{background:#182b50;color:white;padding:28px;border-radius:16px}
header p{color:#dbe7ff}
nav{display:flex;flex-wrap:wrap;gap:10px;margin-top:22px}
nav button{margin:0;background:white;color:#182b50}
nav button.active{background:#2458d3;color:white}
.panel{background:white;padding:22px;margin-top:18px;border-radius:14px;box-shadow:0 3px 15px #15264b0d}
.hidden{display:none!important}
label{display:block;margin:15px 0 7px;font-weight:bold}
textarea,input{width:100%;padding:12px;border:1px solid #ccd4e0;border-radius:8px;font:inherit}
button{margin-top:18px;background:#2458d3;color:white;border:0;border-radius:8px;padding:12px 20px;cursor:pointer;font-weight:bold}
button:disabled{opacity:.6;cursor:wait}
.small{color:#637087;font-size:14px}
.result{background:#edf4ff;padding:16px;border-radius:10px;margin-top:12px;overflow-wrap:anywhere}
.metrics{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin:18px 0}
.metric{background:#f1f5fb;padding:17px;border-radius:10px;min-width:0}
.metric .value{font-size:26px;font-weight:bold;margin-top:8px;overflow-wrap:anywhere}
.metric .label{color:#637087;font-size:13px}
.table-wrap{width:100%;overflow-x:auto;margin-top:16px}
table{width:100%;border-collapse:collapse;font-size:14px;min-width:900px}
th,td{text-align:left;padding:12px;border-bottom:1px solid #e1e7f0;vertical-align:top;overflow-wrap:anywhere}
th{background:#edf2fa;position:sticky;top:0}
.status-correct{color:#087443;font-weight:bold}
.status-wrong{color:#bd3030;font-weight:bold}
.notice{padding:13px;border-radius:8px;margin-top:14px;background:#fff5dc;color:#76520a;overflow-wrap:anywhere}
.success{background:#e7f8ef;color:#11663c}
.error{background:#fff0f0;color:#a42323}
.bar-row{display:grid;grid-template-columns:160px minmax(0,1fr) 65px;gap:10px;align-items:center;margin:13px 0}
.bar-track{background:#e5eaf2;border-radius:8px;height:18px;overflow:hidden}
.bar{height:100%;background:#2458d3;width:0;transition:width .2s}
.bar.nltk{background:#15976b}
@media(max-width:700px){
 main{margin:10px auto;padding:12px}
 header,.panel{padding:17px}
 .metrics{grid-template-columns:repeat(2,minmax(0,1fr))}
 .bar-row{grid-template-columns:105px minmax(0,1fr) 50px}
}
</style>
</head>
<body>
<main>
<header>
<h1>Lesk WSD Explorer</h1>
<p>Word Sense Disambiguation using WordNet</p>
<p>Compare a custom Simplified Lesk implementation with NLTK's built-in Lesk algorithm.</p>
<nav>
<button type="button" class="active" id="tabAnalyze" onclick="showTab('analyze')">Sentence Analyzer</button>
<button type="button" id="tabEvaluate" onclick="showTab('evaluate')">Dataset Evaluation</button>
<button type="button" id="tabTargets" onclick="showTab('targets')">View target words</button>
<button type="button" id="tabDataset" onclick="showTab('dataset')">View complete dataset</button>
</nav>
</header>

<section class="panel" id="analyzePanel">
<h2>Analyze a sentence</h2>
<p class="small">Enter a sentence and a target word to compare both models.</p>
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
<p>Evaluate both models against the gold-standard synsets in your annotated CSV dataset.</p>
<p class="small">Dataset path: data/annotated.csv</p>
<button id="evaluateButton" type="button">Run dataset comparison</button>
<div id="evaluationStatus" aria-live="polite"></div>
<div id="evaluationResults"></div>
</section>

<section class="panel hidden" id="targetsPanel">
<h2>View target words</h2>
<p class="small">Unique target words read directly from data/annotated.csv.</p>
<div id="targetWordsStatus" aria-live="polite"></div>
<div id="targetWordsResults"></div>
</section>

<section class="panel hidden" id="datasetPanel">
<h2>View complete dataset</h2>
<p class="small">All columns and rows from data/annotated.csv.</p>
<div id="datasetStatus" aria-live="polite"></div>
<div id="datasetResults"></div>
</section>
</main>

<script>
function showTab(tab){
    const tabs=['analyze','evaluate','targets','dataset'];
    for(const name of tabs){
        document.getElementById(name+'Panel').classList.toggle('hidden',name!==tab);
        document.getElementById('tab'+name.charAt(0).toUpperCase()+name.slice(1)).classList.toggle('active',name===tab);
    }
    if(tab==='targets'&&!window.targetWordsLoaded)loadTargetWords();
    if(tab==='dataset'&&!window.completeDatasetLoaded)loadCompleteDataset();
}

function addText(parent,tag,text,className=''){
    const el=document.createElement(tag);
    el.textContent=text==null?'':String(text);
    if(className)el.className=className;
    parent.appendChild(el);
    return el;
}

async function postJSON(url,body){
    const response=await fetch(url,{
        method:'POST',
        headers:{'Content-Type':'application/json'},
        body:JSON.stringify(body)
    });
    const raw=await response.text();
    let data;
    try{data=JSON.parse(raw)}
    catch{
        throw new Error('Server returned HTTP '+response.status+
            ' with a non-JSON response: '+raw.slice(0,250));
    }
    if(!response.ok)throw new Error(data.error||'HTTP '+response.status);
    return data;
}

document.getElementById('form').addEventListener('submit',async e=>{
    e.preventDefault();
    const button=document.getElementById('run');
    const output=document.getElementById('output');
    button.disabled=true;
    output.replaceChildren();
    addText(output,'p','Analyzing sentence...');

    try{
        const data=await postJSON('/api/predict',{
            sentence:document.getElementById('sentence').value,
            target_word:document.getElementById('target').value
        });
        output.replaceChildren();
        addText(output,'h3','Predictions');

        for(const [title,result] of [
            ['Custom Simplified Lesk',data.custom],
            ['NLTK Lesk baseline',data.nltk]
        ]){
            const card=document.createElement('div');
            card.className='result';
            addText(card,'h3',title);
            addText(card,'p','Predicted synset: '+(result.predicted_synset||'No sense found'));
            if(result.definition)addText(card,'p','Definition: '+result.definition);
            if(result.score!==undefined&&result.score!==null)addText(card,'p','Score: '+result.score);
            if(result.context&&result.context.length)addText(card,'p','Context: '+result.context.join(', '));
            if(result.overlap_words&&result.overlap_words.length)addText(card,'p','Overlap words: '+result.overlap_words.join(', '));

            if(result.candidates&&result.candidates.length){
                addText(card,'h4','Top custom candidates');
                const list=document.createElement('ol');
                for(const c of result.candidates.slice(0,5)){
                    addText(list,'li',c.name+' — '+c.score+' — '+c.definition);
                }
                card.appendChild(list);
            }
            output.appendChild(card);
        }
    }catch(error){
        output.replaceChildren();
        addText(output,'p','Error: '+error.message,'notice error');
    }finally{
        button.disabled=false;
    }
});

async function loadTargetWords(){
    const status=document.getElementById('targetWordsStatus');
    const container=document.getElementById('targetWordsResults');
    status.replaceChildren();
    container.replaceChildren();
    addText(status,'p','Loading target words...');

    try{
        const response=await fetch('/api/target-words');
        const data=await response.json();
        if(!response.ok)throw new Error(data.error||'Could not load target words');

        status.replaceChildren();
        addText(status,'p','Unique target words: '+data.count,'notice success');

        const list=document.createElement('div');
        list.className='metrics';
        for(const word of data.target_words){
            const card=document.createElement('div');
            card.className='metric';
            addText(card,'div',word,'value');
            list.appendChild(card);
        }
        container.appendChild(list);
        window.targetWordsLoaded=true;
    }catch(error){
        status.replaceChildren();
        addText(status,'p','Could not load target words: '+error.message,'notice error');
    }
}

async function loadCompleteDataset(){
    const status=document.getElementById('datasetStatus');
    const container=document.getElementById('datasetResults');
    status.replaceChildren();
    container.replaceChildren();
    addText(status,'p','Loading complete dataset...');

    try{
        const response=await fetch('/api/dataset');
        const data=await response.json();
        if(!response.ok)throw new Error(data.error||'Could not load dataset');

        status.replaceChildren();
        addText(status,'p','Rows: '+data.rows.length+' | Columns: '+data.columns.length,'notice success');

        const search=document.createElement('input');
        search.placeholder='Filter any column...';
        search.setAttribute('aria-label','Filter complete dataset');
        container.appendChild(search);

        const wrap=document.createElement('div');
        wrap.className='table-wrap';
        const table=document.createElement('table');
        const thead=document.createElement('thead');
        const head=document.createElement('tr');

        for(const col of data.columns)addText(head,'th',col);
        thead.appendChild(head);
        table.appendChild(thead);

        const tbody=document.createElement('tbody');
        table.appendChild(tbody);
        wrap.appendChild(table);
        container.appendChild(wrap);

        function renderDataset(filter=''){
            tbody.replaceChildren();
            const q=filter.toLowerCase();
            const filtered=data.rows.filter(row=>
                data.columns.some(col=>
                    String(row[col]??'').toLowerCase().includes(q)
                )
            );

            for(const row of filtered){
                const tr=document.createElement('tr');
                for(const col of data.columns){
                    const td=document.createElement('td');
                    td.textContent=row[col]??'';
                    tr.appendChild(td);
                }
                tbody.appendChild(tr);
            }

            if(!filtered.length){
                const tr=document.createElement('tr');
                const td=document.createElement('td');
                td.colSpan=Math.max(data.columns.length,1);
                td.textContent='No matching rows.';
                tr.appendChild(td);
                tbody.appendChild(tr);
            }
        }

        renderDataset();
        search.addEventListener('input',()=>renderDataset(search.value));

        const download=document.createElement('button');
        download.type='button';
        download.textContent='Download annotated.csv';
        download.addEventListener('click',()=>{
            const esc=v=>'"'+String(v??'').replace(/"/g,'""')+'"';
            const csv='\uFEFF'+[
                data.columns,
                ...data.rows.map(row=>data.columns.map(col=>row[col]??''))
            ].map(row=>row.map(esc).join(',')).join('\r\n');

            const blob=new Blob([csv],{type:'text/csv;charset=utf-8;'});
            const url=URL.createObjectURL(blob);
            const link=document.createElement('a');
            link.href=url;
            link.download='annotated.csv';
            document.body.appendChild(link);
            link.click();
            link.remove();
            URL.revokeObjectURL(url);
        });
        container.appendChild(download);
        window.completeDatasetLoaded=true;
    }catch(error){
        status.replaceChildren();
        addText(status,'p','Could not load complete dataset: '+error.message,'notice error');
    }
}

function downloadResultsCSV(results){
    const headers=[
        'No.','Sentence','Target','Gold synset','Custom prediction',
        'Custom score','Custom result','NLTK prediction','NLTK result'
    ];
    const esc=v=>'"'+String(v==null?'':v).replace(/"/g,'""')+'"';
    const rows=results.map(r=>[
        r.index,r.sentence,r.target_word,r.gold_synset,
        r.custom_prediction||'No sense found',r.custom_score??'',
        r.custom_correct?'CORRECT':'INCORRECT',
        r.nltk_prediction||'No sense found',
        r.nltk_correct?'CORRECT':'INCORRECT'
    ]);
    const csv='\uFEFF'+[headers,...rows].map(row=>row.map(esc).join(',')).join('\r\n');
    const blob=new Blob([csv],{type:'text/csv;charset=utf-8;'});
    const url=URL.createObjectURL(blob);
    const link=document.createElement('a');
    link.href=url;
    link.download='wsd_evaluation_results.csv';
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(url);
}

document.getElementById('evaluateButton').addEventListener('click',async()=>{
    const button=document.getElementById('evaluateButton');
    const status=document.getElementById('evaluationStatus');
    const container=document.getElementById('evaluationResults');
    button.disabled=true;
    status.replaceChildren();
    container.replaceChildren();
    addText(status,'p','Evaluating dataset. Please wait...');

    try{
        const data=await postJSON('/api/evaluate',{});
        status.replaceChildren();

        if(data.failed>0){
            addText(status,'p','Completed with '+data.failed+
                ' failed row(s). Accuracy uses '+data.metrics.total+
                ' successfully evaluated rows.','notice');
        }else{
            addText(status,'p','Evaluation completed successfully.','notice success');
        }

        const m=data.metrics;
        const metrics=document.createElement('div');
        metrics.className='metrics';

        for(const [label,value] of [
            ['Dataset rows',data.dataset_rows],
            ['Evaluated rows',m.total],
            ['Custom correct',m.custom_correct],
            ['NLTK correct',m.nltk_correct]
        ]){
            const box=document.createElement('div');
            box.className='metric';
            addText(box,'div',label,'label');
            addText(box,'div',value,'value');
            metrics.appendChild(box);
        }
        container.appendChild(metrics);

        const summary=document.createElement('div');
        summary.className='result';
        addText(summary,'h3','Accuracy comparison');

        for(const [label,accuracy,isNltk] of [
            ['Custom Simplified Lesk',m.custom_accuracy,false],
            ['NLTK Lesk',m.nltk_accuracy,true]
        ]){
            const row=document.createElement('div');
            row.className='bar-row';
            addText(row,'div',label);
            const track=document.createElement('div');
            track.className='bar-track';
            const bar=document.createElement('div');
            bar.className='bar'+(isNltk?' nltk':'');
            bar.style.width=Math.max(0,Math.min(100,accuracy))+'%';
            track.appendChild(bar);
            row.appendChild(track);
            addText(row,'strong',accuracy.toFixed(2)+'%');
            summary.appendChild(row);
        }

        addText(summary,'p','Accuracy difference (Custom − NLTK): '+
            (m.custom_accuracy-m.nltk_accuracy>=0?'+':'')+
            (m.custom_accuracy-m.nltk_accuracy).toFixed(2)+'%');
        container.appendChild(summary);

        const wrap=document.createElement('div');
        wrap.className='table-wrap';
        const table=document.createElement('table');
        table.style.minWidth='650px';
        const thead=document.createElement('thead');
        const head=document.createElement('tr');

        for(const h of ['Model','Accuracy','Macro precision','Macro recall','Macro F1-score']){
            addText(head,'th',h);
        }
        thead.appendChild(head);
        table.appendChild(thead);

        const body=document.createElement('tbody');
        for(const [name,prefix] of [
            ['Custom Simplified Lesk','custom'],
            ['NLTK Lesk','nltk']
        ]){
            const tr=document.createElement('tr');
            addText(tr,'td',name);
            for(const value of [
                m[prefix+'_accuracy'],
                m[prefix+'_macro_precision'],
                m[prefix+'_macro_recall'],
                m[prefix+'_macro_f1']
            ]){
                addText(tr,'td',Number(value).toFixed(2)+'%');
            }
            body.appendChild(tr);
        }

        table.appendChild(body);
        wrap.appendChild(table);
        container.appendChild(wrap);
        addText(container,'p',
            'Precision, recall, and F1-score are macro-averaged across the synset labels (one-vs-rest).',
            'small');

        const download=document.createElement('button');
        download.type='button';
        download.textContent='Download final CSV';
        download.addEventListener('click',()=>downloadResultsCSV(data.results));
        container.appendChild(download);

        addText(container,'h3','Detailed test results');
        const search=document.createElement('input');
        search.placeholder='Filter by sentence, target word, or synset...';
        search.setAttribute('aria-label','Filter evaluation results');
        container.appendChild(search);

        const tableWrap=document.createElement('div');
        tableWrap.className='table-wrap';
        const resultTable=document.createElement('table');
        const resultHead=document.createElement('thead');
        const resultHeader=document.createElement('tr');
        const columns=[
            'No.','Sentence','Target','Gold synset','Custom prediction',
            'Custom score','Custom result','NLTK prediction','NLTK result'
        ];
        for(const col of columns)addText(resultHeader,'th',col);
        resultHead.appendChild(resultHeader);
        resultTable.appendChild(resultHead);

        const tbody=document.createElement('tbody');
        resultTable.appendChild(tbody);
        tableWrap.appendChild(resultTable);
        container.appendChild(tableWrap);

        function renderRows(filter=''){
            tbody.replaceChildren();
            const q=filter.toLowerCase();
            const filtered=data.results.filter(r=>
                [r.sentence,r.target_word,r.gold_synset,r.custom_prediction,r.nltk_prediction]
                .join(' ').toLowerCase().includes(q)
            );

            for(const r of filtered){
                const tr=document.createElement('tr');
                const values=[
                    r.index,r.sentence,r.target_word,r.gold_synset,
                    r.custom_prediction||'No sense found',r.custom_score??'',
                    r.custom_correct?'CORRECT':'INCORRECT',
                    r.nltk_prediction||'No sense found',
                    r.nltk_correct?'CORRECT':'INCORRECT'
                ];

                values.forEach((value,i)=>{
                    const td=document.createElement('td');
                    td.textContent=value==null?'':String(value);
                    if(i===6||i===8){
                        td.className=value==='CORRECT'?'status-correct':'status-wrong';
                    }
                    tr.appendChild(td);
                });
                tbody.appendChild(tr);
            }

            if(!filtered.length){
                const tr=document.createElement('tr');
                const td=document.createElement('td');
                td.colSpan=columns.length;
                td.textContent='No matching rows.';
                tr.appendChild(td);
                tbody.appendChild(tr);
            }
        }

        renderRows();
        search.addEventListener('input',()=>renderRows(search.value));

        if(data.errors&&data.errors.length){
            const details=document.createElement('details');
            addText(details,'summary','View failed rows ('+data.errors.length+')');
            const list=document.createElement('ul');
            for(const item of data.errors){
                addText(list,'li','Row '+item.index+': '+item.error);
            }
            details.appendChild(list);
            container.appendChild(details);
        }
    }catch(error){
        status.replaceChildren();
        addText(status,'p','Evaluation failed: '+error.message,'notice error');
    }finally{
        button.disabled=false;
    }
});
</script>
</body>
</html>
"""


def load_csv_rows(file_path):
    """Read every column and row from the CSV file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError("Dataset not found: " + file_path)

    with open(file_path, "r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames:
            raise ValueError("CSV dataset is empty or has no header row")

        columns = [str(column) for column in reader.fieldnames]
        rows = [
            {column: (row.get(column) or "") for column in columns}
            for row in reader
        ]

    return {"columns": columns, "rows": rows}


def load_annotated_data(file_path):
    """Load the fields needed for WSD evaluation."""
    if not os.path.exists(file_path):
        raise FileNotFoundError("Dataset not found: " + file_path)

    data = []
    with open(file_path, "r", encoding="utf-8-sig", newline="") as file:
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

            if sentence and target and gold:
                data.append({
                    "sentence": sentence,
                    "target_word": target,
                    "gold_synset": gold
                })

    return data


def get_synset_name(synset):
    if synset is None:
        return None
    return synset if isinstance(synset, str) else synset.name()


def evaluate_sentence(sentence, target_word, gold_synset_name, window_size):
    try:
        gold_synset = wn.synset(str(gold_synset_name).strip())
    except Exception as error:
        raise ValueError(
            "Invalid WordNet gold synset: " + str(gold_synset_name)
        ) from error

    custom = SimplifiedLesk(window_size=window_size).disambiguate(
        sentence, target_word
    )
    nltk_result = NLTKLeskModel().disambiguate(sentence, target_word)

    custom_synset = custom.get("predicted_synset")
    nltk_synset = nltk_result.get("predicted_synset")

    custom_name = get_synset_name(
        custom.get("predicted_name") or custom_synset
    )
    nltk_name = get_synset_name(nltk_synset)
    gold_name = gold_synset.name()

    return {
        "sentence": sentence,
        "target_word": target_word,
        "gold_synset": gold_name,
        "custom_prediction": custom_name,
        "custom_score": custom.get("score", 0),
        "custom_correct": custom_name == gold_name,
        "nltk_prediction": nltk_name,
        "nltk_correct": nltk_name == gold_name
    }


def calculate_metrics(results):
    total = len(results)
    custom_correct = sum(bool(r["custom_correct"]) for r in results)
    nltk_correct = sum(bool(r["nltk_correct"]) for r in results)

    labels = sorted(
        {str(r["gold_synset"]) for r in results}
        | {
            str(r[key])
            for r in results
            for key in ("custom_prediction", "nltk_prediction")
            if r[key]
        }
    )

    def macro_scores(prediction_key):
        if not labels:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

        scores = []
        for label in labels:
            tp = sum(
                r["gold_synset"] == label and r[prediction_key] == label
                for r in results
            )
            fp = sum(
                r["gold_synset"] != label and r[prediction_key] == label
                for r in results
            )
            fn = sum(
                r["gold_synset"] == label and r[prediction_key] != label
                for r in results
            )

            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            f1 = (
                2 * precision * recall / (precision + recall)
                if precision + recall else 0.0
            )
            scores.append((precision, recall, f1))

        n = len(scores)
        return {
            key: sum(score[i] for score in scores) / n * 100
            for i, key in enumerate(("precision", "recall", "f1"))
        }

    custom_macro = macro_scores("custom_prediction")
    nltk_macro = macro_scores("nltk_prediction")

    return {
        "total": total,
        "custom_correct": custom_correct,
        "nltk_correct": nltk_correct,
        "custom_accuracy": custom_correct / total * 100 if total else 0.0,
        "nltk_accuracy": nltk_correct / total * 100 if total else 0.0,
        "custom_macro_precision": custom_macro["precision"],
        "custom_macro_recall": custom_macro["recall"],
        "custom_macro_f1": custom_macro["f1"],
        "nltk_macro_precision": nltk_macro["precision"],
        "nltk_macro_recall": nltk_macro["recall"],
        "nltk_macro_f1": nltk_macro["f1"]
    }


@app.get("/")
def home():
    return render_template_string(PAGE)


@app.get("/api/health")
def health():
    return jsonify(status="ok", message="Flask API is running")


@app.get("/api/target-words")
def target_words():
    """Return unique target words from annotated.csv."""
    try:
        dataset = load_csv_rows(DATA_FILE)
        targets = sorted(
            {
                str(row.get("target_word", "")).strip()
                for row in dataset["rows"]
                if str(row.get("target_word", "")).strip()
            },
            key=str.casefold
        )
        return jsonify({"count": len(targets), "target_words": targets})
    except Exception as error:
        app.logger.exception("Could not load target words")
        return jsonify(error="Could not load target words: " + str(error)), 500


@app.get("/api/dataset")
def complete_dataset():
    """Return all columns and rows from annotated.csv."""
    try:
        return jsonify(load_csv_rows(DATA_FILE))
    except Exception as error:
        app.logger.exception("Could not load complete dataset")
        return jsonify(error="Could not load complete dataset: " + str(error)), 500


@app.post("/api/predict")
def predict():
    data = request.get_json(silent=True) or {}
    sentence = str(data.get("sentence", "")).strip()
    target = str(data.get("target_word", "")).strip().lower()

    if not sentence or not target:
        return jsonify(error="Enter a sentence and target word."), 400

    if not re.search(r"\b" + re.escape(target) + r"\b", sentence, flags=re.IGNORECASE):
        return jsonify(error="Target word was not found in the sentence."), 400

    try:
        custom = SimplifiedLesk(window_size=DEFAULT_WINDOW_SIZE).disambiguate(
            sentence, target
        )
        nltk_result = NLTKLeskModel().disambiguate(sentence, target)

        custom_synset = custom.get("predicted_synset")
        nltk_synset = nltk_result.get("predicted_synset")

        return jsonify({
            "custom": {
                "predicted_synset": (
                    custom.get("predicted_name") or get_synset_name(custom_synset)
                ),
                "definition": (
                    custom_synset.definition()
                    if hasattr(custom_synset, "definition") else None
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
                "predicted_synset": get_synset_name(nltk_synset),
                "definition": (
                    nltk_synset.definition()
                    if hasattr(nltk_synset, "definition") else None
                )
            }
        })
    except LookupError as error:
        app.logger.exception("Missing NLTK resource")
        return jsonify(error="Missing NLTK resource: " + str(error)), 500
    except Exception:
        app.logger.exception("Prediction failed")
        return jsonify(error="Prediction failed. Check Vercel runtime logs."), 500


@app.post("/api/evaluate")
def evaluate():
    try:
        dataset = load_annotated_data(DATA_FILE)
    except Exception as error:
        app.logger.exception("Could not load annotated dataset")
        return jsonify(error="Could not load dataset: " + str(error)), 500

    results, errors = [], []

    for index, item in enumerate(dataset, start=1):
        try:
            result = evaluate_sentence(
                item["sentence"],
                item["target_word"],
                item["gold_synset"],
                DEFAULT_WINDOW_SIZE
            )

            try:
                result["custom_score"] = float(
                    result["custom_score"]
                    if result["custom_score"] is not None else 0.0
                )
            except (TypeError, ValueError):
                result["custom_score"] = 0.0

            result["index"] = index
            results.append(result)
        except Exception as error:
            app.logger.warning(
                "Evaluation failed for CSV row %s: %s", index, error
            )
            errors.append({
                "index": index,
                "sentence": item["sentence"],
                "error": str(error)
            })

    return jsonify({
        "dataset_rows": len(dataset),
        "evaluated": len(results),
        "failed": len(errors),
        "window_size": DEFAULT_WINDOW_SIZE,
        "metrics": calculate_metrics(results),
        "results": results,
        "errors": errors
    })


if __name__ == "__main__":
    app.run(debug=False)
