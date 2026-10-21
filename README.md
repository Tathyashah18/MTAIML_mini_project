# Simplified Lesk Algorithm for Closed-Caption Disambiguation

## 1. Project Overview

Closed-captioning systems convert spoken language into text that can be translated into other languages. A major challenge is **word-sense ambiguity**, where the same word can have multiple meanings depending on its context.

For example:

> "He deposited money in the bank."

Here, **bank** refers to a financial institution.

Whereas:

> "The fisherman sat near the bank of the river."

Here, **bank** refers to the land beside a river.

This project implements a **Simplified Lesk Word Sense Disambiguation (WSD)** system from scratch. The system identifies the most appropriate WordNet sense of a target polysemous word by comparing its surrounding context with the definitions, example sentences, and synonyms associated with candidate WordNet synsets.

The implementation is designed as a lightweight and transparent baseline suitable for **real-time closed-caption disambiguation and downstream translation systems**.

---

## 2. Objectives

The main objectives of this project are:

* Implement the Simplified Lesk algorithm from scratch.
* Identify the correct sense of a polysemous target word.
* Use a configurable context window around the target word.
* Preprocess text using tokenization, stop-word removal, and lemmatization.
* Use Part-of-Speech (POS) information to reduce irrelevant WordNet senses.
* Compare context words with WordNet definitions and examples.
* Incorporate synonym matching.
* Use controlled phrase matching for additional contextual evidence.
* Compare the custom implementation with NLTK's built-in `lesk()` algorithm.
* Evaluate both approaches using an annotated test dataset.
* Provide a transparent and interpretable WSD baseline.

---

## 3. Problem Statement

A word can have multiple meanings, and selecting the correct meaning is essential for accurate language translation.

For example, the word **"bat"** may refer to:

* An animal
* A piece of sports equipment

Consider:

> "The bat flew out of the cave."

and:

> "The player swung the bat."

Although the target word is identical, the surrounding context determines its intended meaning.

The objective of this project is therefore:

> **Given a sentence and a target polysemous word, determine the most appropriate WordNet synset using contextual overlap and lexical information.**

---

## 4. Methodology

The overall workflow is:

```text
Input Sentence
       |
       v
Text Preprocessing
       |
       v
Target Word Identification
       |
       v
Context Window Extraction
       |
       v
POS Detection
       |
       v
WordNet Candidate Synsets
       |
       v
Sense Signature Generation
       |
       +-----------------------------+
       |                             |
       v                             v
Definition                     Examples
       |                             |
       +-------------+---------------+
                     |
                     v
                Synonyms
                     |
                     v
             Context Matching
                     |
                     v
          Weighted Lesk Scoring
                     |
                     v
             Candidate Ranking
                     |
                     v
             Predicted Synset
```

---

## 5. Algorithm

The custom implementation follows a controlled version of the Simplified Lesk algorithm.

For a target word `w`, candidate WordNet senses are obtained:

```text
S = {s1, s2, ..., sn}
```

For each candidate sense, a signature is created from:

* WordNet definition
* Example sentences
* Synonyms

The context of the target word is represented as:

```text
C = {context tokens}
```

The overlap between the context and each sense signature is then calculated.

The basic scoring function is:

```text
Score(s) =
    Definition Weight × Definition Overlap
  + Example Weight × Example Overlap
  + Synonym Weight × Synonym Overlap
  + Phrase Weight × Phrase Match
  + Semantic Weight × Semantic Similarity
```

The candidate with the highest score is selected as the predicted sense.

---

## 6. Main Features

### 6.1 Context Window

Only a limited number of words surrounding the target are considered.

The current implementation uses:

```text
window_size = 7
```

This reduces irrelevant information from distant parts of the sentence.

---

### 6.2 Text Preprocessing

The preprocessing stage includes:

* Text cleaning
* Sentence processing
* Tokenization
* Stop-word removal
* Lemmatization
* POS tagging

This converts raw text into normalized tokens suitable for WordNet comparison.

---

### 6.3 POS Filtering

WordNet contains senses belonging to different grammatical categories:

```text
n = noun
v = verb
a = adjective
r = adverb
```

The target word's POS is detected from the sentence and used to restrict candidate senses where possible.

This prevents unrelated noun, verb, adjective, and adverb senses from competing with each other.

---

### 6.4 Definition Matching

The definition of every candidate WordNet synset is processed and compared with the context.

Example:

```text
Context:
deposit money bank

Definition:
a financial institution that accepts deposits
```

Words such as `deposit` and `money` provide evidence for the financial meaning.

---

### 6.5 Example Sentence Matching

WordNet also provides example sentences for some senses.

Example sentences can provide stronger contextual evidence than a short definition.

Therefore, example overlap receives a higher weight than ordinary definition overlap.

---

### 6.6 Synonym Matching

The lemma names associated with each WordNet synset are also included in the sense signature.

This allows the algorithm to detect contextual relationships where the context uses a synonym of the target sense.

---

### 6.7 Phrase Matching

The system additionally considers two-word and three-word sequences.

For example:

```text
river bank
bank account
rock music
bass guitar
application form
```

Phrase-level evidence can provide stronger contextual information than isolated words.

---

### 6.8 Controlled Semantic Similarity

A small WordNet semantic similarity component is used as a secondary signal.

The implementation uses WordNet semantic similarity only as a **small bonus** rather than expanding every synset with large numbers of hypernyms and hyponyms.

This prevents semantic expansion from dominating the original Lesk overlap score.

---

### 6.9 Target-Word Leakage Protection

The target word is removed from the contextual representation.

This prevents the target itself from artificially increasing the overlap score.

For example:

```text
She planted a small plant in the garden.
```

The second occurrence of `plant` should not automatically cause the plant-related sense to receive an unfair score.

---

## 7. Project Structure

The project follows the required structure:

```text
project-folder/
│
├── docs/
│   ├── Architecture_and_Methodology.pdf
│   └── Evaluation_Report.pdf
│
├── src/
│   ├── preprocessing.py
│   ├── models.py
│   └── pipeline.py
│
├── data/
│   ├── raw_transcripts/
│   └── annotated/
│       ├── test_data.csv
│       └── evaluation_results.csv
│
├── requirements.txt
│
└── README.md
```

### File Description

| File                               | Purpose                                                                        |
| ---------------------------------- | ------------------------------------------------------------------------------ |
| `preprocessing.py`                 | Text cleaning, tokenization, POS tagging, stop-word removal, and lemmatization |
| `models.py`                        | Custom Simplified Lesk and NLTK Lesk implementations                           |
| `pipeline.py`                      | Dataset loading, evaluation, metric calculation, and result generation         |
| `test_data.csv`                    | Annotated evaluation dataset                                                   |
| `evaluation_results.csv`           | Predictions and evaluation results                                             |
| `requirements.txt`                 | Python dependencies                                                            |
| `Architecture_and_Methodology.pdf` | Technical architecture and methodology                                         |
| `Evaluation_Report.pdf`            | Experimental evaluation and analysis                                           |

---

## 8. Dataset Format

The evaluation dataset is stored at:

```text
data/annotated/test_data.csv
```

The required columns are:

```text
sentence,target_word,gold_synset
```

Example:

```csv
sentence,target_word,gold_synset
He went to the bank to deposit his money.,bank,bank.n.01
The fisherman sat on the bank of the river.,bank,bank.n.02
The player swung the bat and hit the ball.,bat,bat.n.01
The bat flew out of the cave.,bat,bat.n.02
```

The `gold_synset` column contains WordNet synset identifiers.

Example:

```text
bank.n.01
bank.n.02
bat.n.01
bat.n.02
```

---

## 9. Requirements

The project requires:

* Python 3.x
* NLTK
* WordNet
* WordNet Open Multilingual WordNet data
* POS tagger resources

Install the Python dependency:

```bash
pip install nltk
```

The required NLTK resources are automatically downloaded by the preprocessing module.

The resources include:

```text
punkt
punkt_tab
stopwords
wordnet
omw-1.4
averaged_perceptron_tagger
averaged_perceptron_tagger_eng
```

---

## 10. Installation

### Step 1: Clone or copy the project

Place the project in your desired directory.

### Step 2: Open the project directory

```bash
cd project-folder
```

### Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

If required:

```bash
pip install nltk
```

### Step 4: Run the preprocessing/model test

```bash
cd src
python models.py
```

This runs several sample sentences and displays the predicted WordNet senses.

---

## 11. Running the Evaluation

From the `src` directory:

```bash
python pipeline.py
```

The pipeline:

1. Loads the annotated dataset.
2. Reads each sentence and target word.
3. Retrieves the Gold WordNet synset.
4. Runs the custom Simplified Lesk model.
5. Runs NLTK's built-in Lesk model.
6. Compares predictions with the Gold synset.
7. Calculates accuracy.
8. Displays evaluation results.
9. Saves the results to:

```text
data/annotated/evaluation_results.csv
```

---

## 12. Evaluation Metrics

The primary metric is **accuracy**.

```text
Accuracy =
Correct Predictions / Total Valid Test Cases × 100
```

For example:

```text
Correct Predictions = 47
Valid Test Cases    = 98

Accuracy = 47 / 98 × 100
         = 47.96%
```

The system also reports the accuracy difference between the custom implementation and the NLTK baseline.

```text
Accuracy Difference =
Custom Accuracy - NLTK Accuracy
```

---

## 13. Model Comparison

The project compares two approaches.

### Custom Simplified Lesk

The custom implementation uses:

```text
POS filtering
      +
Context window
      +
Definition overlap
      +
Example overlap
      +
Synonym overlap
      +
Phrase matching
      +
Controlled semantic similarity
```

### NLTK Lesk

The baseline uses:

```python
from nltk.wsd import lesk
```

NLTK's implementation is treated as the baseline for comparison.

---

## 14. Experimental Results

An earlier evaluation of the dataset produced the following results:

| Model                  | Correct |   Accuracy |
| ---------------------- | ------: | ---------: |
| Custom Simplified Lesk | 47 / 98 | **47.96%** |
| NLTK Lesk              | 15 / 98 | **15.31%** |

Accuracy difference:

```text
47.96% - 15.31%
= +32.65 percentage points
```

> **Note:** The evaluation dataset currently contains 98 valid cases because some Gold synset labels require verification. Results should be reported using the same verified test set when comparing different model versions.

---

## 15. Example

### Input

```text
Sentence:
The fisherman sat on the bank of the river.

Target:
bank
```

### Possible WordNet Senses

```text
bank.n.01
bank.n.02
...
```

### Context

```text
fisherman
sat
river
```

The context contains strong evidence associated with the geographical meaning of `bank`.

The system therefore selects the corresponding WordNet synset.

---

## 16. Advantages

### Lightweight

The system does not require:

* Large language models
* GPU acceleration
* Model training
* Large neural networks

### Interpretable

The system can show why a sense was selected:

```text
Definition overlap
Example overlap
Synonym overlap
Phrase overlap
Semantic score
```

### Fast

Because the system uses a small context window and WordNet resources, it is suitable for lightweight WSD applications.

### Modular

The implementation separates:

```text
Preprocessing
       ↓
WSD Model
       ↓
Evaluation Pipeline
```

This makes future improvements easier.

---

## 17. Limitations

The current approach is a lexical WSD baseline and has several limitations.

### Exact Word Matching

Words with related meanings may not have identical lexical forms.

For example:

```text
music
instrument
guitar
```

may provide evidence for the same sense even when the exact words do not appear in a WordNet definition.

### WordNet Coverage

The system depends on the senses and relationships available in WordNet.

### Ambiguous Context

Very short sentences may not contain enough information to identify the intended sense.

### Gold Annotation Quality

The evaluation depends on the correctness of the annotated Gold synset.

Invalid or inconsistent WordNet sense IDs can affect the evaluation.

### Limited Semantic Understanding

The system does not perform deep contextual reasoning like transformer-based language models.

---

## 18. Future Scope

The system can be extended in several directions.

### Context Embeddings

Use contextual embeddings from models such as:

```text
BERT
RoBERTa
DistilBERT
```

to capture semantic relationships beyond exact lexical overlap.

### Hybrid WSD

Combine:

```text
Lesk score
+
WordNet similarity
+
Contextual embedding similarity
```

to create a stronger hybrid model.

### Real-Time Caption Processing

Integrate the WSD system into a live closed-caption pipeline:

```text
Speech
  ↓
Speech-to-Text
  ↓
Caption Text
  ↓
Target Word Detection
  ↓
WSD
  ↓
Correct Word Sense
  ↓
Translation
```

### Multilingual Translation

Map the selected WordNet synset to the appropriate translation equivalent in the target language.

### Larger Benchmark Datasets

Future evaluation can use established Word Sense Disambiguation datasets such as:

* SemCor
* Senseval
* SemEval WSD datasets
* Unified WSD Evaluation Framework datasets

### Confidence Scores

The model can provide a confidence value based on the difference between the top two candidate senses.

---

## 19. Applications

Potential applications include:

* Closed-captioning systems
* Machine translation
* Subtitle translation
* Text understanding
* Search engines
* Information retrieval
* Voice assistants
* Educational language tools
* Context-aware dictionaries
* Natural Language Processing systems

---

## 20. Conclusion

This project implements a lightweight and interpretable **Simplified Lesk Word Sense Disambiguation system** for closed-caption disambiguation.

The system uses contextual information together with WordNet definitions, examples, synonyms, POS information, and controlled phrase/semantic matching to select the most appropriate sense of an ambiguous word.

The custom implementation is evaluated against NLTK's built-in Lesk algorithm using an annotated dataset. The experimental results demonstrate that a customized Lesk implementation can provide substantially better performance than the basic NLTK baseline on the current test set.

The project also provides a foundation for future improvements using contextual embeddings, multilingual WordNet resources, and hybrid neural-symbolic WSD approaches.

```

This README is written to match your **current project implementation**, including the custom model, NLTK baseline, dataset format, evaluation workflow, limitations, and future scope.
```
