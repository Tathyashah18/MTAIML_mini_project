
import os
import re
from pathlib import Path

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize



# ---------------------------------------------------------
# NLTK DATA CONFIGURATION
# ---------------------------------------------------------

# Project root is the directory containing app.py.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Prefer the project's bundled NLTK data.
# Also support src/nltk_data for compatibility with a previous setup.
NLTK_DATA_DIRS = [
    PROJECT_ROOT / "nltk_data",
    Path(__file__).resolve().parent / "nltk_data",
]

# Add bundled data directories to NLTK's search path.
# Do not use /tmp or download resources during module import.
for data_dir in NLTK_DATA_DIRS:
    if data_dir.is_dir():
        nltk_data_path = str(data_dir)
        if nltk_data_path not in nltk.data.path:
            nltk.data.path.insert(0, nltk_data_path)


# NLTK resource paths and the corresponding downloader package names.
NLTK_RESOURCES = [
    ("tokenizers/punkt", "punkt"),
    ("tokenizers/punkt_tab", "punkt_tab"),
    ("corpora/stopwords", "stopwords"),
    ("corpora/wordnet", "wordnet"),
    ("corpora/omw-1.4", "omw-1.4"),
    (
        "taggers/averaged_perceptron_tagger",
        "averaged_perceptron_tagger",
    ),
    (
        "taggers/averaged_perceptron_tagger_eng",
        "averaged_perceptron_tagger_eng",
    ),
]


def check_nltk_resource(resource_path, resource_name):
    """
    Verify that an NLTK resource is available.

    This function does not download anything. Resources must
    already be installed locally or bundled with the deployment.
    """
    try:
        nltk.data.find(resource_path)
        return True

    except LookupError as error:
        searched_paths = "\n".join(
            f"  - {path}" for path in nltk.data.path
        )

        raise LookupError(
            f"Required NLTK resource '{resource_name}' is missing.\n"
            f"Expected resource: {resource_path}\n"
            f"NLTK searched these directories:\n{searched_paths}\n\n"
            "Install the resource locally and include it in your "
            "GitHub repository's nltk_data directory. "
            "Do not rely on downloading it during Vercel startup."
        ) from error


def download_nltk_resources():
    
    for resource_path, resource_name in NLTK_RESOURCES:
        check_nltk_resource(resource_path, resource_name)

    print("All required NLTK resources are available.")


# IMPORTANT:
# Do not call download_nltk_resources() at module import time.
# Vercel should use the NLTK resources bundled with the project.


# ---------------------------------------------------------
# LEMMATIZER
# ---------------------------------------------------------

lemmatizer = WordNetLemmatizer()



def clean_text(text):
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"[^a-z0-9\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def segment_sentences(text):
    if not isinstance(text, str):
        return []

    if not text.strip():
        return []

    check_nltk_resource("tokenizers/punkt", "punkt")
    check_nltk_resource("tokenizers/punkt_tab", "punkt_tab")

    return sent_tokenize(text)


def tokenize_text(text):
    if not isinstance(text, str):
        return []

    if not text.strip():
        return []

    check_nltk_resource("tokenizers/punkt", "punkt")
    check_nltk_resource("tokenizers/punkt_tab", "punkt_tab")

    return word_tokenize(text)


# ---------------------------------------------------------
# STOPWORD REMOVAL
# ---------------------------------------------------------

def remove_stopwords(tokens):
    check_nltk_resource("corpora/stopwords", "stopwords")

    stop_words = set(stopwords.words("english"))

    return [
        token
        for token in tokens
        if token.lower() not in stop_words
    ]


# ---------------------------------------------------------
# LEMMATIZATION
# ---------------------------------------------------------

def lemmatize_tokens(tokens):
    check_nltk_resource("corpora/wordnet", "wordnet")

    return [
        lemmatizer.lemmatize(token.lower())
        for token in tokens
    ]


def get_wordnet_pos(treebank_tag):
    if not treebank_tag:
        return None

    if treebank_tag.startswith("J"):
        return "a"

    if treebank_tag.startswith("V"):
        return "v"

    if treebank_tag.startswith("N"):
        return "n"

    if treebank_tag.startswith("R"):
        return "r"

    return None


def pos_tag_tokens(tokens):
    if not tokens:
        return []

    # Newer NLTK versions use the English-specific tagger.
    try:
        check_nltk_resource(
            "taggers/averaged_perceptron_tagger_eng",
            "averaged_perceptron_tagger_eng",
        )
    except LookupError:
        # Compatibility with older NLTK installations.
        check_nltk_resource(
            "taggers/averaged_perceptron_tagger",
            "averaged_perceptron_tagger",
        )

    return nltk.pos_tag(tokens)


def lemmatize_token_with_pos(token, pos=None):
    check_nltk_resource("corpora/wordnet", "wordnet")

    token = token.lower()

    if pos in {"n", "v", "a", "r"}:
        return lemmatizer.lemmatize(token, pos)

    return lemmatizer.lemmatize(token)


def lemmatize_tokens_with_pos(tagged_tokens):
    result = []

    for token, tag in tagged_tokens:
        wordnet_pos = get_wordnet_pos(tag)

        result.append(
            lemmatize_token_with_pos(token, wordnet_pos)
        )

    return result


# ---------------------------------------------------------
# GENERAL PREPROCESSING
# ---------------------------------------------------------

def preprocess_text(
    text,
    remove_stop_words=True,
    apply_lemmatization=True,
):
    text = clean_text(text)
    tokens = tokenize_text(text)

    if remove_stop_words:
        tokens = remove_stopwords(tokens)

    if apply_lemmatization:
        tokens = lemmatize_tokens(tokens)

    return tokens


# ---------------------------------------------------------
# TARGET WORD AND CONTEXT WINDOW
# ---------------------------------------------------------

def find_target_index(tokens, target_word):
    if not tokens or not target_word:
        return -1

    target_word = target_word.lower()

    for index, token in enumerate(tokens):
        if token.lower() == target_word:
            return index

    return -1


def get_context_window(
    tokens,
    target_word,
    window_size=7,
):
    if not tokens or not target_word:
        return []

    target_index = find_target_index(tokens, target_word)

    if target_index == -1:
        return []

    start = max(0, target_index - window_size)
    end = min(len(tokens), target_index + window_size + 1)

    return (
        tokens[start:target_index]
        + tokens[target_index + 1:end]
    )


def prepare_context(
    sentence,
    target_word,
    window_size=7,
    remove_stop_words=True,
    apply_lemmatization=True,
):
    if not sentence or not target_word:
        return []

    cleaned_sentence = clean_text(sentence)
    tokens = tokenize_text(cleaned_sentence)

    target_index = find_target_index(tokens, target_word)

    if target_index == -1:
        return []

    start = max(0, target_index - window_size)
    end = min(len(tokens), target_index + window_size + 1)

    context_tokens = (
        tokens[start:target_index]
        + tokens[target_index + 1:end]
    )

    if remove_stop_words:
        context_tokens = remove_stopwords(context_tokens)

    if apply_lemmatization:
        context_tokens = lemmatize_tokens(context_tokens)

    return context_tokens


# ---------------------------------------------------------
# POS-AWARE CONTEXT PREPARATION
# ---------------------------------------------------------

def prepare_context_with_pos(
    sentence,
    target_word,
    window_size=7,
    remove_stop_words=True,
):
    empty_result = {
        "tokens": [],
        "tagged_tokens": [],
        "target_pos": None,
    }

    if not sentence or not target_word:
        return empty_result

    cleaned_sentence = clean_text(sentence)
    tokens = tokenize_text(cleaned_sentence)

    target_index = find_target_index(tokens, target_word)

    if target_index == -1:
        return empty_result

    tagged_sentence = pos_tag_tokens(tokens)

    target_tag = tagged_sentence[target_index][1]
    target_pos = get_wordnet_pos(target_tag)

    start = max(0, target_index - window_size)
    end = min(len(tokens), target_index + window_size + 1)

    context_tagged = (
        tagged_sentence[start:target_index]
        + tagged_sentence[target_index + 1:end]
    )

    if remove_stop_words:
        stop_words = set(stopwords.words("english"))

        context_tagged = [
            (token, tag)
            for token, tag in context_tagged
            if token.lower() not in stop_words
        ]

    processed_tokens = lemmatize_tokens_with_pos(
        context_tagged
    )

    return {
        "tokens": processed_tokens,
        "tagged_tokens": context_tagged,
        "target_pos": target_pos,
    }


# ---------------------------------------------------------
# TOKEN NORMALIZATION
# ---------------------------------------------------------

def normalize_tokens(tokens):
    if not tokens:
        return []

    normalized = []

    for token in tokens:
        if not isinstance(token, str):
            continue

        token = token.lower().strip()

        if not re.match(r"^[a-z0-9]+$", token):
            continue

        normalized.append(token)

    return normalized


# ---------------------------------------------------------
# WORDNET SIGNATURE PREPROCESSING
# ---------------------------------------------------------

def preprocess_signature(signature):
    if not isinstance(signature, str):
        return []

    tokens = preprocess_text(
        signature,
        remove_stop_words=True,
        apply_lemmatization=True,
    )

    return normalize_tokens(tokens)


def preprocess_signature_with_pos(
    signature,
    pos=None,
):
    if not isinstance(signature, str):
        return []

    cleaned = clean_text(signature)
    tokens = tokenize_text(cleaned)

    stop_words = set(stopwords.words("english"))

    tokens = [
        token
        for token in tokens
        if token.lower() not in stop_words
    ]

    result = []

    for token in tokens:
        lemma = lemmatize_token_with_pos(token, pos)
        result.append(lemma)

    return normalize_tokens(result)
