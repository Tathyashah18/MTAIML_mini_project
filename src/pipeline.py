import os
import csv
import sys
# nltk.download("wordnet", quiet=True)
# nltk.download("omw-1.4", quiet=True)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from nltk.corpus import wordnet as wn
from models import SimplifiedLesk, NLTKLeskModel

DATA_FILE = os.path.join(PROJECT_ROOT, "data", "annotated.csv")
OUTPUT_FILE = os.path.join(
    PROJECT_ROOT, "data", "annotated", "evaluation_results.csv"
)
DEFAULT_WINDOW_SIZE = 5



def load_annotated_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found:\n{file_path}")

    data = []
    with open(
        file_path, "r", encoding="utf-8-sig", newline=""
    ) as file:
        reader = csv.DictReader(file)
        required_columns = {"sentence", "target_word", "gold_synset"}

        if not required_columns.issubset(reader.fieldnames):
            raise ValueError(
                "CSV must contain columns: sentence, target_word, gold_synset"
            )

        for row in reader:
            sentence = row["sentence"].strip()
            target_word = row["target_word"].strip()
            gold_synset = row["gold_synset"].strip()

            if not sentence or not target_word or not gold_synset:
                continue

            data.append(
                {
                    "sentence": sentence,
                    "target_word": target_word,
                    "gold_synset": gold_synset,
                }
            )

    return data


def get_synset_name(synset):
    if synset is None:
        return None
    if isinstance(synset, str):
        return synset
    return synset.name()


def get_gold_synset(gold_synset_name):
    if not gold_synset_name:
        return None
    try:
        return wn.synset(gold_synset_name)
    except Exception:
        return None


def evaluate_sentence(
    sentence, target_word, gold_synset_name, window_size=DEFAULT_WINDOW_SIZE
):
    custom_model = SimplifiedLesk(window_size=window_size)
    nltk_model = NLTKLeskModel()

    gold_synset = get_gold_synset(gold_synset_name)
    if gold_synset is None:
        raise ValueError(f"Invalid WordNet synset: {gold_synset_name}")

    custom_result = custom_model.disambiguate(sentence, target_word)
    custom_synset = custom_result.get("predicted_synset")
    custom_sense = get_synset_name(custom_synset)

    nltk_result = nltk_model.disambiguate(sentence, target_word)
    nltk_synset = nltk_result.get("predicted_synset")
    nltk_sense = get_synset_name(nltk_synset)

    gold_sense = gold_synset.name()
    custom_correct = custom_sense == gold_sense
    nltk_correct = nltk_sense == gold_sense

    return {
        "sentence": sentence,
        "target_word": target_word,
        "gold_synset": gold_sense,
        "custom_prediction": custom_sense,
        "custom_score": custom_result.get("score", 0),
        "custom_overlap": ", ".join(custom_result.get("overlap_words", [])),
        "custom_correct": custom_correct,
        "nltk_prediction": nltk_sense,
        "nltk_correct": nltk_correct,
    }


def evaluate_dataset(data, window_size=DEFAULT_WINDOW_SIZE):
    results = []

    for index, item in enumerate(data, start=1):
        sentence = item["sentence"]
        target_word = item["target_word"]
        gold_synset = item["gold_synset"]

        try:
            result = evaluate_sentence(
                sentence, target_word, gold_synset, window_size
            )
            results.append(result)

            print(
                f"[{index:02d}/{len(data):02d}] "
                f"{target_word:<10} "
                f"Gold: {result['gold_synset']:<15} "
                f"Custom: {str(result['custom_prediction']):<15} "
                f"NLTK: {str(result['nltk_prediction']):<15}"
            )
        except Exception as error:
            print()
            print(f"Error processing: {sentence}")
            print(f"Reason: {error}")

    return results


def calculate_metrics(results):
    total = len(results)

    if total == 0:
        return {
            "total": 0,
            "custom_correct": 0,
            "nltk_correct": 0,
            "custom_accuracy": 0.0,
            "nltk_accuracy": 0.0,
        }

    custom_correct = sum(1 for result in results if result["custom_correct"])
    nltk_correct = sum(1 for result in results if result["nltk_correct"])

    custom_accuracy = (custom_correct / total) * 100
    nltk_accuracy = (nltk_correct / total) * 100

    return {
        "total": total,
        "custom_correct": custom_correct,
        "nltk_correct": nltk_correct,
        "custom_accuracy": custom_accuracy,
        "nltk_accuracy": nltk_accuracy,
    }


def display_results(results):
    print()
    print("=" * 90)
    print("DETAILED EVALUATION RESULTS")
    print("=" * 90)

    for index, result in enumerate(results, start=1):
        custom_status = "PASS" if result["custom_correct"] else "FAIL"
        nltk_status = "PASS" if result["nltk_correct"] else "FAIL"

        print()
        print(f"Test Case {index}")
        print(f"Sentence : {result['sentence']}")
        print(f"Target   : {result['target_word']}")
        print(f"Gold     : {result['gold_synset']}")
        print(f"Custom   : {result['custom_prediction']} [{custom_status}]")
        print(f"Score    : {result['custom_score']}")
        print(f"Overlap  : {result['custom_overlap']}")
        print(f"NLTK     : {result['nltk_prediction']} [{nltk_status}]")


def display_summary(metrics):
    print()
    print("=" * 90)
    print("EVALUATION SUMMARY")
    print("=" * 90)

    print()
    print(f"Total test cases       : {metrics['total']}")
    print(f"Custom Lesk correct    : {metrics['custom_correct']}")
    print(f"Custom Lesk accuracy   : {metrics['custom_accuracy']:.2f}%")

    print()
    print(f"NLTK Lesk correct      : {metrics['nltk_correct']}")
    print(f"NLTK Lesk accuracy     : {metrics['nltk_accuracy']:.2f}%")

    print()
    difference = metrics["custom_accuracy"] - metrics["nltk_accuracy"]
    print(f"Accuracy difference    : {difference:+.2f}%")
    print("=" * 90)


def save_results(results, output_file):
    if not results:
        return

    output_directory = os.path.dirname(output_file)
    os.makedirs(output_directory, exist_ok=True)

    fieldnames = [
        "sentence",
        "target_word",
        "gold_synset",
        "custom_prediction",
        "custom_score",
        "custom_overlap",
        "custom_correct",
        "nltk_prediction",
        "nltk_correct",
    ]

    with open(output_file, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def main():
    print("=" * 90)
    print("SIMPLIFIED LESK WORD SENSE DISAMBIGUATION")
    print("Closed-Caption Disambiguation System")
    print("=" * 90)

    print()
    print("Loading annotated dataset...")

    try:
        data = load_annotated_data(DATA_FILE)
        print(f"Loaded {len(data)} test cases.")
    except Exception as error:
        print()
        print("Failed to load dataset.")
        print(f"Error: {error}")
        return

    print()
    print(f"Context Window Size: {DEFAULT_WINDOW_SIZE}")

    print()
    print("Running evaluation...")
    print()

    results = evaluate_dataset(data, DEFAULT_WINDOW_SIZE)

    if not results:
        print()
        print("No results were generated.")
        return

    metrics = calculate_metrics(results)
    display_results(results)
    display_summary(metrics)
    save_results(results, OUTPUT_FILE)

    print()
    print("Evaluation results saved to:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()