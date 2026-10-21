from collections import Counter

from nltk.corpus import wordnet as wn
from nltk.wsd import lesk as nltk_lesk

from preprocessing import (
    prepare_context,
    prepare_context_with_pos,
    preprocess_signature,
    preprocess_signature_with_pos,
)


class SimplifiedLesk:
    """
    Improved Simplified Lesk Word Sense Disambiguation model.

    Improvements over the basic implementation:
        1. POS filtering
        2. Larger context window
        3. Definition overlap
        4. Example sentence overlap
        5. Synonym overlap
        6. Weighted scoring
        7. Frequency-aware tie breaking
        8. Better handling of zero-overlap cases
    """

    def __init__(
        self,
        window_size=7,
        use_pos_filter=True,
        use_synonyms=True,
        definition_weight=1.0,
        example_weight=2.0,
        synonym_weight=0.5
    ):
        self.window_size = window_size
        self.use_pos_filter = use_pos_filter
        self.use_synonyms = use_synonyms

        self.definition_weight = definition_weight
        self.example_weight = example_weight
        self.synonym_weight = synonym_weight

    # ---------------------------------------------------------
    # GET WORDNET SENSES
    # ---------------------------------------------------------

    def get_synsets(self, target_word, target_pos=None):
        """
        Return possible WordNet synsets for the target word.

        If POS is available, only matching POS senses
        are returned.
        """
        if not target_word:
            return []

        target_word = target_word.lower()

        if self.use_pos_filter and target_pos:
            synsets = wn.synsets(
                target_word,
                pos=target_pos
            )
        else:
            synsets = wn.synsets(target_word)

        return synsets

    # ---------------------------------------------------------
    # SIGNATURE
    # ---------------------------------------------------------

    def get_signature_parts(
        self,
        synset,
        target_pos=None
    ):
        """
        Build separate signature components for a synset.

        Returns:
            {
                "definition": [...],
                "examples": [...],
                "synonyms": [...]
            }
        """

        definition = synset.definition()

        definition_tokens = preprocess_signature_with_pos(
            definition,
            target_pos
        )

        # -----------------------------------------------------
        # EXAMPLES
        # -----------------------------------------------------

        example_text = " ".join(
            synset.examples()
        )

        example_tokens = preprocess_signature_with_pos(
            example_text,
            target_pos
        )

        # -----------------------------------------------------
        # SYNONYMS
        # -----------------------------------------------------

        synonym_tokens = []

        if self.use_synonyms:
            for lemma in synset.lemmas():

                lemma_name = lemma.name()

                # WordNet often uses "_" for multiword terms.
                lemma_name = lemma_name.replace("_", " ")

                synonym_tokens.extend(
                    preprocess_signature(
                        lemma_name
                    )
                )

        return {
            "definition": definition_tokens,
            "examples": example_tokens,
            "synonyms": synonym_tokens,
        }

    # ---------------------------------------------------------
    # OVERLAP
    # ---------------------------------------------------------

    def calculate_overlap(
        self,
        context_tokens,
        signature_tokens
    ):
        """
        Calculate simple set-based overlap.
        """
        context_set = set(context_tokens)
        signature_set = set(signature_tokens)

        return context_set.intersection(
            signature_set
        )

    # ---------------------------------------------------------
    # WEIGHTED SCORE
    # ---------------------------------------------------------

    def calculate_weighted_score(
        self,
        context_tokens,
        signature_parts
    ):
        """
        Calculate weighted Lesk score.

        Formula:

        Score =
            definition overlap * definition weight
          + example overlap * example weight
          + synonym overlap * synonym weight
        """

        context_set = set(context_tokens)

        definition_set = set(
            signature_parts["definition"]
        )

        example_set = set(
            signature_parts["examples"]
        )

        synonym_set = set(
            signature_parts["synonyms"]
        )

        definition_overlap = (
            context_set & definition_set
        )

        example_overlap = (
            context_set & example_set
        )

        synonym_overlap = (
            context_set & synonym_set
        )

        definition_score = (
            len(definition_overlap)
            * self.definition_weight
        )

        example_score = (
            len(example_overlap)
            * self.example_weight
        )

        synonym_score = (
            len(synonym_overlap)
            * self.synonym_weight
        )

        total_score = (
            definition_score
            + example_score
            + synonym_score
        )

        all_overlap = (
            definition_overlap
            | example_overlap
            | synonym_overlap
        )

        return {
            "score": total_score,
            "definition_score": definition_score,
            "example_score": example_score,
            "synonym_score": synonym_score,
            "definition_overlap": sorted(
                definition_overlap
            ),
            "example_overlap": sorted(
                example_overlap
            ),
            "synonym_overlap": sorted(
                synonym_overlap
            ),
            "overlap_words": sorted(
                all_overlap
            ),
        }

    # ---------------------------------------------------------
    # FREQUENCY
    # ---------------------------------------------------------

    def get_synset_frequency(
        self,
        synset,
        target_word
    ):
        """
        Estimate WordNet lemma frequency for the target word.

        Used only for tie-breaking, not as the main score.
        """
        target_word = target_word.lower()

        frequency = 0

        for lemma in synset.lemmas():

            lemma_name = (
                lemma.name()
                .lower()
                .replace("_", " ")
            )

            if lemma_name == target_word:
                frequency += lemma.count()

        return frequency

    # ---------------------------------------------------------
    # DISAMBIGUATION
    # ---------------------------------------------------------

    def disambiguate(
        self,
        sentence,
        target_word
    ):
        """
        Predict the most appropriate WordNet sense.
        """

        # -----------------------------------------------------
        # STEP 1: PREPARE CONTEXT + POS
        # -----------------------------------------------------

        context_info = prepare_context_with_pos(
            sentence,
            target_word,
            window_size=self.window_size,
            remove_stop_words=True
        )

        context_tokens = context_info["tokens"]
        target_pos = context_info["target_pos"]

        # -----------------------------------------------------
        # STEP 2: GET CANDIDATE SYNSETS
        # -----------------------------------------------------

        synsets = self.get_synsets(
            target_word,
            target_pos
        )

        # Fallback:
        # If POS filtering produces no candidates,
        # consider all WordNet senses.
        if not synsets:
            synsets = self.get_synsets(
                target_word,
                target_pos=None
            )

        if not synsets:
            return {
                "sentence": sentence,
                "target_word": target_word,
                "context": context_tokens,
                "target_pos": target_pos,
                "predicted_synset": None,
                "predicted_name": None,
                "score": 0,
                "overlap_words": [],
                "candidates": [],
            }

        # -----------------------------------------------------
        # STEP 3: SCORE EVERY SYNSET
        # -----------------------------------------------------

        candidates = []

        for synset in synsets:

            signature_parts = self.get_signature_parts(
                synset,
                target_pos
            )

            score_info = self.calculate_weighted_score(
                context_tokens,
                signature_parts
            )

            frequency = self.get_synset_frequency(
                synset,
                target_word
            )

            candidate = {
                "synset": synset,
                "name": synset.name(),
                "definition": synset.definition(),
                "score": score_info["score"],
                "definition_score": score_info[
                    "definition_score"
                ],
                "example_score": score_info[
                    "example_score"
                ],
                "synonym_score": score_info[
                    "synonym_score"
                ],
                "definition_overlap": score_info[
                    "definition_overlap"
                ],
                "example_overlap": score_info[
                    "example_overlap"
                ],
                "synonym_overlap": score_info[
                    "synonym_overlap"
                ],
                "overlap_words": score_info[
                    "overlap_words"
                ],
                "frequency": frequency,
            }

            candidates.append(candidate)

        # -----------------------------------------------------
        # STEP 4: SORT CANDIDATES
        # -----------------------------------------------------

        candidates.sort(
            key=lambda item: (
                item["score"],
                len(item["example_overlap"]),
                len(item["definition_overlap"]),
                item["frequency"],
            ),
            reverse=True
        )

        # -----------------------------------------------------
        # STEP 5: BEST CANDIDATE
        # -----------------------------------------------------

        best_candidate = candidates[0]

        return {
            "sentence": sentence,
            "target_word": target_word,
            "context": context_tokens,
            "target_pos": target_pos,

            "predicted_synset": (
                best_candidate["synset"]
            ),

            "predicted_name": (
                best_candidate["name"]
            ),

            "score": (
                best_candidate["score"]
            ),

            "overlap_words": (
                best_candidate["overlap_words"]
            ),

            "definition_overlap": (
                best_candidate["definition_overlap"]
            ),

            "example_overlap": (
                best_candidate["example_overlap"]
            ),

            "synonym_overlap": (
                best_candidate["synonym_overlap"]
            ),

            "candidates": candidates,
        }


# =============================================================
# NLTK LESK BASELINE
# =============================================================

class NLTKLeskModel:
    """
    Wrapper around NLTK's built-in Lesk algorithm.

    This is kept as the baseline for comparison.
    """

    def __init__(self, window_size=7):
        self.window_size = window_size

    def disambiguate(
        self,
        sentence,
        target_word
    ):
        """
        Run NLTK's built-in Lesk algorithm.
        """

        # Import here to avoid unnecessary processing
        # outside the function.
        from preprocessing import (
            clean_text,
            tokenize_text
        )

        cleaned = clean_text(sentence)
        tokens = tokenize_text(cleaned)

        synset = nltk_lesk(
            tokens,
            target_word.lower()
        )

        return {
            "sentence": sentence,
            "target_word": target_word,
            "predicted_synset": synset,
            "predicted_name": (
                synset.name()
                if synset is not None
                else None
            ),
        }


# =============================================================
# MODEL COMPARISON
# =============================================================

def compare_models(
    sentence,
    target_word,
    window_size=7
):
    """
    Compare improved custom Lesk with NLTK Lesk.
    """

    custom_model = SimplifiedLesk(
        window_size=window_size
    )

    nltk_model = NLTKLeskModel(
        window_size=window_size
    )

    custom_result = custom_model.disambiguate(
        sentence,
        target_word
    )

    nltk_result = nltk_model.disambiguate(
        sentence,
        target_word
    )

    return {
        "custom": custom_result,
        "nltk": nltk_result,
    }


# =============================================================
# DISPLAY
# =============================================================

def display_result(result):
    """
    Display the custom Lesk prediction.
    """

    print("\n" + "=" * 70)

    print(
        f"Sentence : {result['sentence']}"
    )

    print(
        f"Target   : {result['target_word']}"
    )

    print(
        f"POS      : {result['target_pos']}"
    )

    print(
        f"Context  : {result['context']}"
    )

    print(
        f"Prediction : {result['predicted_name']}"
    )

    print(
        f"Score      : {result['score']}"
    )

    print(
        f"Overlap    : {result['overlap_words']}"
    )

    print(
        f"Definition overlap : "
        f"{result['definition_overlap']}"
    )

    print(
        f"Example overlap    : "
        f"{result['example_overlap']}"
    )

    print(
        f"Synonym overlap    : "
        f"{result['synonym_overlap']}"
    )

    print("\nTop Candidates:")

    for index, candidate in enumerate(
        result["candidates"][:5],
        start=1
    ):
        print(
            f"{index}. "
            f"{candidate['name']} | "
            f"score={candidate['score']} | "
            f"freq={candidate['frequency']} | "
            f"overlap={candidate['overlap_words']}"
        )

        print(
            f"   {candidate['definition']}"
        )

    print("=" * 70)


# =============================================================
# TEST
# =============================================================

if __name__ == "__main__":

    sentence = (
        "The fisherman sat on the bank "
        "of the river."
    )

    target_word = "bank"

    result = compare_models(
        sentence,
        target_word,
        window_size=7
    )

    display_result(
        result["custom"]
    )

    print("\nNLTK Lesk Prediction:")

    print(
        result["nltk"]["predicted_name"]
    )