import argparse
import json
import math
import os
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path


VERSION = "DE"
BASE_VERSION = "0.9"


# Default question classification rules
DEFAULT_QUESTION_CLASSIFICATION = {
    "GREETING": {
        "exact": ["hi", "hello", "hey", "hiya", "greetings", "yo"],
        "prefix": ["hi ", "hello ", "hey ", "hiya "],
    },
    "GOODBYE": {
        "exact": ["bye", "goodbye", "exit", "quit"],
        "prefix": [],
    },
    "ADVANTAGE": {
        "keywords": [
            "advantage", "advantages", "benefit", "benefits",
            "pro", "pros", "strength", "strengths",
            "positive", "positives", "good"
        ]
    },
    "DISADVANTAGE": {
        "keywords": [
            "disadvantage", "disadvantages", "drawback", "drawbacks",
            "con", "cons", "weakness", "weaknesses",
            "negative", "negatives", "bad",
            "limitation", "limitations", "limit", "limits"
        ]
    },
    "COMPARISON": {
        "keywords": [
            "compare", "comparison", "difference", "differences",
            "versus", "vs", "better", "worse", "similar", "different"
        ]
    },
    "LIST": {
        "keywords": [
            "list", "examples", "example", "types", "kinds",
            "ways", "uses", "applications", "features"
        ]
    },
    "DEFINITION": {
        "prefix": [
            "what is ", "what are ", "who is ", "who are ",
            "define ", "definition of ", "meaning of ", "explain "
        ]
    },
    "WHY": {"prefix": ["why "]},
    "HOW": {"prefix": ["how "]},
    "WHEN": {"prefix": ["when "]},
    "WHERE": {"prefix": ["where "]},
    "WHO": {"prefix": ["who "]},
    "QUESTION": {"contains": ["?"]},
    "GENERAL": {}
}

# Default target extraction rules
DEFAULT_TARGET_EXTRACTION = {
    "prefixes": [
        "what is ", "what are ", "who is ", "who are ",
        "define ", "definition of ", "meaning of ", "explain ",
        "advantages of ", "advantage of ", "benefits of ", "benefit of ",
        "pros of ", "pros for ", "disadvantages of ", "disadvantage of ",
        "cons of ", "cons for ", "uses of ", "use of ",
        "examples of ", "example of ", "features of "
    ],
    "remove_question_words": [
        "what", "why", "how", "when", "where", "who", "which", "whom"
    ],
    "remove_category_words": True,
    "category_groups": ["ADVANTAGE", "DISADVANTAGE", "COMPARISON", "LIST"]
}

# Default question-fit patterns per question type
DEFAULT_QUESTION_FIT = {
    "DEFINITION": {
        "patterns": [
            [r"\b(is|are|means|refers to|defined as|is the)\b", 1.0],
            [r"\b(can be|describes|consists of)\b", 0.65]
        ],
        "fallback": 0.05
    },
    "ADVANTAGE": {
        "patterns": [
            [r"\bbenefit\b", 1.0], [r"\bbenefits\b", 1.0],
            [r"\badvantage\b", 1.0], [r"\badvantages\b", 1.0],
            [r"\buseful\b", 1.0], [r"\bhelps\b", 1.0],
            [r"\ballows\b", 1.0], [r"\bimproves\b", 1.0],
            [r"\bfaster\b", 1.0], [r"\befficient\b", 1.0],
            [r"\bcan help\b", 1.0]
        ],
        "fallback": 0.15
    },
    "DISADVANTAGE": {
        "patterns": [
            [r"\bdisadvantage\b", 1.0], [r"\bdisadvantages\b", 1.0],
            [r"\bdrawback\b", 1.0], [r"\bdrawbacks\b", 1.0],
            [r"\blimit\b", 1.0], [r"\blimits\b", 1.0],
            [r"\blimitation\b", 1.0], [r"\blimitations\b", 1.0],
            [r"\brisk\b", 1.0], [r"\bproblem\b", 1.0],
            [r"\bproblematic\b", 1.0], [r"\bfails\b", 1.0],
            [r"\bfailure\b", 1.0]
        ],
        "fallback": 0.15
    },
    "COMPARISON": {
        "patterns": [
            [r"\bhowever\b", 0.8], [r"\bwhereas\b", 0.8],
            [r"\bwhile\b", 0.5], [r"\bdifferent\b", 0.8],
            [r"\bsimilar\b", 0.8], [r"\bthan\b", 0.5]
        ],
        "fallback": 0.15
    },
    "LIST": {
        "patterns": [[r",", 0.8], [r";", 0.8], [r"\band\b", 0.35]],
        "fallback": 0.15
    },
    "WHY": {
        "patterns": [
            [r"\bbecause\b", 0.9], [r"\bsince\b", 0.6],
            [r"\bdue to\b", 0.6], [r"\bcaused by\b", 0.8]
        ],
        "fallback": 0.2
    },
    "HOW": {
        "patterns": [
            [r"\bfirst\b", 0.4], [r"\bthen\b", 0.4],
            [r"\bnext\b", 0.5], [r"\bfinally\b", 0.5],
            [r"\bprocess\b", 0.7], [r"\bsteps?\b", 0.8]
        ],
        "fallback": 0.2
    },
    "WHEN": {
        "patterns": [
            [r"\bwhen\b", 0.4], [r"\bduring\b", 0.6],
            [r"\bafter\b", 0.5], [r"\bbefore\b", 0.5],
            [r"\byear\b", 0.5], [r"\btime\b", 0.5]
        ],
        "fallback": 0.2
    },
    "WHERE": {
        "patterns": [
            [r"\bin\b", 0.25], [r"\bat\b", 0.25],
            [r"\blocated\b", 0.8], [r"\bplace\b", 0.5],
            [r"\barea\b", 0.5]
        ],
        "fallback": 0.2
    },
    "WHO": {
        "patterns": [
            [r"\bperson\b", 0.7], [r"\bpeople\b", 0.7],
            [r"\bscientist\b", 0.7], [r"\bdeveloper\b", 0.7],
            [r"\binventor\b", 0.7], [r"\bfounded by\b", 0.8]
        ],
        "fallback": 0.2
    },
    "QUESTION": {"patterns": [], "fallback": 0.2},
    "GENERAL": {"patterns": [], "fallback": 0.15}
}

# Default stopwords
DEFAULT_LANGUAGE = {
    "stopwords": [
        "a", "an", "the", "is", "are", "was", "were", "be", "been",
        "being", "to", "of", "in", "on", "at", "for", "from", "by",
        "with", "about", "into", "and", "or", "but", "as", "that",
        "this", "these", "those", "it", "its", "they", "them", "their",
        "there", "here", "can", "could", "would", "should", "may",
        "might", "do", "does", "did", "what", "why", "how", "when",
        "where", "who", "which", "whom", "your", "you", "i", "we",
        "me", "my", "our", "us", "he", "she", "his", "her", "have",
        "has", "had", "will", "shall", "than", "then", "also", "very",
        "more", "most", "some", "any", "many", "much"
    ]
}

# Default generation settings
DEFAULT_GENERATION = {
    "min_tokens": 5,
    "max_tokens": 32,
    "temperature": 0.75,
    "attempts_per_seed": 3,
    "max_candidates": 10,
    "stop_probability": 0.08,
    "max_token_repeats": 3,
    "max_sentence_words": 55,
    "min_sentence_words": 4
}

# Default scoring weights
DEFAULT_SCORING = {
    "retrieval": {
        "coverage_weight": 7.0,
        "phrase_weight": 5.0,
        "question_fit_weight": 8.0,
        "definition_fit_weight": 8.0,
        "topic_fit_weight": 7.0,
        "definition_extra_fit_weight": 10.0,
        "definition_extra_question_weight": 8.0,
        "category_topic_extra_weight": 6.0,
        "confidence_strength_divisor": 30.0,
        "confidence_single_divisor": 25.0,
        "confidence_strength_weight": 0.7,
        "confidence_separation_weight": 0.3
    },
    "generation": {
        "bigram_weight": 1.0,
        "trigram_weight": 0.5,
        "fourgram_weight": 0.35,
        "relevance_weight": 8.0,
        "answer_fit_weight": 7.0,
        "language_weight": 0.30,
        "repetition_penalty": 4.0,
        "parroting_penalty": 2.0,
        "topic_candidate_multiplier": 2.5,
        "confidence_divisor": 15.0,
        "confidence_strength_weight": 0.5,
        "confidence_separation_weight": 0.2,
        "confidence_evidence_weight": 0.3,
        "router_generation_margin": 0.08
    }
}

# Default dictionary for typo correction
DEFAULT_DICTIONARY = {
    "max_edit_distance": 2,
    "min_word_length": 2,
    "auto_correct": True,
    "corpus_words_are_valid": True,
    "frequency_bonus": 1.0,
    "context_bonus": 2.5,
    "minimum_score": 0.72,
    "maximum_candidates": 5,
    "protected_words": [
        "chatnnl", "project", "non", "gguf", "qwen", "deepseek",
        "onnx", "rwkv", "lightgbm", "xgboost", "tfidf", "svd",
        "python", "javascript", "html", "css", "json", "api",
        "ai", "ml", "llm", "gpu", "cpu", "ram", "android"
    ],
    "words": {
        "about": 100000, "above": 90000, "after": 90000, "again": 80000,
        "against": 70000, "all": 120000, "also": 120000, "always": 80000,
        "answer": 100000, "application": 70000, "astronomy": 60000,
        "because": 130000, "before": 90000, "between": 100000,
        "computer": 120000, "communication": 60000, "concept": 60000,
        "could": 120000, "data": 130000, "definition": 50000,
        "development": 60000, "different": 100000, "dictionary": 45000,
        "example": 90000, "explain": 60000, "feature": 70000,
        "first": 120000, "from": 180000, "general": 70000, "generate": 50000,
        "generation": 60000, "good": 130000, "have": 180000,
        "information": 100000, "intelligence": 70000, "language": 100000,
        "learning": 90000, "machine": 80000, "model": 100000,
        "network": 70000, "neural": 50000, "next": 90000, "number": 90000,
        "object": 70000, "programming": 70000, "question": 90000,
        "reason": 70000, "retrieval": 45000, "science": 70000,
        "sentence": 70000, "should": 100000, "simple": 60000,
        "software": 80000, "space": 70000, "statistical": 50000,
        "system": 130000, "technical": 50000, "technology": 80000,
        "term": 70000, "text": 120000, "that": 220000, "their": 160000,
        "there": 150000, "these": 130000, "they": 170000, "this": 200000,
        "through": 120000, "token": 60000, "topic": 60000, "under": 90000,
        "use": 160000, "useful": 70000, "using": 140000, "version": 60000,
        "very": 130000, "vocabulary": 45000, "what": 170000, "when": 100000,
        "where": 90000, "which": 150000, "while": 100000, "with": 190000,
        "word": 150000, "work": 130000, "would": 140000
    }
}


def deep_merge(base, override):
    # Recursively merge two dicts, override wins
    if not isinstance(base, dict):
        return override
    result = dict(base)
    if not isinstance(override, dict):
        return result
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_json_file(path, required=False):
    # Load JSON config file into a dict
    if not path:
        return {}
    if not os.path.exists(path):
        if required:
            raise FileNotFoundError(f"JSON configuration not found: {path}")
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"JSON configuration must contain an object: {path}")
    return data


def as_list(value):
    # Normalize any value into a list
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


class ChatNNL:
    def __init__(
        self,
        corpus_paths,
        aliases_path=None,
        question_fit_path=None,
        target_extraction_path=None,
        question_classification_path=None,
        dictionary_path=None,
        config_path=None,
        debug=False,
        no_generation=False,
        only_generate=False
    ):
        self.corpus_paths = corpus_paths
        self.aliases_path = aliases_path
        self.question_fit_path = question_fit_path
        self.target_extraction_path = target_extraction_path
        self.question_classification_path = question_classification_path
        self.dictionary_path = dictionary_path
        self.config_path = config_path
        self.debug = debug
        self.no_generation = no_generation
        self.only_generate = only_generate

        external_config = load_json_file(config_path, required=bool(config_path))

        # Merge built-in defaults with external config
        self.config = deep_merge({
            "language": DEFAULT_LANGUAGE,
            "generation": DEFAULT_GENERATION,
            "scoring": DEFAULT_SCORING,
            "question_fit": DEFAULT_QUESTION_FIT,
            "target_extraction": DEFAULT_TARGET_EXTRACTION,
            "question_classification": DEFAULT_QUESTION_CLASSIFICATION,
            "dictionary": DEFAULT_DICTIONARY
        }, external_config)

        # Dedicated files override master config
        self.config["question_fit"] = deep_merge(
            self.config["question_fit"],
            load_json_file(question_fit_path, required=bool(question_fit_path))
        )
        self.config["target_extraction"] = deep_merge(
            self.config["target_extraction"],
            load_json_file(target_extraction_path, required=bool(target_extraction_path))
        )
        self.config["question_classification"] = deep_merge(
            self.config["question_classification"],
            load_json_file(
                question_classification_path,
                required=bool(question_classification_path)
            )
        )
        self.config["dictionary"] = deep_merge(
            self.config["dictionary"],
            load_json_file(dictionary_path, required=bool(dictionary_path))
        )

        self.sentences = []
        self.sentence_sources = []

        self.vocab = Counter()
        self.unigrams = Counter()
        self.bigrams = Counter()
        self.trigrams = Counter()
        self.fourgrams = Counter()
        self.fivegrams = Counter()

        self.bigram_next = defaultdict(Counter)
        self.trigram_next = defaultdict(Counter)
        self.fourgram_next = defaultdict(Counter)
        self.fivegram_next = defaultdict(Counter)

        self.sentence_starts = []
        self.sentence_ends = []

        self.document_frequency = Counter()
        self.idf = {}
        self.sentence_tokens = []
        self.sentence_sets = []

        self.aliases = {}
        self.alias_lookup = {}
        self.alias_concepts = 0

        self.stopwords = set(
            str(x).lower()
            for x in self.config["language"].get("stopwords", [])
        )

        self.question_words = set(
            str(x).lower()
            for x in self.config["target_extraction"].get(
                "remove_question_words", []
            )
        )

        # Precompute category word sets for target extraction
        self.category_words = {}
        for category in self.config["target_extraction"].get("category_groups", []):
            section = self.config["question_classification"].get(category, {})
            words = set()
            for key in ("keywords", "exact", "prefix"):
                for item in as_list(section.get(key)):
                    item = str(item).strip().lower()
                    words.update(self.tokenize(item))
            self.category_words[category] = words

        self.greetings = set(
            str(x).lower()
            for x in self.config["question_classification"].get(
                "GREETING", {}
            ).get("exact", [])
        )
        self.goodbyes = set(
            str(x).lower()
            for x in self.config["question_classification"].get(
                "GOODBYE", {}
            ).get("exact", [])
        )

        self.dictionary_words = {}
        self.dictionary_deletes = defaultdict(set)
        self.dictionary_corrections = 0
        self.dictionary_known_words = 0
        self.build_dictionary()

        self.load_aliases()
        self.load_corpora()
        self.build_statistics()

        # Corpus words become valid dictionary entries
        if self.config["dictionary"].get("corpus_words_are_valid", True):
            self.add_corpus_words_to_dictionary()

    # Tokenize text into words
    def tokenize(self, text):
        text = str(text).lower()
        return re.findall(r"[a-z0-9]+(?:'[a-z]+)?", text)

    # Split text into sentences
    def split_sentences(self, text):
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        chunks = re.split(r"(?<=[.!?])\s+|\n+", text)
        result = []
        for chunk in chunks:
            chunk = chunk.strip()
            if not chunk:
                continue
            if len(self.tokenize(chunk)) < 2:
                continue
            result.append(chunk)
        return result

    # Build dictionary words from config
    def build_dictionary(self):
        cfg = self.config.get("dictionary", {})
        raw_words = cfg.get("words", {})

        self.dictionary_words = {}

        if isinstance(raw_words, list):
            for word in raw_words:
                word = self.normalize_dictionary_word(word)
                if word:
                    self.dictionary_words[word] = 1
        elif isinstance(raw_words, dict):
            for word, frequency in raw_words.items():
                word = self.normalize_dictionary_word(word)
                if not word:
                    continue
                try:
                    frequency = float(frequency)
                except (TypeError, ValueError):
                    frequency = 1.0
                self.dictionary_words[word] = max(1.0, frequency)

        # Protected words get very high frequency
        for word in as_list(cfg.get("protected_words")):
            word = self.normalize_dictionary_word(word)
            if word:
                self.dictionary_words[word] = max(
                    self.dictionary_words.get(word, 1.0),
                    10_000_000.0
                )

        self.dictionary_known_words = len(self.dictionary_words)
        self.rebuild_dictionary_index()

    # Clean a word for dictionary storage
    def normalize_dictionary_word(self, word):
        if not isinstance(word, str):
            return ""
        word = word.strip().lower()
        if not re.fullmatch(r"[a-z0-9]+(?:'[a-z]+)?", word):
            return ""
        return word

    # Generate all delete variants of a word up to max_distance
    def dictionary_deletes_for(self, word, max_distance):
        deletes = {word}
        current = {word}
        for _ in range(max_distance):
            next_level = set()
            for item in current:
                for i in range(len(item)):
                    deleted = item[:i] + item[i + 1:]
                    if deleted not in deletes:
                        deletes.add(deleted)
                        next_level.add(deleted)
            current = next_level
            if not current:
                break
        deletes.discard(word)
        return deletes

    # Build delete index for fast typo lookup
    def rebuild_dictionary_index(self):
        self.dictionary_deletes = defaultdict(set)
        cfg = self.config.get("dictionary", {})
        try:
            max_distance = int(cfg.get("max_edit_distance", 2))
        except (TypeError, ValueError):
            max_distance = 2
        max_distance = max(0, min(max_distance, 2))

        for word in self.dictionary_words:
            if len(word) < int(cfg.get("min_word_length", 2)):
                continue
            for deleted in self.dictionary_deletes_for(word, max_distance):
                self.dictionary_deletes[deleted].add(word)

    # Add words to dictionary if frequency is higher
    def add_dictionary_words(self, words):
        changed = False
        for word, frequency in words.items():
            normalized = self.normalize_dictionary_word(word)
            if not normalized:
                continue
            try:
                frequency = float(frequency)
            except (TypeError, ValueError):
                frequency = 1.0
            old = self.dictionary_words.get(normalized, 0.0)
            if frequency > old:
                self.dictionary_words[normalized] = frequency
                changed = True
        if changed:
            self.dictionary_known_words = len(self.dictionary_words)
            self.rebuild_dictionary_index()

    # Add corpus vocabulary to dictionary
    def add_corpus_words_to_dictionary(self):
        additions = {}
        for word, count in self.vocab.items():
            if word not in self.dictionary_words:
                additions[word] = max(1.0, float(count))
        self.add_dictionary_words(additions)

    # Damerau-Levenshtein edit distance
    def edit_distance(self, a, b):
        if a == b:
            return 0
        if not a:
            return len(b)
        if not b:
            return len(a)

        prev2 = list(range(len(b) + 1))
        prev1 = prev2[:]
        prev1[0] = 0

        for i in range(1, len(a) + 1):
            current = [i] + [0] * len(b)
            for j in range(1, len(b) + 1):
                cost = 0 if a[i - 1] == b[j - 1] else 1
                current[j] = min(
                    prev1[j] + 1,
                    current[j - 1] + 1,
                    prev1[j - 1] + cost
                )
                if (
                    i > 1 and j > 1
                    and a[i - 1] == b[j - 2]
                    and a[i - 2] == b[j - 1]
                ):
                    current[j] = min(
                        current[j],
                        prev2[j - 2] + cost
                    )
            prev2, prev1 = prev1, current

        return prev1[-1]

    # Find dictionary candidates for a token
    def dictionary_candidates(self, word):
        word = self.normalize_dictionary_word(word)
        if not word:
            return []

        if word in self.dictionary_words:
            return [{
                "word": word,
                "distance": 0,
                "frequency": self.dictionary_words[word],
                "score": 1.0,
                "known": True
            }]

        cfg = self.config.get("dictionary", {})
        try:
            max_distance = int(cfg.get("max_edit_distance", 2))
        except (TypeError, ValueError):
            max_distance = 2
        max_distance = max(0, min(max_distance, 2))

        if len(word) < int(cfg.get("min_word_length", 2)):
            return []

        candidate_words = set()

        # Symmetric delete lookup
        for deleted in self.dictionary_deletes_for(word, max_distance):
            candidate_words.update(self.dictionary_deletes.get(deleted, set()))

        # Also try prefix matches for short terms
        prefix = word[:min(4, len(word))]
        for candidate in self.dictionary_words:
            if candidate.startswith(prefix) or prefix.startswith(candidate[:len(prefix)]):
                if abs(len(candidate) - len(word)) <= max_distance:
                    candidate_words.add(candidate)

        results = []
        max_freq = max(self.dictionary_words.values(), default=1.0)
        freq_bonus = float(cfg.get("frequency_bonus", 1.0))

        for candidate in candidate_words:
            distance = self.edit_distance(word, candidate)
            if distance > max_distance:
                continue

            # Distance dominates, frequency breaks ties
            distance_score = 1.0 - (
                distance / max(1, max_distance + 1)
            )
            frequency_score = (
                math.log1p(self.dictionary_words.get(candidate, 1.0))
                / max(1.0, math.log1p(max_freq))
            )

            score = (
                distance_score * 0.85
                + frequency_score * 0.15 * freq_bonus
            )

            results.append({
                "word": candidate,
                "distance": distance,
                "frequency": self.dictionary_words.get(candidate, 1.0),
                "score": score,
                "known": False
            })

        results.sort(
            key=lambda x: (
                x["distance"],
                -x["score"],
                -x["frequency"]
            )
        )

        return results[:int(cfg.get("maximum_candidates", 5))]

    # Correct a single token using the dictionary
    def dictionary_correct_token(self, token, context_tokens=None):
        token = self.normalize_dictionary_word(token)
        if not token:
            return token, None

        if token in self.dictionary_words:
            return token, None

        cfg = self.config.get("dictionary", {})
        if not cfg.get("auto_correct", True):
            return token, None

        candidates = self.dictionary_candidates(token)
        if not candidates:
            return token, None

        context_tokens = context_tokens or []
        best = None

        for candidate in candidates:
            score = candidate["score"]

            # Boost candidates using bigram context
            if context_tokens:
                prev = context_tokens[-1]
                bigram_count = self.bigrams.get((prev, candidate["word"]), 0)
                if bigram_count:
                    score += float(cfg.get("context_bonus", 2.5)) * min(
                        0.25, math.log1p(bigram_count) / 20.0
                    )

            candidate["context_score"] = score

            if best is None or score > best["context_score"]:
                best = candidate

        threshold = float(cfg.get("minimum_score", 0.72))

        if best and best["context_score"] >= threshold:
            self.dictionary_corrections += 1
            return best["word"], best

        return token, best

    # Correct a list of tokens
    def correct_tokens(self, tokens):
        corrected = []
        changes = []

        for token in tokens:
            # Skip short and numeric tokens
            if len(token) <= 2 or token.isdigit():
                corrected.append(token)
                continue

            corrected_token, evidence = self.dictionary_correct_token(
                token,
                corrected
            )
            corrected.append(corrected_token)

            if corrected_token != token:
                changes.append({
                    "from": token,
                    "to": corrected_token,
                    "distance": evidence["distance"] if evidence else None,
                    "score": evidence["context_score"] if evidence else 0.0
                })

        return corrected, changes

    # Correct free text
    def correct_text(self, text):
        tokens = self.tokenize(text)
        corrected, changes = self.correct_tokens(tokens)
        return " ".join(corrected), changes

    # Load aliases from JSON
    def load_aliases(self):
        self.aliases = {}
        self.alias_lookup = {}

        if not self.aliases_path:
            self.alias_concepts = 0
            return

        data = load_json_file(self.aliases_path, required=True)

        if "aliases" in data and isinstance(data["aliases"], dict):
            data = data["aliases"]

        for concept, values in data.items():
            concept = str(concept).strip().lower()
            if not concept:
                continue
            if isinstance(values, str):
                values = [values]
            if not isinstance(values, list):
                continue

            normalized = {concept}
            for value in values:
                if isinstance(value, str):
                    value = value.strip().lower()
                    if value:
                        normalized.add(value)

            self.aliases[concept] = sorted(normalized)
            self.alias_lookup[concept] = concept
            for alias in normalized:
                self.alias_lookup[alias] = concept

        self.alias_concepts = len(self.aliases)

    # Load all corpus files
    def load_corpora(self):
        total_tokens = 0

        for path in self.corpus_paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Corpus not found: {path}")

            with open(path, "r", encoding="utf-8") as f:
                text = f.read()

            sentences = self.split_sentences(text)

            for sentence in sentences:
                self.sentences.append(sentence)
                self.sentence_sources.append(path)
                total_tokens += len(self.tokenize(sentence))

        self.total_tokens = total_tokens

    # Build n-gram statistics from corpus
    def build_statistics(self):
        for sentence in self.sentences:
            tokens = self.tokenize(sentence)
            if not tokens:
                continue

            self.sentence_tokens.append(tokens)
            self.sentence_sets.append(set(tokens))
            self.vocab.update(tokens)
            self.unigrams.update(tokens)
            self.sentence_starts.append(tokens[0])
            self.sentence_ends.append(tokens[-1])

            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                self.bigrams[pair] += 1
                self.bigram_next[tokens[i]][tokens[i + 1]] += 1

            for i in range(len(tokens) - 2):
                gram = tuple(tokens[i:i + 3])
                self.trigrams[gram] += 1
                self.trigram_next[gram[:2]][gram[2]] += 1

            for i in range(len(tokens) - 3):
                gram = tuple(tokens[i:i + 4])
                self.fourgrams[gram] += 1
                self.fourgram_next[gram[:3]][gram[3]] += 1

            for i in range(len(tokens) - 4):
                gram = tuple(tokens[i:i + 5])
                self.fivegrams[gram] += 1
                self.fivegram_next[gram[:4]][gram[4]] += 1

            for token in set(tokens):
                self.document_frequency[token] += 1

        sentence_count = max(len(self.sentences), 1)
        for token, df in self.document_frequency.items():
            self.idf[token] = math.log(
                (sentence_count + 1) / (df + 1)
            ) + 1.0

    # Normalize text to space-separated tokens
    def normalize(self, text):
        return " ".join(self.tokenize(text))

    # Expand tokens with alias terms
    def expand_terms(self, tokens):
        expanded = set(tokens)

        for token in tokens:
            concept = self.alias_lookup.get(token)
            if concept:
                expanded.add(concept)
                for alias in self.aliases.get(concept, []):
                    expanded.update(self.tokenize(alias))

        # Check multi-word alias phrases
        phrase_candidates = []
        for size in range(2, min(5, len(tokens)) + 1):
            for i in range(len(tokens) - size + 1):
                phrase_candidates.append(" ".join(tokens[i:i + size]))

        for phrase in phrase_candidates:
            concept = self.alias_lookup.get(phrase)
            if concept:
                expanded.add(concept)
                for alias in self.aliases.get(concept, []):
                    expanded.update(self.tokenize(alias))

        return expanded

    # Filter out stopwords and short tokens
    def important_tokens(self, tokens):
        return [
            token for token in tokens
            if token not in self.stopwords and len(token) > 1
        ]

    # Detect question type from text
    def detect_question_type(self, text, tokens):
        lower = text.lower().strip()
        rules = self.config.get("question_classification", {})

        for question_type, rule in rules.items():
            if not isinstance(rule, dict):
                continue
            exact = {
                str(x).strip().lower()
                for x in as_list(rule.get("exact"))
            }
            if lower in exact:
                return question_type

        for question_type, rule in rules.items():
            if not isinstance(rule, dict):
                continue
            prefixes = [str(x).lower() for x in as_list(rule.get("prefix"))]
            if any(lower.startswith(x) for x in prefixes if x):
                return question_type

        important = set(self.important_tokens(tokens))

        for question_type, rule in rules.items():
            if not isinstance(rule, dict):
                continue
            keywords = {
                str(x).strip().lower()
                for x in as_list(rule.get("keywords"))
            }
            if important & keywords:
                return question_type

        for question_type, rule in rules.items():
            if not isinstance(rule, dict):
                continue
            contains = [str(x).lower() for x in as_list(rule.get("contains"))]
            if any(x in lower for x in contains if x):
                return question_type

        return "GENERAL"

    # Extract the target/topic from the query
    def extract_target(self, text, tokens, question_type):
        lower = text.lower().strip()
        config = self.config.get("target_extraction", {})

        prefixes = [
            str(x).strip().lower()
            for x in as_list(config.get("prefixes"))
        ]
        prefixes.sort(key=len, reverse=True)

        for prefix in prefixes:
            if lower.startswith(prefix):
                target = lower[len(prefix):].strip(" ?!.")
                if target:
                    return target

        important = self.important_tokens(tokens)

        if config.get("remove_question_words", True):
            important = [
                token for token in important
                if token not in self.question_words
            ]

        if (
            config.get("remove_category_words", True)
            and question_type in config.get("category_groups", [])
        ):
            category_words = self.category_words.get(question_type, set())
            important = [
                token for token in important
                if token not in category_words
            ]

        return " ".join(important)

    # Collect term set for target including aliases
    def target_terms(self, target):
        tokens = self.tokenize(target)
        terms = set(tokens)

        for token in tokens:
            concept = self.alias_lookup.get(token)
            if concept:
                terms.add(concept)
                terms.update(self.tokenize(concept))
                for alias in self.aliases.get(concept, []):
                    terms.update(self.tokenize(alias))

        concept = self.alias_lookup.get(target)
        if concept:
            terms.add(concept)
            for alias in self.aliases.get(concept, []):
                terms.update(self.tokenize(alias))

        return terms

    # Score phrase matches between query and sentence
    def phrase_score(self, query_tokens, sentence_tokens):
        if not query_tokens or not sentence_tokens:
            return 0.0

        query = " ".join(query_tokens)
        sentence = " ".join(sentence_tokens)

        if query in sentence:
            return 1.0

        matches = 0
        total = 0

        for size in range(2, min(5, len(query_tokens)) + 1):
            for i in range(len(query_tokens) - size + 1):
                total += 1
                phrase = " ".join(query_tokens[i:i + size])
                if phrase in sentence:
                    matches += 1

        return matches / total if total else 0.0

    # Sum of IDF for matching tokens
    def lexical_score(self, terms, sentence_tokens):
        return sum(
            self.idf.get(token, 1.0)
            for token in sentence_tokens
            if token in terms
        )

    # Fraction of query terms present in sentence
    def coverage_score(self, terms, sentence_tokens):
        if not terms:
            return 0.0
        matched = len(set(sentence_tokens) & terms)
        return min(1.0, matched / max(1, len(terms)))

    # Score sentence fit for question type
    def question_fit(self, sentence, question_type):
        section = self.config.get("question_fit", {}).get(question_type, {})
        if not isinstance(section, dict):
            return 0.0

        for item in section.get("patterns", []):
            if not isinstance(item, list) or len(item) < 2:
                continue
            pattern = str(item[0])
            try:
                weight = float(item[1])
            except (TypeError, ValueError):
                continue
            try:
                if re.search(pattern, sentence.lower()):
                    return weight
            except re.error:
                continue

        try:
            return float(section.get("fallback", 0.0))
        except (TypeError, ValueError):
            return 0.0

    # Score definition-like patterns
    def definition_fit(self, sentence, target):
        if not target:
            return 0.0

        target_terms = self.target_terms(target)
        sentence_tokens = self.tokenize(sentence)

        if not (set(sentence_tokens) & target_terms):
            return 0.0

        text = sentence.lower()

        if re.search(r"\b(is|are|means|refers to|defined as|is the)\b", text):
            return 1.0

        if re.search(r"\b(can be|describes|consists of|is a|is an)\b", text):
            return 0.85

        return 0.0

    # Score topic overlap
    def topic_fit(self, sentence, target):
        if not target:
            return 0.0

        terms = self.target_terms(target)
        tokens = set(self.tokenize(sentence))

        if not terms:
            return 0.0

        return min(1.0, len(tokens & terms) / max(1, min(3, len(terms))))

    # Retrieve top candidate sentences for a query
    def retrieve(self, query, question_type, target, limit=12):
        query_tokens = self.tokenize(query)
        important = self.important_tokens(query_tokens)

        terms = self.expand_terms(important)
        terms.update(self.target_terms(target))
        scoring = self.config["scoring"]["retrieval"]

        candidates = []

        for index, sentence in enumerate(self.sentences):
            sentence_tokens = self.sentence_tokens[index]

            lexical = self.lexical_score(terms, sentence_tokens)
            coverage = self.coverage_score(terms, sentence_tokens)
            phrase = self.phrase_score(important, sentence_tokens)
            qfit = self.question_fit(sentence, question_type)
            dfit = self.definition_fit(sentence, target)
            tfit = self.topic_fit(sentence, target)

            score = (
                lexical
                + coverage * float(scoring.get("coverage_weight", 7.0))
                + phrase * float(scoring.get("phrase_weight", 5.0))
                + qfit * float(scoring.get("question_fit_weight", 8.0))
                + dfit * float(scoring.get("definition_fit_weight", 8.0))
                + tfit * float(scoring.get("topic_fit_weight", 7.0))
            )

            # Extra weight for DEFINITION questions
            if question_type == "DEFINITION":
                score += dfit * float(scoring.get("definition_extra_fit_weight", 10.0))
                score += qfit * float(scoring.get("definition_extra_question_weight", 8.0))

            # Extra weight for advantage/disadvantage questions
            if question_type in {"ADVANTAGE", "DISADVANTAGE"}:
                score += tfit * float(scoring.get("category_topic_extra_weight", 6.0))

            if score <= 0:
                continue

            candidates.append({
                "index": index,
                "sentence": sentence,
                "source": self.sentence_sources[index],
                "lexical": lexical,
                "coverage": coverage,
                "phrase": phrase,
                "question_fit": qfit,
                "definition_fit": dfit,
                "topic_fit": tfit,
                "score": score
            })

        candidates.sort(key=lambda x: x["score"], reverse=True)
        return candidates[:limit]

    # Estimate retrieval confidence
    def retrieval_confidence(self, candidates):
        if not candidates:
            return 0.0

        scoring = self.config["scoring"]["retrieval"]
        top = candidates[0]["score"]

        if len(candidates) == 1:
            return min(
                1.0,
                top / float(scoring.get("confidence_single_divisor", 25.0))
            )

        second = candidates[1]["score"]
        strength = min(
            1.0,
            top / float(scoring.get("confidence_strength_divisor", 30.0))
        )
        separation = min(
            1.0,
            max(0.0, (top - second) / max(top, 1.0))
        )

        return min(
            1.0,
            strength * float(scoring.get("confidence_strength_weight", 0.7))
            + separation * float(scoring.get("confidence_separation_weight", 0.3))
        )

    # Score how natural a token sequence is
    def language_score(self, tokens):
        if not tokens:
            return -100.0

        weights = self.config["scoring"]["generation"]
        score = 0.0

        for i in range(len(tokens) - 1):
            count = self.bigrams.get((tokens[i], tokens[i + 1]), 0)
            score += math.log(count + 1) * float(weights.get("bigram_weight", 1.0))

        for i in range(len(tokens) - 2):
            count = self.trigrams.get(tuple(tokens[i:i + 3]), 0)
            score += math.log(count + 1) * float(weights.get("trigram_weight", 0.5))

        for i in range(len(tokens) - 3):
            count = self.fourgrams.get(tuple(tokens[i:i + 4]), 0)
            score += math.log(count + 1) * float(weights.get("fourgram_weight", 0.35))

        return score / max(1, len(tokens))

    # Get next-token candidates from highest matching n-gram
    def generation_next_candidates(self, context, allowed_terms=None):
        context = list(context)
        candidates = Counter()

        # Try 5-gram down to bigram
        if len(context) >= 4:
            key = tuple(context[-4:])
            if key in self.fivegram_next:
                candidates.update(self.fivegram_next[key])

        if not candidates and len(context) >= 3:
            key = tuple(context[-3:])
            if key in self.fourgram_next:
                candidates.update(self.fourgram_next[key])

        if not candidates and len(context) >= 2:
            key = tuple(context[-2:])
            if key in self.trigram_next:
                candidates.update(self.trigram_next[key])

        if not candidates and len(context) >= 1:
            key = context[-1]
            if key in self.bigram_next:
                candidates.update(self.bigram_next[key])

        if not candidates:
            candidates.update(self.unigrams)

        # Boost topic-relevant candidates
        if allowed_terms:
            topic_candidates = Counter()
            multiplier = float(
                self.config["scoring"]["generation"].get(
                    "topic_candidate_multiplier", 2.5
                )
            )
            for token, count in candidates.items():
                if token in allowed_terms:
                    topic_candidates[token] = count * multiplier
            if topic_candidates:
                candidates.update(topic_candidates)

        return candidates

    # Weighted random choice with temperature
    def weighted_choice(self, counter, temperature=None):
        if not counter:
            return None

        if temperature is None:
            temperature = float(
                self.config["generation"].get("temperature", 0.75)
            )

        temperature = max(0.05, temperature)
        items = list(counter.items())
        weights = [float(count) ** (1.0 / temperature) for _, count in items]
        total = sum(weights)

        if total <= 0:
            return random.choice(items)[0]

        point = random.random() * total
        cumulative = 0.0

        for (token, _), weight in zip(items, weights):
            cumulative += weight
            if point <= cumulative:
                return token

        return items[-1][0]

    # Build seed sequences for generation
    def generation_seed_candidates(self, target, retrieved, question_type):
        seeds = []
        target_terms = self.target_terms(target)

        for item in retrieved[:8]:
            tokens = self.sentence_tokens[item["index"]]
            if not tokens:
                continue

            positions = [
                i for i, token in enumerate(tokens)
                if token in target_terms
            ]

            for pos in positions:
                start = max(0, pos - 2)
                end = min(len(tokens), pos + 2)
                seed = tokens[start:end]
                if len(seed) >= 2:
                    seeds.append(seed)

            if len(tokens) >= 3:
                seeds.append(tokens[:3])

        # Add category-specific seeds for advantage/disadvantage
        if question_type in {"ADVANTAGE", "DISADVANTAGE"}:
            category_words = self.category_words.get(question_type, set())
            for tokens in self.sentence_tokens:
                if not (set(tokens) & target_terms):
                    continue
                if not (set(tokens) & category_words):
                    continue
                seeds.append(tokens[:min(5, len(tokens))])

        unique = []
        seen = set()
        for seed in seeds:
            key = tuple(seed)
            if key in seen:
                continue
            seen.add(key)
            unique.append(seed)

        return unique

    # Generate one candidate sentence from a seed
    def generate_sentence(self, seed, target_terms, question_type):
        if not seed:
            return None

        settings = self.config["generation"]
        min_tokens = int(settings.get("min_tokens", 5))
        max_tokens = int(settings.get("max_tokens", 32))
        stop_probability = float(settings.get("stop_probability", 0.08))
        max_repeats = int(settings.get("max_token_repeats", 3))

        output = list(seed)

        for _ in range(max_tokens):
            if len(output) >= max_tokens:
                break

            if len(output) >= min_tokens and random.random() < stop_probability:
                break

            candidates = self.generation_next_candidates(output, target_terms)
            if not candidates:
                break

            candidates.pop(output[-1], None)
            if not candidates:
                break

            filtered = Counter()
            for token, count in candidates.items():
                if output.count(token) >= max_repeats:
                    continue
                filtered[token] = count

            if filtered:
                candidates = filtered

            next_token = self.weighted_choice(candidates)
            if not next_token:
                break

            output.append(next_token)

        if len(output) < min_tokens:
            return None

        return self.sentence_clean(" ".join(output))

    # Reject obviously bad generated sentences
    def valid_generation_sentence(self, sentence):
        tokens = self.tokenize(sentence)
        settings = self.config["generation"]

        min_words = int(settings.get("min_sentence_words", 4))
        max_words = int(settings.get("max_sentence_words", 55))

        if len(tokens) < min_words or len(tokens) > max_words:
            return False

        if sentence.count(".") > 3:
            return False

        banned_starts = self.config.get(
            "generation_quality", {}
        ).get(
            "banned_starts",
            [
                "and ", "or ", "but ", "because ",
                "therefore ", "while ", "which ",
                "that ", "also "
            ]
        )

        if sentence.lower().startswith(
            tuple(str(x).lower() for x in banned_starts)
        ):
            return False

        counts = Counter(tokens)
        max_repeats = int(settings.get("max_token_repeats", 3))

        for token, count in counts.items():
            if len(token) > 3 and count >= max_repeats + 1:
                return False

        return True

    # Jaccard similarity between two sentences
    def sentence_similarity(self, a, b):
        sa = set(self.tokenize(a))
        sb = set(self.tokenize(b))
        if not sa or not sb:
            return 0.0
        return len(sa & sb) / len(sa | sb)

    # Topic relevance of a generated sentence
    def generation_relevance(self, sentence, target_terms):
        tokens = set(self.tokenize(sentence))
        if not target_terms:
            return 0.0
        return min(
            1.0,
            len(tokens & target_terms) / max(1, min(5, len(target_terms)))
        )

    # Answer fit of a generated sentence
    def generation_answer_fit(self, sentence, question_type, target):
        score = self.question_fit(sentence, question_type)
        if question_type == "DEFINITION":
            score = max(score, self.definition_fit(sentence, target))
        return score

    # Score a generated sentence
    def score_generated_sentence(
        self,
        sentence,
        target_terms,
        question_type,
        target,
        existing
    ):
        tokens = self.tokenize(sentence)
        if not tokens:
            return -100.0, 0.0, 0.0

        weights = self.config["scoring"]["generation"]
        language = self.language_score(tokens)
        relevance = self.generation_relevance(sentence, target_terms)
        answer_fit = self.generation_answer_fit(sentence, question_type, target)

        # Penalize repetition with previously generated sentences
        repetition = 0.0
        for previous in existing:
            repetition = max(
                repetition,
                self.sentence_similarity(sentence, previous)
            )

        # Check parroting against corpus
        corpus_similarity = 0.0
        for corpus_sentence in self.sentences:
            corpus_similarity = max(
                corpus_similarity,
                self.sentence_similarity(sentence, corpus_sentence)
            )

        score = (
            language * float(weights.get("language_weight", 0.30))
            + relevance * float(weights.get("relevance_weight", 8.0))
            + answer_fit * float(weights.get("answer_fit_weight", 7.0))
            - repetition * float(weights.get("repetition_penalty", 4.0))
        )

        if corpus_similarity > 0.95:
            score -= float(weights.get("parroting_penalty", 2.0))

        return score, repetition, corpus_similarity

    # Generate and score candidate sentences
    def generate_candidates(
        self,
        query,
        question_type,
        target,
        retrieved,
        limit=None
    ):
        if not retrieved:
            return []

        settings = self.config["generation"]
        if limit is None:
            limit = int(settings.get("max_candidates", 10))

        target_terms = self.target_terms(target)
        target_terms.update(
            self.important_tokens(self.tokenize(query))
        )

        seeds = self.generation_seed_candidates(target, retrieved, question_type)
        generated = []
        seen = set()

        attempts_per_seed = int(settings.get("attempts_per_seed", 3))

        for seed in seeds:
            for _ in range(attempts_per_seed):
                sentence = self.generate_sentence(seed, target_terms, question_type)
                if not sentence:
                    continue

                normalized = self.normalize(sentence)
                if normalized in seen or not self.valid_generation_sentence(sentence):
                    continue

                seen.add(normalized)
                generated.append(sentence)

        # Fallback to more seeds if not enough candidates
        if len(generated) < limit:
            fallback_starts = []
            for tokens in self.sentence_tokens:
                if tokens and set(tokens) & target_terms:
                    fallback_starts.append(tokens[:min(3, len(tokens))])

            random.shuffle(fallback_starts)

            for seed in fallback_starts[:20]:
                sentence = self.generate_sentence(seed, target_terms, question_type)
                if not sentence:
                    continue

                normalized = self.normalize(sentence)
                if normalized in seen or not self.valid_generation_sentence(sentence):
                    continue

                seen.add(normalized)
                generated.append(sentence)
                if len(generated) >= limit:
                    break

        scored = []

        for sentence in generated:
            previous = [item["sentence"] for item in scored]

            score, repetition, parroting = self.score_generated_sentence(
                sentence,
                target_terms,
                question_type,
                target,
                previous
            )

            tokens = self.tokenize(sentence)
            relevance = self.generation_relevance(sentence, target_terms)
            answer_fit = self.generation_answer_fit(
                sentence, question_type, target
            )

            scored.append({
                "sentence": sentence,
                "language": self.language_score(tokens),
                "relevance": relevance,
                "answer_fit": answer_fit,
                "evidence": relevance,
                "repetition": repetition,
                "parroting": parroting,
                "score": score
            })

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:limit]

    # Confidence score for generation candidates
    def generation_confidence(self, candidates):
        if not candidates:
            return 0.0

        scoring = self.config["scoring"]["generation"]
        top = candidates[0]["score"]
        divisor = float(scoring.get("confidence_divisor", 15.0))

        if len(candidates) == 1:
            return min(1.0, max(0.0, top / divisor))

        second = candidates[1]["score"]
        strength = min(1.0, max(0.0, top / divisor))
        separation = min(
            1.0,
            max(0.0, (top - second) / max(abs(top), 1.0))
        )
        evidence = candidates[0]["evidence"]

        return min(
            1.0,
            strength * float(scoring.get("confidence_strength_weight", 0.5))
            + separation * float(scoring.get("confidence_separation_weight", 0.2))
            + evidence * float(scoring.get("confidence_evidence_weight", 0.3))
        )

    # Pick best retrieval answer
    def best_retrieval_answer(self, candidates, question_type):
        if not candidates:
            return None

        # Prefer definition-matching sentences for DEFINITION questions
        if question_type == "DEFINITION":
            definition_candidates = [
                c for c in candidates
                if c["definition_fit"] >= 0.8
            ]
            if definition_candidates:
                return max(
                    definition_candidates,
                    key=lambda x: (
                        x["definition_fit"],
                        x["question_fit"],
                        x["topic_fit"],
                        x["coverage"],
                        x["score"]
                    )
                )

        return candidates[0]

    # Pick best generation answer
    def best_generation_answer(self, candidates):
        return candidates[0] if candidates else None

    # Capitalize and punctuate a sentence
    def sentence_clean(self, sentence):
        sentence = re.sub(r"\s+", " ", sentence).strip()
        if not sentence:
            return ""
        sentence = sentence[0].upper() + sentence[1:]
        if sentence[-1] not in ".!?":
            sentence += "."
        return sentence

    # Print full pipeline diagnostics
    def debug_query(
        self,
        raw,
        normalized,
        corrected_text,
        corrections,
        question_type,
        important,
        corrected,
        expanded,
        target,
        retrieved,
        generated,
        retrieval_confidence,
        generation_confidence,
        decision,
        reason,
        output
    ):
        print()
        print("╔══════════════════════════════════════════════════════════════╗")
        print("║ QUERY ANALYSIS                                               ║")
        print("╚══════════════════════════════════════════════════════════════╝")
        print(f"Raw input:\n  {raw}")
        print(f"Normalized:\n  {normalized}")
        print(f"Dictionary corrected:\n  {corrected_text}")

        if corrections:
            for item in corrections:
                print(
                    f"  {item['from']} -> {item['to']} "
                    f"(distance={item['distance']}, score={item['score']:.3f})"
                )
        else:
            print("  (no corrections)")

        print(f"Question type:\n  {question_type}")
        print(
            "Important tokens:\n  "
            + (" ".join(important) if important else "(none)")
        )
        print(
            "Corrected tokens:\n  "
            + (" ".join(corrected) if corrected else "(none)")
        )
        print(
            "Expanded terms:\n  "
            + (" ".join(sorted(expanded)) if expanded else "(none)")
        )
        print(f"Target:\n  {target}")

        print()
        print("╔══════════════════════════════════════════════════════════════╗")
        print("║ CANDIDATE RETRIEVAL                                          ║")
        print("╚══════════════════════════════════════════════════════════════╝")

        if not retrieved:
            print("No retrieval candidates.")

        for i, item in enumerate(retrieved, 1):
            print()
            print(f"[{i}] SOURCE: {item['source']}")
            print(f"    Sentence ID: {item['index']}")
            print(f"    Sentence: {item['sentence']}")
            print(f"    lexical score: {item['lexical']:.3f}")
            print(f"    coverage:      {item['coverage']:.3f}")
            print(f"    phrase:        {item['phrase']:.3f}")
            print(f"    question fit:  {item['question_fit']:.3f}")
            print(f"    definition fit:{item['definition_fit']:.3f}")
            print(f"    topic fit:     {item['topic_fit']:.3f}")
            print(f"    final score:   {item['score']:.3f}")

        print()
        print(f"Retrieval confidence: {retrieval_confidence:.3f}")

        if not self.no_generation:
            print()
            print("╔══════════════════════════════════════════════════════════════╗")
            print("║ CANDIDATE GENERATION                                         ║")
            print("╚══════════════════════════════════════════════════════════════╝")

            if not generated:
                print("No generation candidates.")

            for i, item in enumerate(generated, 1):
                print()
                print(f"[{i}] {item['sentence']}")
                print(f"    language:       {item['language']:.3f}")
                print(f"    relevance:      {item['relevance']:.3f}")
                print(f"    answer fit:     {item['answer_fit']:.3f}")
                print(f"    evidence:       {item['evidence']:.3f}")
                print(f"    repetition:     {-item['repetition']:.3f}")
                print(f"    parroting:      {item['parroting']:.3f}")
                print(f"    final score:    {item['score']:.3f}")

            print()
            print(f"Generation confidence: {generation_confidence:.3f}")

        print()
        print("╔══════════════════════════════════════════════════════════════╗")
        print("║ ROUTER                                                       ║")
        print("╚══════════════════════════════════════════════════════════════╝")
        print(f"Retrieval confidence: {retrieval_confidence:.3f}")

        if not self.no_generation:
            print(f"Generation confidence: {generation_confidence:.3f}")

        print()
        print(f"Decision: {decision}")
        print(f"Reason: {reason}")
        print()
        print("╔══════════════════════════════════════════════════════════════╗")
        print("║ OUTPUT                                                       ║")
        print("╚══════════════════════════════════════════════════════════════╝")
        print(output)

    # Human-readable expected answer type
    def expected_answer(self, question_type):
        custom = self.config.get("expected_answers", {})
        if question_type in custom:
            return str(custom[question_type])

        mapping = {
            "DEFINITION": "definition / explanation",
            "ADVANTAGE": "advantages / benefits",
            "DISADVANTAGE": "disadvantages / limitations",
            "COMPARISON": "comparison",
            "LIST": "list / examples",
            "WHY": "cause / explanation",
            "HOW": "process / explanation",
            "WHEN": "time / condition",
            "WHERE": "location / context",
            "WHO": "person / entity",
            "QUESTION": "answer",
            "GENERAL": "relevant information"
        }
        return mapping.get(question_type, "relevant information")

    # Full answer pipeline
    def answer(self, raw):
        raw = str(raw)

        # Correct typos before anything else
        corrected_text, corrections = self.correct_text(raw)
        normalized = corrected_text

        tokens = self.tokenize(corrected_text)

        if not tokens:
            return ""

        if normalized in self.goodbyes:
            return "Goodbye."

        if normalized in self.greetings:
            # Prefer knowledge-db greeting if available
            greeting_knowledge = self.retrieve_knowledge(
                corrected_text,
                "GREETING",
                normalized
            )
            if greeting_knowledge:
                return greeting_knowledge[0]["sentence"]
            return "Hello, I am ChatNNL."

        question_type = self.detect_question_type(
            corrected_text,
            tokens
        )

        important = self.important_tokens(tokens)

        corrected = []
        for token in important:
            corrected.append(self.alias_lookup.get(token, token))

        expanded = self.expand_terms(corrected)

        target = self.extract_target(
            corrected_text,
            tokens,
            question_type
        )

        retrieved = self.retrieve(
            corrected_text,
            question_type,
            target
        )

        retrieval_confidence = self.retrieval_confidence(retrieved)

        generated = []

        if not self.no_generation:
            generated = self.generate_candidates(
                corrected_text,
                question_type,
                target,
                retrieved
            )

        generation_confidence = (
            self.generation_confidence(generated)
            if generated else 0.0
        )

        retrieval_answer = self.best_retrieval_answer(
            retrieved,
            question_type
        )
        generation_answer = self.best_generation_answer(generated)

        # Route the answer
        if self.no_generation:
            decision = "RETRIEVAL"
            reason = "Generation disabled by --no-generation."
            output = (
                "[R] " + retrieval_answer["sentence"]
                if retrieval_answer
                else "[R] I could not find enough information in the corpus."
            )

        elif self.only_generate:
            decision = "GENERATION"
            reason = "Generation-only mode enabled by --only-generate."

            if generation_answer:
                output = "[G] " + generation_answer["sentence"]
            elif retrieval_answer:
                output = "[G] " + retrieval_answer["sentence"]
            else:
                output = "[G] I could not generate an answer from the corpus."

        else:
            # DEFINITION questions prefer strong retrieval
            if question_type == "DEFINITION" and retrieval_answer:
                definition_strength = (
                    retrieval_answer["definition_fit"] * 0.45
                    + retrieval_answer["question_fit"] * 0.25
                    + retrieval_answer["topic_fit"] * 0.20
                    + retrieval_answer["coverage"] * 0.10
                )

                if definition_strength >= 0.70 and retrieval_confidence >= 0.45:
                    decision = "RETRIEVAL"
                    reason = (
                        "Strong definition-style retrieval matched "
                        "the requested concept."
                    )
                    output = "[R] " + retrieval_answer["sentence"]

                elif (
                    generation_answer
                    and generation_confidence > retrieval_confidence
                ):
                    decision = "GENERATION"
                    reason = "Generation produced a stronger supported candidate."
                    output = "[G] " + generation_answer["sentence"]

                else:
                    decision = "RETRIEVAL"
                    reason = "Retrieval provided the strongest supported answer."
                    output = "[R] " + retrieval_answer["sentence"]

            elif (
                retrieval_answer
                and generation_answer
                and generation_confidence
                > retrieval_confidence
                + float(
                    self.config["scoring"]["generation"].get(
                        "router_generation_margin", 0.08
                    )
                )
            ):
                decision = "GENERATION"
                reason = "Generation scored higher than retrieval."
                output = "[G] " + generation_answer["sentence"]

            elif retrieval_answer:
                decision = "RETRIEVAL"
                reason = "Retrieval provided stronger corpus-supported evidence."
                output = "[R] " + retrieval_answer["sentence"]

            elif generation_answer:
                decision = "GENERATION"
                reason = (
                    "Retrieval had no usable answer, so statistical "
                    "generation was selected."
                )
                output = "[G] " + generation_answer["sentence"]

            else:
                decision = "NONE"
                reason = "No usable answer candidate was found."
                output = "I could not find enough information in the corpus."

        if self.debug:
            self.debug_query(
                raw,
                normalized,
                corrected_text,
                corrections,
                question_type,
                important,
                corrected,
                expanded,
                target,
                retrieved,
                generated,
                retrieval_confidence,
                generation_confidence,
                decision,
                reason,
                output
            )

        return output

    # Print model info
    def print_info(self):
        print("Project Non")
        print("========================================")
        print(f"ChatNNL Version: {VERSION}")
        print("Preparing statistical language model...")
        print()

        print(f"Corpora loaded: {len(self.corpus_paths)}")

        for path in self.corpus_paths:
            count = 0
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    count = len(self.tokenize(f.read()))
            print(f"  {path}: {count} tokens")

        print()
        print(f"Corpus tokens: {self.total_tokens}")
        print(f"Corpus sentences: {len(self.sentences)}")
        print(f"Vocabulary: {len(self.vocab)}")
        print(f"Dictionary words: {self.dictionary_known_words}")
        print(f"Dictionary corrections this session: {self.dictionary_corrections}")
        print(f"Alias concepts: {self.alias_concepts}")
        print(f"1-token contexts: {len(self.unigrams)}")
        print(f"2-token contexts: {len(self.bigrams)}")
        print(f"3-token contexts: {len(self.trigrams)}")
        print(f"4-token contexts: {len(self.fourgrams)}")
        print(f"5-token contexts: {len(self.fivegrams)}")

        print()
        print("Project Non")
        print("========================================")
        print(f"ChatNNL Version: {VERSION}")
        print("Neural-NetworkLess statistical chatbot")
        print()

        if self.no_generation:
            print("MODE: RETRIEVAL ONLY")
            print("Generation pipeline disabled.")
        elif self.only_generate:
            print("MODE: GENERATION ONLY")
            print("Statistical generation is always selected.")
        else:
            print("MODE: HYBRID")
            print("Retrieval and statistical generation are both enabled.")

        print()
        print("Configuration:")
        print(f"  Aliases: {self.aliases_path or 'built-in fallback / none'}")
        print(f"  Dictionary: {self.dictionary_path or 'built-in defaults'}")
        print(f"  Question fit: {self.question_fit_path or 'built-in defaults'}")
        print(f"  Target extraction: {self.target_extraction_path or 'built-in defaults'}")
        print(
            "  Question classification: "
            f"{self.question_classification_path or 'built-in defaults'}"
        )
        print(f"  Main config: {self.config_path or 'built-in defaults'}")

        print()
        print("Dictionary:")
        print(
            "  Correction: "
            + (
                "enabled"
                if self.config["dictionary"].get("auto_correct", True)
                else "disabled"
            )
        )
        print(
            f"  Max edit distance: "
            f"{self.config['dictionary'].get('max_edit_distance', 2)}"
        )
        print("  Frequency-aware candidate ranking: enabled")
        print("  Corpus vocabulary expansion: " + (
            "enabled"
            if self.config["dictionary"].get("corpus_words_are_valid", True)
            else "disabled"
        ))

        print()
        print("Generation:")
        print("  5-gram -> 4-gram -> trigram -> bigram -> unigram")
        print(
            f"  Temperature: "
            f"{self.config['generation'].get('temperature', 0.75)}"
        )
        print(
            f"  Max tokens: "
            f"{self.config['generation'].get('max_tokens', 32)}"
        )

        if self.debug:
            print()
            print("DEBUG MODE ENABLED")
            print("Full pipeline diagnostics will be displayed.")

        print()

    # Interactive chat loop
    def chat(self):
        print("Type 'bye' or 'exit' to leave.")
        print()

        while True:
            try:
                user = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break

            if not user:
                continue

            if user.lower() in self.goodbyes:
                print("ChatNNL: Goodbye.")
                break

            response = self.answer(user)
            print(f"ChatNNL: {response}")


def parse_args_legacy():
    parser = argparse.ArgumentParser(
        prog="non.py",
        description="Project Non / ChatNNL 0.9"
    )

    parser.add_argument(
        "corpus",
        nargs="+",
        help="One or more corpus text files"
    )

    parser.add_argument(
        "--aliases",
        "-a",
        default=None,
        help="Optional aliases JSON file"
    )

    parser.add_argument(
        "--dictionary",
        "-d",
        default=None,
        help=(
            "Optional dictionary JSON file. "
            "Built-in dictionary is used automatically when omitted."
        )
    )

    parser.add_argument(
        "--question-fit",
        default=None,
        help="Optional question-fit JSON file"
    )

    parser.add_argument(
        "--target-extraction",
        default=None,
        help="Optional target-extraction JSON file"
    )

    parser.add_argument(
        "--question-classification",
        default=None,
        help="Optional question-classification JSON file"
    )

    parser.add_argument(
        "--config",
        default=None,
        help=(
            "Optional master JSON configuration file. "
            "Dedicated JSON files override it."
        )
    )

    mode = parser.add_mutually_exclusive_group()

    mode.add_argument(
        "--no-generation",
        action="store_true",
        help="Disable generation and use retrieval only"
    )

    mode.add_argument(
        "--only-generate",
        action="store_true",
        help="Always use statistical generation"
    )

    parser.add_argument(
        "-D",
        "--debug",
        action="store_true",
        help="Enable full pipeline diagnostics"
    )

    return parser.parse_args()


def main_legacy():
    args = parse_args_legacy()

    try:
        bot = ChatNNL(
            corpus_paths=args.corpus,
            aliases_path=args.aliases,
            question_fit_path=args.question_fit,
            target_extraction_path=args.target_extraction,
            question_classification_path=args.question_classification,
            dictionary_path=args.dictionary,
            config_path=args.config,
            debug=args.debug,
            no_generation=args.no_generation,
            only_generate=args.only_generate
        )

        bot.print_info()
        bot.chat()

    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON: {e}")
        sys.exit(1)

    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    except KeyboardInterrupt:
        print()
        sys.exit(0)


# Developer Edition adds flexible config formats and multiple corpora

DE_VERSION = "Developer Edition"
SUPPORTED_CONFIG_FORMATS = {"json", "yaml", "yml", "xml", "jsonl"}


# Convert a text value into the appropriate scalar type
def _parse_scalar(text):
    text = str(text).strip()
    if not text:
        return ""
    low = text.lower()
    if low in {"true", "yes", "on"}:
        return True
    if low in {"false", "no", "off"}:
        return False
    try:
        if "." in text:
            return float(text)
        return int(text)
    except ValueError:
        return text.strip('"\'')


# Minimal YAML loader, uses PyYAML if available
def _simple_yaml_load(text):
    try:
        import yaml  # type: ignore
        data = yaml.safe_load(text)
        return {} if data is None else data
    except ImportError:
        pass

    lines = []
    for raw in text.splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        body = raw.strip()
        lines.append((indent, body))

    def parse_block(pos, indent):
        if pos >= len(lines):
            return {}, pos
        is_list = lines[pos][0] == indent and lines[pos][1].startswith("-")
        obj = [] if is_list else {}
        while pos < len(lines) and lines[pos][0] == indent:
            body = lines[pos][1]
            if is_list:
                if not body.startswith("-"):
                    break
                item = body[1:].strip()
                pos += 1
                if not item:
                    if pos < len(lines) and lines[pos][0] > indent:
                        child, pos = parse_block(pos, lines[pos][0])
                        obj.append(child)
                    else:
                        obj.append(None)
                elif ":" in item and not item.startswith(('"', "'")):
                    k, v = item.split(":", 1)
                    child = {k.strip(): _parse_scalar(v) if v.strip() else {}}
                    if pos < len(lines) and lines[pos][0] > indent:
                        extra, pos = parse_block(pos, lines[pos][0])
                        if isinstance(extra, dict):
                            child.update(extra)
                    obj.append(child)
                else:
                    obj.append(_parse_scalar(item))
            else:
                if ":" not in body:
                    pos += 1
                    continue
                key, value = body.split(":", 1)
                key = key.strip().strip('"\'')
                value = value.strip()
                pos += 1
                if value:
                    obj[key] = _parse_scalar(value)
                elif pos < len(lines) and lines[pos][0] > indent:
                    child, pos = parse_block(pos, lines[pos][0])
                    obj[key] = child
                else:
                    obj[key] = {}
        return obj, pos

    data, _ = parse_block(0, lines[0][0] if lines else 0)
    return data


# Convert XML tree into a dict
def _xml_to_data(text):
    import xml.etree.ElementTree as ET
    root = ET.fromstring(text)

    def convert(node):
        children = list(node)
        if not children:
            return _parse_scalar(node.text or "")
        grouped = defaultdict(list)
        for child in children:
            grouped[child.tag].append(convert(child))
        out = {}
        for tag, values in grouped.items():
            out[tag] = values if len(values) > 1 else values[0]
        for key, value in node.attrib.items():
            out[key] = value
        return out

    data = convert(root)
    return data if isinstance(data, dict) else {root.tag: data}


# Parse JSONL text into a list of records
def _jsonl_to_data(text):
    rows = []
    for line_no, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at line {line_no}: {exc}")
    return rows


# Load a config file of any supported format
def load_flexible_file(path, required=False):
    if not path:
        return {}
    if not os.path.exists(path):
        if required:
            raise FileNotFoundError(f"Configuration not found: {path}")
        return {}
    suffix = Path(path).suffix.lower().lstrip(".")
    if suffix not in SUPPORTED_CONFIG_FORMATS:
        raise ValueError(
            f"Unsupported configuration format '.{suffix}' for {path}. "
            f"Supported: JSON, YAML, XML, JSONL."
        )
    text = Path(path).read_text(encoding="utf-8")
    if suffix == "json":
        data = json.loads(text)
    elif suffix in {"yaml", "yml"}:
        data = _simple_yaml_load(text)
    elif suffix == "xml":
        data = _xml_to_data(text)
    else:
        data = _jsonl_to_data(text)
    return data


# Ensure loaded config is a mapping
def _as_mapping(data, name):
    if isinstance(data, dict):
        return data
    if isinstance(data, list):
        # Merge object records in order
        merged = {}
        for item in data:
            if isinstance(item, dict):
                merged = deep_merge(merged, item)
        if merged:
            return merged
    raise ValueError(f"{name} configuration must contain an object or object records")


# Format-agnostic JSON loader that accepts multiple files
def load_json_file(path, required=False):
    if not path:
        return {}
    if isinstance(path, (list, tuple)):
        merged = {}
        for item in path:
            data = _as_mapping(load_flexible_file(item, required=required), "Configuration")
            merged = deep_merge(merged, data)
        return merged
    return _as_mapping(load_flexible_file(path, required=required), "Configuration")


# Flatten any path value into a list of strings
def as_paths(value):
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        out = []
        for item in value:
            out.extend(as_paths(item))
        return out
    return [str(value)]


class ChatNNLDE(ChatNNL):
    # Developer Edition: multi-file config, multi-corpus, JSON knowledge db

    def __init__(self, corpus_paths=None, aliases_path=None,
                 question_fit_path=None, target_extraction_path=None,
                 question_classification_path=None, dictionary_path=None,
                 config_path=None, debug=False, no_generation=False,
                 only_generate=False, prompt=None, knowledge_paths=None,
                 typo_mode=None):
        self.prompt = prompt
        self.knowledge_paths = list(knowledge_paths or [])
        self.typo_mode_override = typo_mode

        # Load master config first so corpora can be defined there
        external = _as_mapping(load_flexible_file(config_path, required=True), "Main") if config_path else {}
        self.de_config = external

        # CLI paths override config paths
        aliases_path = as_paths(aliases_path) or as_paths(external.get("aliases"))
        dictionary_path = as_paths(dictionary_path) or as_paths(external.get("dictionary"))
        question_fit_path = as_paths(question_fit_path) or as_paths(external.get("question_fit", external.get("question-fit")))
        target_extraction_path = as_paths(target_extraction_path) or as_paths(external.get("target_extraction", external.get("target-extraction")))
        question_classification_path = as_paths(question_classification_path) or as_paths(external.get("question_classification", external.get("question-classification")))
        if prompt is None:
            prompt = external.get("prompt")
        self.prompt = prompt
        self.de_config = external

        # Corpora can be split into text and knowledge JSON
        config_corpora = external.get("corpora", external.get("corpus", []))
        if isinstance(config_corpora, dict):
            text_cfg = config_corpora.get("text", config_corpora.get("texts", []))
            knowledge_cfg = config_corpora.get("json", config_corpora.get("knowledge", []))
            corpus_paths = as_paths(corpus_paths) + as_paths(text_cfg)
            self.knowledge_paths.extend(as_paths(knowledge_cfg))
        else:
            for path in as_paths(config_corpora):
                if str(path).lower().endswith(".json"):
                    self.knowledge_paths.append(path)
                else:
                    corpus_paths = as_paths(corpus_paths) + [path]

        corpus_paths = as_paths(corpus_paths)
        # JSON positional corpora become knowledge databases
        for path in list(corpus_paths):
            if Path(path).suffix.lower() == ".json":
                self.knowledge_paths.append(path)
        corpus_paths = [p for p in corpus_paths if Path(p).suffix.lower() != ".json"]

        # Typo correction mode
        typo_cfg = external.get("typo_correction", {})
        self.typo_mode = (
            typo_mode
            or typo_cfg.get("mode")
            or "adcm"
        ).lower()
        if self.typo_mode not in {"acm", "dm", "adcm"}:
            raise ValueError("Typo correction mode must be ACM, DM, or ADCM")

        self.knowledge_entries = []
        super().__init__(
            corpus_paths=corpus_paths,
            aliases_path=aliases_path,
            question_fit_path=question_fit_path,
            target_extraction_path=target_extraction_path,
            question_classification_path=question_classification_path,
            dictionary_path=dictionary_path,
            config_path=config_path,
            debug=debug,
            no_generation=no_generation,
            only_generate=only_generate,
        )
        self.load_knowledge_databases()

    # Load and merge multiple alias files
    def load_aliases(self):
        self.aliases = {}
        self.alias_lookup = {}

        paths = as_paths(self.aliases_path)
        if not paths:
            self.alias_concepts = 0
            return

        for path in paths:
            data = load_flexible_file(path, required=True)
            data = _as_mapping(data, "Aliases")
            if "aliases" in data and isinstance(data["aliases"], dict):
                data = data["aliases"]

            for concept, values in data.items():
                concept = str(concept).strip().lower()
                if not concept:
                    continue
                if isinstance(values, str):
                    values = [values]
                if not isinstance(values, list):
                    continue

                # Merge with existing entries for this concept
                existing = set(self.aliases.get(concept, []))
                existing.add(concept)
                for value in values:
                    if isinstance(value, str) and value.strip():
                        existing.add(value.strip().lower())
                self.aliases[concept] = sorted(existing)
                self.alias_lookup[concept] = concept
                for alias in existing:
                    self.alias_lookup[alias] = concept

        self.alias_concepts = len(self.aliases)

    # Find candidate words using edit distance against corpus vocab
    def _algorithmic_candidates(self, token):
        token = self.normalize_dictionary_word(token)
        if not token or len(token) <= 2 or token.isdigit():
            return []
        cfg = self.config.get("dictionary", {})
        max_distance = max(0, min(int(cfg.get("max_edit_distance", 2)), 3))
        vocab = self.vocab
        if not vocab:
            return []
        max_candidates = int(cfg.get("maximum_candidates", 5))
        scored = []
        for candidate, freq in vocab.items():
            if abs(len(candidate) - len(token)) > max_distance:
                continue
            distance = self.edit_distance(token, candidate)
            if distance > max_distance:
                continue
            distance_score = 1.0 - distance / max(1, max_distance + 1)
            frequency_score = math.log1p(freq) / max(1.0, math.log1p(max(vocab.values(), default=1)))
            score = distance_score * 0.85 + frequency_score * 0.15
            scored.append({"word": candidate, "distance": distance,
                           "frequency": freq, "score": score, "known": False})
        scored.sort(key=lambda x: (x["distance"], -x["score"], -x["frequency"]))
        return scored[:max_candidates]

    # Correct tokens using configured typo correction mode
    def correct_tokens(self, tokens):
        corrected, changes = [], []
        for token in tokens:
            if len(token) <= 2 or token.isdigit():
                corrected.append(token)
                continue

            dictionary_best = None
            algorithmic_best = None

            if self.typo_mode in {"dm", "adcm"}:
                candidates = self.dictionary_candidates(token)
                if candidates and candidates[0].get("known"):
                    dictionary_best = candidates[0]
                elif candidates:
                    # Pick best candidate using bigram context
                    best_score = -1.0
                    for candidate in candidates:
                        score = candidate["score"]
                        if corrected:
                            count = self.bigrams.get((corrected[-1], candidate["word"]), 0)
                            score += float(self.config["dictionary"].get("context_bonus", 2.5)) * min(0.25, math.log1p(count) / 20.0) if count else 0
                        if score > best_score:
                            best_score = score
                            dictionary_best = dict(candidate, context_score=score)
                    if dictionary_best and dictionary_best.get("context_score", 0) < float(self.config["dictionary"].get("minimum_score", 0.72)):
                        dictionary_best = None

            if self.typo_mode in {"acm", "adcm"}:
                candidates = self._algorithmic_candidates(token)
                if candidates:
                    algorithmic_best = candidates[0]

            chosen = None
            if self.typo_mode == "dm":
                chosen = dictionary_best
            elif self.typo_mode == "acm":
                chosen = algorithmic_best
            else:
                # Hybrid: dictionary wins exact/protected, else best score wins
                if dictionary_best and dictionary_best.get("known"):
                    chosen = dictionary_best
                elif dictionary_best and algorithmic_best:
                    chosen = dictionary_best if dictionary_best.get("context_score", dictionary_best["score"]) >= algorithmic_best["score"] else algorithmic_best
                else:
                    chosen = dictionary_best or algorithmic_best

            replacement = chosen["word"] if chosen else token
            corrected.append(replacement)
            if replacement != token:
                changes.append({
                    "from": token, "to": replacement,
                    "distance": chosen.get("distance") if chosen else None,
                    "score": chosen.get("context_score", chosen.get("score", 0.0)) if chosen else 0.0,
                    "mode": self.typo_mode.upper()
                })
                self.dictionary_corrections += 1
        return corrected, changes

    # Normalize knowledge records into a flat list
    def _knowledge_records(self, data, source):
        records = []
        if isinstance(data, dict):
            # Handle explicit container keys
            for key in ("knowledge", "entries", "records", "facts", "data"):
                if key in data:
                    return self._knowledge_records(data[key], source)
            for key, value in data.items():
                if isinstance(value, str):
                    records.append({"key": str(key), "text": value, "source": source})
                elif isinstance(value, dict):
                    record = dict(value)
                    record.setdefault("key", str(key))
                    record.setdefault("source", source)
                    records.append(record)
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, str):
                            records.append({"key": str(key), "text": item, "source": source})
                        elif isinstance(item, dict):
                            record = dict(item)
                            record.setdefault("key", str(key))
                            record.setdefault("source", source)
                            records.append(record)
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, str):
                    records.append({"text": item, "source": source})
                elif isinstance(item, dict):
                    record = dict(item)
                    record.setdefault("source", source)
                    records.append(record)
        return records

    # Load JSON knowledge databases
    def load_knowledge_databases(self):
        self.knowledge_entries = []
        for path in self.knowledge_paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Knowledge database not found: {path}")
            data = load_flexible_file(path, required=True)
            self.knowledge_entries.extend(self._knowledge_records(data, path))

    # Extract answer text from a knowledge record
    def _knowledge_text(self, record):
        for key in ("answer", "definition", "text", "content", "description", "value"):
            value = record.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        if "key" in record:
            return str(record["key"])
        return ""

    # Extract lookup key from a knowledge record
    def _knowledge_key(self, record):
        parts = []
        for key in ("key", "topic", "term", "question", "title", "name"):
            value = record.get(key)
            if isinstance(value, str):
                parts.append(value)
        return " ".join(parts)

    # Search knowledge database
    def retrieve_knowledge(self, query, question_type, target, limit=12):
        if not self.knowledge_entries:
            return []
        query_terms = set(self.important_tokens(self.tokenize(query)))
        target_terms = self.target_terms(target)
        terms = query_terms | target_terms
        results = []
        for idx, record in enumerate(self.knowledge_entries):
            text = self._knowledge_text(record)
            key = self._knowledge_key(record)
            combined = f"{key} {text}".strip()
            tokens = self.tokenize(combined)
            if not tokens:
                continue
            overlap = len(set(tokens) & terms)
            if not overlap:
                continue
            key_overlap = len(set(self.tokenize(key)) & target_terms)
            exact_target = 1.0 if target and target.lower().strip() in key.lower() else 0.0
            definition = 1.0 if question_type == "DEFINITION" and re.search(r"\b(is|are|means|refers to|defined as)\b", text.lower()) else 0.0
            score = overlap * 3.0 + key_overlap * 5.0 + exact_target * 12.0 + definition * 8.0
            results.append({
                "index": idx, "sentence": self.sentence_clean(text),
                "source": record.get("source", "knowledge"),
                "lexical": float(overlap), "coverage": min(1.0, overlap / max(1, len(terms))),
                "phrase": 1.0 if target and target.lower() in combined.lower() else 0.0,
                "question_fit": 1.0 if question_type == "DEFINITION" and definition else 0.2,
                "definition_fit": definition,
                "topic_fit": min(1.0, key_overlap / max(1, min(3, len(target_terms)))),
                "score": score, "knowledge": True
            })
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:limit]

    # Retrieve from knowledge db first, then corpus
    def retrieve(self, query, question_type, target, limit=12):
        knowledge = self.retrieve_knowledge(query, question_type, target, limit)
        if knowledge:
            return knowledge
        candidates = super().retrieve(query, question_type, target, limit)
        # Boost direct definition matches in DEFINITION questions
        if question_type == "DEFINITION" and target:
            target_low = target.lower().strip()
            for item in candidates:
                sentence_low = item["sentence"].lower()
                if target_low and target_low in sentence_low:
                    item["score"] += 6.0
                    item["phrase"] = max(item["phrase"], 1.0)
                if re.search(rf"\b{re.escape(target_low)}\b\s+(is|are|means|refers to|is a|is an)\b", sentence_low):
                    item["definition_fit"] = 1.0
                    item["score"] += 8.0
            candidates.sort(key=lambda x: x["score"], reverse=True)
        return candidates

    # Zero lexical match should never look confident
    def retrieval_confidence(self, candidates):
        if not candidates:
            return 0.0
        if all(float(c.get("lexical", 0.0)) <= 0 for c in candidates[:3]):
            return 0.0
        return super().retrieval_confidence(candidates)

    # Prefer strong definition matches
    def best_retrieval_answer(self, candidates, question_type):
        if not candidates:
            return None
        if question_type == "DEFINITION":
            strong = [c for c in candidates if c.get("definition_fit", 0) >= 0.8]
            if strong:
                return max(strong, key=lambda x: (
                    x.get("phrase", 0), x.get("topic_fit", 0),
                    x.get("coverage", 0), x.get("score", 0)
                ))
        return candidates[0]

    # Skip generation if retrieval had zero relevance
    def generate_candidates(self, query, question_type, target, retrieved, limit=None):
        if not retrieved:
            return []
        if all(float(item.get("lexical", 0.0)) <= 0 for item in retrieved[:3]):
            return []
        return super().generate_candidates(query, question_type, target, retrieved, limit)

    # Extended info for DE
    def print_info(self):
        super().print_info()
        print(f"  Developer Edition: {DE_VERSION}")
        print(f"  Alias files: {len(as_paths(self.aliases_path))}")
        print(f"  Dictionary files: {len(as_paths(self.dictionary_path))}")
        print(f"  Knowledge databases: {len(self.knowledge_paths)}")
        print(f"  Knowledge entries: {len(self.knowledge_entries)}")
        print(f"  Typo correction mode: {self.typo_mode.upper()}")
        print("  Config formats: JSON / YAML / XML / JSONL")
        print("  Text + JSON corpora: enabled")
        if self.prompt is not None:
            print("  Prompt mode: enabled")


def parse_args():
    parser = argparse.ArgumentParser(
        prog="non-de.py",
        description="Project Non / ChatNNL Developer Edition"
    )
    parser.add_argument(
        "corpus", nargs="*",
        help="Text corpus files and/or JSON knowledge databases. Config may also provide corpora."
    )
    file_help = "Optional file. Supported formats: JSON, YAML, XML, JSONL."
    parser.add_argument("--aliases", "-a", action="append", default=None, help="Alias file. Repeat for multiple alias files. Supported: JSON, YAML, XML, JSONL.")
    parser.add_argument("--dictionary", "-d", action="append", default=None, help="Dictionary file. Repeat for multiple dictionary files. Supported: JSON, YAML, XML, JSONL.")
    parser.add_argument("--question-fit", default=None, help=file_help)
    parser.add_argument("--target-extraction", default=None, help=file_help)
    parser.add_argument("--question-classification", default=None, help=file_help)
    parser.add_argument("--config", default=None, help="Master configuration: JSON, YAML, XML, or JSONL.")
    parser.add_argument("--prompt", default=None, help="Answer one prompt and exit instead of entering chat mode.")
    parser.add_argument(
        "--api-friendly",
        action="store_true",
        help="Return only the final answer. Requires a prompt and disables interactive mode and diagnostics."
    )

    typo = parser.add_mutually_exclusive_group()
    typo.add_argument("--algorithmic-correction", "--typo-ac", dest="typo_mode", action="store_const", const="acm", help="Algorithmic Correction Mode (ACM)")
    typo.add_argument("--dictionary-correction", "--typo-dc", dest="typo_mode", action="store_const", const="dm", help="Dictionary Mode (DM)")
    typo.add_argument("--algorithmic-dictionary-correction", "--typo-adc", dest="typo_mode", action="store_const", const="adcm", help="Algorithmic Dictionary Correction Mode (ADCM)")

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--no-generation", action="store_true", help="Disable generation and use retrieval only")
    mode.add_argument("--only-generate", action="store_true", help="Always use statistical generation")
    parser.add_argument("-D", "--debug", action="store_true", help="Enable full pipeline diagnostics")
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        bot = ChatNNLDE(
            corpus_paths=args.corpus,
            aliases_path=args.aliases,
            question_fit_path=args.question_fit,
            target_extraction_path=args.target_extraction,
            question_classification_path=args.question_classification,
            dictionary_path=args.dictionary,
            config_path=args.config,
            debug=args.debug,
            no_generation=args.no_generation,
            only_generate=args.only_generate,
            prompt=args.prompt,
            typo_mode=args.typo_mode,
        )
        effective_prompt = args.prompt if args.prompt is not None else bot.prompt

        # API-friendly mode prints only the answer text
        if args.api_friendly:
            if effective_prompt is None:
                raise ValueError("--api-friendly requires --prompt or a prompt in the config")
            response = bot.answer(effective_prompt)
            if response.startswith("[R] ") or response.startswith("[G] "):
                response = response[4:]
            print(response)
            return

        bot.print_info()
        if effective_prompt is not None:
            response = bot.answer(effective_prompt)
            print(f"ChatNNL: {response}")
        else:
            bot.chat()
    except FileNotFoundError as exc:
        print(f"Error: {exc}")
        sys.exit(1)
    except (json.JSONDecodeError, ValueError) as exc:
        print(f"Error: {exc}")
        sys.exit(1)
    except KeyboardInterrupt:
        print()
        sys.exit(0)


if __name__ == "__main__":
    main()