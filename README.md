# ChatNNL Developer Edition

**A dependency-free, statistical chatbot engine with no neural networks — pure n-gram language modeling, edit-distance typo correction, and JSON knowledge databases.**

[![Version](https://img.shields.io/badge/version-DE-blue)](https://github.com/Darky-Github/ChatNNL)
[![Python](https://img.shields.io/badge/python-3.8%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![No Dependencies](https://img.shields.io/badge/dependencies-none-brightgreen)]()
[![Config Formats](https://img.shields.io/badge/config-JSON%20%7C%20YAML%20%7C%20XML%20%7C%20JSONL-orange)]()

ChatNNL (Chat Neural-Network-Less) is a statistical chatbot that builds language models directly from text corpora using n-gram statistics — bigrams, trigrams, fourgrams, and fivegrams. It retrieves relevant sentences from the corpus, generates new sentences via weighted probabilistic sampling, and routes between retrieval and generation based on confidence scores.

The **Developer Edition** adds flexible configuration formats, multi-file support, JSON knowledge databases, three typo-correction modes, and API-friendly output — all with zero external dependencies.

---

## Table of Contents

- [Why ChatNNL?](#why-chatnnl)
- [Features](#features)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Usage](#usage)
  - [Interactive Chat](#interactive-chat)
  - [Single Prompt](#single-prompt)
  - [API-Friendly Mode](#api-friendly-mode)
  - [Debug Mode](#debug-mode)
- [Configuration](#configuration)
  - [Configuration Formats](#configuration-formats)
  - [Master Config File](#master-config-file)
  - [Corpora Setup](#corpora-setup)
- [Typo Correction Modes](#typo-correction-modes)
- [Knowledge Databases](#knowledge-databases)
- [Architecture](#architecture)
- [CLI Reference](#cli-reference)
- [Examples](#examples)
- [What It Doesn't Do](#what-it-doesnt-do)
- [Contributing](#contributing)
- [License](#license)

---

## Why ChatNNL?

Most chatbots require massive neural networks, GPU clusters, and gigabytes of model weights. ChatNNL takes a different approach:

- **No neural networks.** Pure statistical n-gram modeling — the model is built in memory from your corpus at startup.
- **No dependencies.** Python standard library only. No PyTorch, no TensorFlow, no NumPy.
- **No GPU.** Runs on any machine, from a Raspberry Pi to a cloud VM.
- **Fully transparent.** Every scoring decision, every n-gram match, every confidence value is visible in debug mode.
- **Yours to configure.** Question classification, target extraction, scoring weights, dictionary words, and typo correction behavior are all driven by configuration files.

It is ideal for domain-specific FAQ bots, documentation assistants, offline knowledge retrieval, and educational exploration of statistical NLP.

---

## Features

| Feature | Description |
|---|---|
| **Hybrid Retrieval + Generation** | Retrieves corpus sentences with TF-IDF-style scoring; generates new sentences via 5-gram → 4-gram → trigram → bigram fallback. |
| **N-Gram Language Model** | Builds bigram, trigram, fourgram, and fivegram statistics from any text corpus. |
| **Edit-Distance Typo Correction** | Three modes: Algorithmic (ACM), Dictionary (DM), and Hybrid (ADCM). Damerau-Levenshtein distance with frequency and context bonuses. |
| **JSON Knowledge Databases** | Feed structured `question → answer` JSON files directly. Keys and aliases are matched alongside corpus retrieval. |
| **Multi-Format Configuration** | Master config and all section files accept JSON, YAML, XML, and JSONL. Multiple files merge automatically. |
| **Alias Expansion** | Map multiple surface forms to a single concept for better retrieval. Supports multi-word aliases. |
| **Question Classification** | Detects DEFINITION, ADVANTAGE, DISADVANTAGE, COMPARISON, LIST, WHY, HOW, WHEN, WHERE, WHO, and more. |
| **Target Extraction** | Strips question prefixes ("what is ", "define ", etc.) and category words to isolate the topic. |
| **Confidence-Based Routing** | Scores retrieval and generation independently; routes to whichever is stronger. Configurable margin. |
| **Debug Diagnostics** | Full pipeline dump: every retrieval candidate, every generation candidate, all scores, and routing decision. |
| **API-Friendly Output** | `--api-friendly --prompt "..."` prints only the answer — no banners, no debug output. |
| **Zero External Dependencies** | Pure Python 3.8+. No pip install required. |

---

## Quick Start

```bash
# Clone the repo
git clone https://github.com/Darky-Github/ChatNNL.git
cd ChatNNL

# Run with a text corpus
python non-de.py corpus.txt

# Run with a corpus and a JSON knowledge database
python non-de.py corpus.txt knowledge.json

# Run with a master config
python non-de.py --config config.yaml
```

No installation step. No dependencies. Just Python 3.8 or later.

---

## Installation

### Requirements

- Python 3.8 or later
- No external packages

### Setup

```bash
git clone https://github.com/Darky-Github/ChatNNL.git
cd ChatNNL
```

That's it. Run `python non-de.py --help` to verify.

---

## Usage

### Interactive Chat

```bash
python non-de.py corpus.txt
```

```
Project Non
========================================
ChatNNL Version: DE
Preparing statistical language model...

Corpora loaded: 1
  corpus.txt: 45230 tokens

Corpus tokens: 45230
Corpus sentences: 3120
Vocabulary: 4891
...
Type 'bye' or 'exit' to leave.

You: what is machine learning
ChatNNL: [R] Machine learning is a subset of artificial intelligence that enables systems to learn from data.
```

### Single Prompt

```bash
python non-de.py corpus.txt --prompt "what is machine learning"
```

Outputs the full pipeline info plus the answer, then exits.

### API-Friendly Mode

```bash
python non-de.py corpus.txt --api-friendly --prompt "what is machine learning"
```

Outputs **only** the answer text — no banners, no info, no debug. Ideal for piping into other programs.

```bash
# Example: use in a shell script
ANSWER=$(python non-de.py corpus.txt --api-friendly --prompt "what is ML")
echo "$ANSWER"
```

### Debug Mode

```bash
python non-de.py corpus.txt -D --prompt "what is machine learning"
```

Prints the full diagnostic pipeline: corrected tokens, expanded terms, target, every retrieval candidate with individual scores, every generation candidate with individual scores, retrieval confidence, generation confidence, and the routing decision with reason.

```
╔══════════════════════════════════════════════════════════════╗
║ QUERY ANALYSIS                                               ║
╚══════════════════════════════════════════════════════════════╝
Raw input:
  what is machine learning
Normalized:
  what is machine learning
Dictionary corrected:
  what is machine learning
  (no corrections)
Question type:
  DEFINITION
...
```

---

## Configuration

ChatNNL DE loads configuration from multiple sources, with clear precedence:

1. **Built-in defaults** (always loaded)
2. **Master config file** (`--config`) — overrides defaults
3. **Dedicated section files** (`--question-fit`, `--dictionary`, etc.) — override master config
4. **CLI arguments** — override everything

### Configuration Formats

Every configuration file supports four formats:

| Format | Extension | Notes |
|---|---|---|
| JSON | `.json` | Standard JSON. Multiple files merge in order. |
| YAML | `.yaml`, `.yml` | Uses PyYAML if installed; falls back to a built-in subset parser. |
| XML | `.xml` | Attribute values become strings; nested tags become nested objects. |
| JSONL | `.jsonl` | One JSON object per line. Objects merge in order. |

### Master Config File

A single master config can define everything — corpora, section files, dictionary words, question rules, generation settings, and the prompt:

```yaml
# config.yaml
corpora:
  text:
    - docs/faq.txt
    - docs/manual.txt
  json:
    - knowledge/base.json
    - knowledge/extra.json

aliases: aliases.yaml
dictionary:
  - dictionary.json
  - domain-terms.json
question_fit: question-fit.yaml
target_extraction: target-extraction.json

prompt: "what is ChatNNL"

generation:
  temperature: 0.65
  max_tokens: 40
  min_tokens: 6

scoring:
  retrieval:
    coverage_weight: 9.0
    phrase_weight: 6.0
  generation:
    relevance_weight: 10.0
    answer_fit_weight: 9.0

typo_correction:
  mode: adcm
```

### Corpora Setup

Text corpora are plain text files. JSON corpora are treated as knowledge databases.

```
docs/faq.txt          → text corpus
docs/manual.txt       → text corpus
knowledge/base.json   → JSON knowledge database
knowledge/extra.json  → JSON knowledge database
```

A JSON knowledge database is any JSON file where keys or `key`/`topic`/`term` fields map to answer text:

```json
{
  "machine learning": "Machine learning is a subset of AI that enables systems to learn from data.",
  "neural network": {
    "answer": "A neural network is a computing system inspired by biological neural networks.",
    "definition": "A computing system inspired by biological neural networks."
  },
  "python": {
    "answer": "Python is a high-level, interpreted programming language."
  }
}
```

Or a list of records:

```json
[
  {"key": "machine learning", "answer": "ML is a subset of AI..."},
  {"key": "neural network", "definition": "A computing system..."},
  {"key": "python", "text": "Python is a programming language..."}
]
```

---

## Typo Correction Modes

ChatNNL DE offers three typo-correction modes, selectable via CLI or config:

| Mode | Flag | Description |
|---|---|---|
| **ACM** | `--algorithmic-correction` | Checks every token against the full corpus vocabulary using Damerau-Levenshtein distance. Slower but handles domain-specific words not in the dictionary. |
| **DM** | `--dictionary-correction` | Checks only against the built-in/provided dictionary. Fast, conservative, and uses bigram context to pick the best candidate. |
| **ADCM** | `--algorithmic-dictionary-correction` | Hybrid: dictionary wins exact and protected words, algorithmic wins when its score is higher. **Default mode.** |

```bash
python non-de.py corpus.txt --algorithmic-correction
python non-de.py corpus.txt --dictionary-correction
python non-de.py corpus.txt --algorithmic-dictionary-correction
```

The dictionary contains ~150 common English words and a `protected_words` list (Python, JavaScript, JSON, GGUF, etc.). The `corpus_words_are_valid` setting (default `true`) automatically adds every word in your corpus to the dictionary so domain terms are never "corrected".

---

## Knowledge Databases

JSON knowledge databases are first-class citizens. When a knowledge entry matches the query — by key, topic, term, question, title, or name — it is returned **before** any corpus retrieval. This makes it trivial to build a structured FAQ on top of a text corpus:

```bash
python non-de.py docs/faq.txt knowledge/base.json knowledge/extra.json
```

Knowledge entries are scored with:

- **Overlap** between query terms and entry terms
- **Key overlap** between the target and the entry key
- **Exact target match** (target appears literally in the key)
- **Definition pattern** (for DEFINITION questions)

The `retrieve_knowledge` method returns the top matches; if none match, the system falls back to corpus retrieval.

---

## Architecture

```
non-de.py
├── ChatNNL (base engine)
│   ├── tokenize()                  — lowercase word/number tokenization
│   ├── split_sentences()           — sentence boundary detection
│   ├── build_dictionary()          — dictionary words + protected words
│   ├── rebuild_dictionary_index()  — symmetric-delete index for fast lookup
│   ├── edit_distance()             — Damerau-Levenshtein
│   ├── dictionary_candidates()     — frequency-aware candidate ranking
│   ├── correct_tokens()            — dictionary-based correction
│   ├── load_aliases()              — concept → surface forms
│   ├── load_corpora()              — text file ingestion
│   ├── build_statistics()          — n-gram counters + IDF
│   ├── detect_question_type()      — rule-based classification
│   ├── extract_target()            — prefix stripping + stopword removal
│   ├── retrieve()                  — TF-IDF-style corpus retrieval
│   ├── generation_next_candidates()— 5-gram → unigram fallback
│   ├── generate_sentence()         — seeded probabilistic generation
│   ├── score_generated_sentence()  — language + relevance + answer fit
│   ├── answer()                    — full pipeline + routing
│   └── chat()                      — interactive loop
│
└── ChatNNLDE (Developer Edition)
    ├── load_flexible_file()        — JSON/YAML/XML/JSONL loader
    ├── _algorithmic_candidates()   — corpus-vocabulary correction
    ├── correct_tokens()            — ACM/DM/ADCM mode dispatcher
    ├── load_knowledge_databases()  — JSON knowledge ingestion
    ├── retrieve_knowledge()        — knowledge DB scoring
    ├── retrieve()                  — knowledge-first + corpus fallback
    └── print_info()                — extended DE info
```

### N-Gram Hierarchy

Generation always uses the longest matching context:

```
5 tokens → 5-gram next-token distribution
4 tokens → 4-gram next-token distribution
3 tokens → trigram next-token distribution
2 tokens → bigram next-token distribution
1 token  → unigram distribution
```

If a 5-gram context is not found, it falls back to 4-gram, then trigram, then bigram, then unigram. This ensures generation never stalls.

### Routing Logic

The router compares retrieval confidence and generation confidence:

- **DEFINITION questions** prefer strong retrieval (definition fit ≥ 0.70 and retrieval confidence ≥ 0.45) → `[R]`
- **Generation** is selected when `generation_confidence > retrieval_confidence + margin` (default margin 0.08)
- **Retrieval** is selected when it has stronger corpus-supported evidence
- **Generation-only** mode (`--only-generate`) always uses generation
- **Retrieval-only** mode (`--no-generation`) always uses retrieval

---

## CLI Reference

```
usage: non-de.py [-h] [--aliases ALIASES] [--dictionary DICTIONARY]
                 [--question-fit QUESTION_FIT]
                 [--target-extraction TARGET_EXTRACTION]
                 [--question-classification QUESTION_CLASSIFICATION]
                 [--config CONFIG] [--prompt PROMPT] [--api-friendly]
                 [--algorithmic-correction | --dictionary-correction | --algorithmic-dictionary-correction]
                 [--no-generation | --only-generate] [-D]
                 [corpus ...]
```

| Argument | Description |
|---|---|
| `corpus` | Text corpus files and/or JSON knowledge databases. Config may also provide corpora. |
| `--aliases`, `-a` | Alias file. Repeat for multiple files. |
| `--dictionary`, `-d` | Dictionary file. Repeat for multiple files. |
| `--question-fit` | Question-fit patterns file. |
| `--target-extraction` | Target-extraction rules file. |
| `--question-classification` | Question-classification rules file. |
| `--config` | Master configuration file (JSON, YAML, XML, JSONL). |
| `--prompt` | Answer one prompt and exit. |
| `--api-friendly` | Print only the final answer. Requires a prompt. |
| `--algorithmic-correction` | Algorithmic Correction Mode (ACM). |
| `--dictionary-correction` | Dictionary Mode (DM). |
| `--algorithmic-dictionary-correction` | Algorithmic Dictionary Correction Mode (ADCM). Default. |
| `--no-generation` | Disable generation; retrieval only. |
| `--only-generate` | Always use statistical generation. |
| `-D`, `--debug` | Full pipeline diagnostics. |

---

## Examples

### Domain FAQ Bot

```bash
python non-de.py docs/faq.txt knowledge/faq.json --api-friendly --prompt "what is your return policy"
```

### Documentation Assistant

```bash
python non-de.py docs/python-docs.txt --prompt "how do I read a file in python"
```

### Dictionary-Mode Typo Correction

```bash
python non-de.py corpus.txt --dictionary-correction --prompt "what is machien learning"
```

### Generation-Only Mode

```bash
python non-de.py corpus.txt --only-generate --prompt "explain machine learning"
```

### Full Debug Trace

```bash
python non-de.py corpus.txt -D --prompt "advantages of using Python"
```

---

## What It Doesn't Do

ChatNNL is honest about its boundaries:

- **No deep semantic understanding.** It matches surface forms and n-gram statistics. It does not "understand" meaning.
- **No reasoning or inference.** It retrieves and generates from patterns in the corpus. It cannot deduce new facts.
- **No multilingual support out of the box.** Tokenization is ASCII-oriented (`[a-z0-9]`). Non-Latin scripts require tokenizer modification.
- **No persistent model.** The n-gram model is rebuilt from scratch at every startup. There is no saved model file.
- **No streaming or incremental learning.** The corpus is read once at init. New documents require a restart.
- **No neural embeddings.** There are no vectors, no attention, no transformers. Everything is count-based.

---

## Contributing

Contributions are welcome. Before opening a PR:

1. **Keep it dependency-free.** No external packages. Python standard library only.
2. **Test with a real corpus.** Use a text file and verify retrieval, generation, and typo correction all work.
3. **Update the README** if you add a CLI flag, config key, or feature.
4. **Run the debug pipeline** (`-D`) on a few queries and confirm nothing crashes.

```bash
# Fork the repo, create a branch
git checkout -b feature/my-feature

# Make changes, test
python non-de.py test-corpus.txt -D --prompt "test query"

# Commit and push
git commit -m "Add my feature"
git push origin feature/my-feature
```

Open a pull request with a clear description of what changed and why.

---

## License

This project is licensed under the **GNU General Public License v3.0**.

```
ChatNNL Developer Edition — a dependency-free statistical chatbot engine.
Copyright (C) 2024 Darky-Github

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
```

See the full [LICENSE](LICENSE) file, or read the official text at
<https://www.gnu.org/licenses/gpl-3.0.html>.

---

## Acknowledgments

- Built on the original ChatNNL 0.9 statistical engine.
- N-gram language modeling inspired by classical information retrieval and statistical NLP.
- Typo correction uses Damerau-Levenshtein edit distance with symmetric-delete indexing.
- README structure follows popular GitHub templates and best practices.
