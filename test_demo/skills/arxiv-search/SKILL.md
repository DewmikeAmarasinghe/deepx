---
name: arxiv-search
description: Search arXiv preprint repository for papers in physics, mathematics, computer science, quantitative biology, and related fields
---

# arXiv Search Skill

This skill provides access to arXiv, a free distribution service and open-access archive for scholarly articles in physics, mathematics, computer science, quantitative biology, quantitative finance, statistics, electrical engineering, systems science, and economics.

## When to Use This Skill

Use this skill when you need to:
- Find preprints and recent research papers before journal publication
- Search for papers in computational biology, bioinformatics, or systems biology
- Access mathematical or statistical methods papers relevant to biology
- Find machine learning papers applied to biological problems
- Get the latest research that may not yet be in PubMed

## How to Use

The skill provides a Python script that searches arXiv and returns formatted results.

### Basic Usage

**Note:** Always use the absolute path from your skills directory (shown in the system prompt above).

Use the **absolute directory** for this skill from your system prompt (the row for `arxiv-search`
points at `SKILL.md`; the script is `arxiv_search.py` in that same folder). With a venv and the
`arxiv` package installed:
```bash
.venv/bin/python [SKILLS_DIR]/arxiv-search/arxiv_search.py "your search query" [--max-papers N]
```

Or with system Python:
```bash
python3 [SKILLS_DIR]/arxiv-search/arxiv_search.py "your search query" [--max-papers N]
```

Replace `[SKILLS_DIR]/arxiv-search` with the real path from the prompt.

**Arguments:**
- `query` (required): The search query string (e.g., "neural networks protein structure", "single cell RNA-seq")
- `--max-papers` (optional): Maximum number of papers to retrieve (default: 10)

### Examples

Search for machine learning papers:
```bash
.venv/bin/python [SKILLS_DIR]/arxiv-search/arxiv_search.py "deep learning drug discovery" --max-papers 5
```

Search for computational biology papers:
```bash
.venv/bin/python [SKILLS_DIR]/arxiv-search/arxiv_search.py "protein folding prediction"
```

Search for bioinformatics methods:
```bash
.venv/bin/python [SKILLS_DIR]/arxiv-search/arxiv_search.py "genome assembly algorithms"
```

## Output Format

The script returns formatted results with:
- **Title**: Paper title
- **Summary**: Abstract/summary text

Each paper is separated by blank lines for readability.

## Features

- **Relevance sorting**: Results ordered by relevance to query
- **Fast retrieval**: Direct API access with no authentication required
- **Simple interface**: Clean, easy-to-parse output
- **No API key required**: Free access to arXiv database

## Dependencies

This skill requires the `arxiv` Python package. The script will detect if it's missing and show an error.

**If you see "Error: arxiv package not installed":**

If using the project virtual environment (recommended), use the venv's Python:
```bash
.venv/bin/python -m pip install arxiv
```

Or for system-wide install:
```bash
python3 -m pip install arxiv
```

The package is not installed with Deepx core; install on-demand when first using this skill.

## Notes

- arXiv is particularly strong for:
  - Computer science (cs.LG, cs.AI, cs.CV)
  - Quantitative biology (q-bio)
  - Statistics (stat.ML)
  - Physics and mathematics
- Papers are preprints and may not be peer-reviewed
- Results include both recent uploads and older papers
- Best for computational/theoretical work in biology
