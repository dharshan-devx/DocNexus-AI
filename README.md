# DocNexus AI

**DocNexus AI** is a multi-document Retrieval-Augmented Generation (RAG) system built to query, compare, and synthesize insights across independent documents. Unlike traditional RAG pipelines that treat a knowledge base as a single source, this project is designed to handle cross-document reasoning with document-level isolation and contextual retrieval.

The primary use case is analyzing financial 10-K reports, where questions may require comparing data points across multiple companies. For example, a query such as:

> "How much are Apple and AMD investing in R&D?"

requires retrieving relevant information from each report independently before generating a grounded response.

---

## Features

- Query multiple documents simultaneously
- Perform cross-document comparison and synthesis
- Retrieve relevant passages using vector search
- Generate grounded answers using an LLM
- Support financial analysis workflows such as 10-K report comparison
- End-to-end notebook-based implementation for experimentation and reproducibility

---

## Tech Stack

- **LangChain** — orchestration and RAG pipeline framework
- **Elasticsearch** — vector storage and retrieval
- **OpenAI GPT-3.5** — response generation
- **Python** — core implementation

---

## How It Works

1. Documents are ingested and split into chunks.
2. Chunks are embedded and indexed in Elasticsearch.
3. The system retrieves relevant information from multiple documents independently.
4. Retrieved context is passed to the language model.
5. The model generates a final response grounded in the retrieved sources.

---

## Usage

To get started, open the notebook:

- [`blog_nb.ipynb`](blog_nb.ipynb)

This notebook contains the end-to-end implementation of the RAG pipeline, including document processing, retrieval, and generation.

The [`experiments`](experiments) directory contains development scripts and prototype code used during early experimentation.

---

## Project Structure

```bash
.
├── blog_nb.ipynb        # End-to-end implementation notebook
├── experiments/         # Experimental and prototype scripts
├── requirements.txt     # Project dependencies
└── README.md            # Project documentation
