# Multi-Doc RAG

## About

This application enables Retrieval-Augmented Generation (RAG) across multiple documents simultaneously. While standard RAG implementations often process a single document or treat an entire knowledge base as a monolithic source, this solution is specifically designed to isolate and synthesize information across distinct documents.

The primary use case is querying and comparing financial 10-K reports. For example, a query such as "How much are Apple and AMD investing in R&D?" requires retrieving and comparing specific data points from both the Apple and AMD 10-K reports independently. This application dynamically handles such multi-document inquiries.

## Technologies Used

- **LangChain**: Orchestration and framework
- **Elasticsearch**: Vector storage and retrieval
- **OpenAI (GPT-3.5)**: Large Language Model for generation

## Usage

Please refer to the [getting started notebook](blog_nb.ipynb) for an end-to-end implementation and demonstration of the RAG pipeline.

The [`experiments`](experiments) directory contains developmental code and preliminary research scripts used during the initial implementation phase.

## Documentation

A detailed explanation of the architecture and implementation is available in the accompanying blog post at [datascience.fm](https://datascience.fm/multi-doc-rag-on-10k-reports/).

**Note:** The dependencies listed in the blog post are sufficient to run the [notebook](blog_nb.ipynb). The included `requirements.txt` file contains additional developmental dependencies that may not be required for standard usage.
