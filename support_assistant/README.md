# Zepto Support Assistant

## Overview

This module implements a small RAG-based support assistant for Zepto policies.

The system:

1. Loads Zepto policy documents.
2. Chunks and embeds the documents locally.
3. Stores the embeddings in ChromaDB.
4. Uses LangGraph to classify incoming questions.
5. Retrieves relevant policy context for policy questions.
6. Generates a deterministic mock answer in the required offline mode.
7. Validates the final response using a Pydantic schema.
8. Exposes the assistant through a FastAPI `POST /ask` endpoint.

---

## Architecture

```text
Policy Documents
      |
      v
   Ingestion
      |
      |  support_assistant/ingest.py
      v
Chunking
      |
      v
Sentence Transformers
all-MiniLM-L6-v2
      |
      v
   ChromaDB
      |
      |-----------------------------|
                                    |
User Query                          |
      |                             |
      v                             |
FastAPI POST /ask                   |
      |                             |
      v                             |
LangGraph StateGraph                |
      |                             |
      v                             |
classify_intent                     |
      |                             |
      +----------+------------------+
                 |
        +--------+--------+
        |                 |
policy_question     general_question
        |                 |
        v                 v
retrieve_and_answer   direct_answer
        |                 |
        v                 |
   ChromaDB retrieval    |
        |                 |
        +--------+--------+
                 |
                 v
          Pydantic validation
                 |
                 v
          JSON response


## Pipeline Stages

### 1. Ingestion

The policy corpus is stored in:

```text
support_assistant/docs/

### 2. Embedding

Embeddings are generated locally using the
`all-MiniLM-L6-v2` sentence-transformers model.

No LLM API key is required for embeddings.

The vectors are stored in the ChromaDB collection located under:

```text
support_assistant/chroma_db/
```
### 3. Retrieval

For `policy_question` queries, the `retrieve_and_answer` LangGraph node embeds the incoming query and retrieves the top 3 most similar chunks from ChromaDB.

The retrieval step runs in both mock and real-LLM modes because embedding and ChromaDB retrieval do not require an external LLM API.

The retrieved chunk IDs are stored in the `sources` field of the final response.

### 4. Generation

The `retrieve_and_answer` node generates the final answer using the retrieved policy context.

In the default mock mode, the answer is generated deterministically from the most similar retrieved chunk using the format:

```text
Based on the retrieved context: ...
```
```text
MOCK_LLM=0
```

### 5. Validation

The final response is validated using a Pydantic model before being returned by the API.

The response schema contains:

- `answer`: Final assistant response.
- `sources`: Retrieved document/chunk IDs.
- `confidence`: Retrieval confidence score.

Validation ensures that the API always returns a response matching the expected structure.

### 6. API

The assistant is exposed through a FastAPI application.

The main endpoint is:

```text
POST /ask

```markdown
Example request:

```json
{
  "query": "What is Zepto delivery time?"
}



### Important

**Do not modify `graph.py`.**  
**Do not modify `main.py`.**  
**Do not rerun the tests.**

