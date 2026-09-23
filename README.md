# Zepto Data & AI Platform

An end-to-end AI/ML capstone project combining data engineering, data analytics, predictive modeling, and a Generative AI support assistant into one connected platform.

## Project Overview

This project consists of three integrated modules:

1. **Data Pipeline** – Scrapes, cleans, transforms, and stores product data in a normalized SQLite database.
2. **Analytics Pipeline** – Performs exploratory data analysis, data cleaning, visualization, predictive modeling, model evaluation, and regression analysis using the Titanic dataset.
3. **GenAI Support Assistant** – Implements a Retrieval-Augmented Generation (RAG) based support assistant using Zepto policy documents, embeddings, ChromaDB, LangGraph, Pydantic, and FastAPI.

All three modules are maintained in this single repository.

## Project Structure

```text
zepto-data-analytics-project/
│
├── analytics/
│
├── data_pipeline/
│
├── support_assistant/
│
├── .gitignore
├── requirements.txt
└── README.md
