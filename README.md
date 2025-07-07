# Fine-Tuned Embeddings for Sales Conversion Prediction

## Overview
This project fine-tunes embeddings for sales call transcripts to improve conversion prediction accuracy. It uses Gemini API for embeddings, contrastive learning for fine-tuning, and LangChain for orchestration.

## Features
- Domain-specific embedding fine-tuning
- Baseline vs. fine-tuned comparison
- Conversion prediction pipeline
- LangChain integration

## Setup
1. Clone the repo and navigate to the project directory.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Add your Gemini API key to a `.env` file:
   ```env
   GEMINI_API_KEY=your_key_here
   ```
4. Place your data in the `data/` directory.

## Usage
- Run baseline pipeline:
  ```bash
  python src/baseline.py
  ```
- Run fine-tuning:
  ```bash
  python src/contrastive.py
  ```
- Evaluate:
  ```bash
  python src/evaluate.py
  ```
- Launch LangChain app:
  ```bash
  python src/langchain_app.py
  ```

## Directory Structure
- `data/` - Raw and processed data
- `src/` - Source code

## Requirements
See `requirements.txt`. 