# Natural Language Interface for SQL-Free Business Data Exploration

---

## Overview
- This project provides an AI Powered assistant that converts natural language queries into SQL to interact with business databases like PostgreSQl or BigQuery.
- It uses Langchain, Haystack, and GPT-Neo (can be used with Llama 2 or gemini 1.5) for query generation and explanation.

---

## Setup
1. Clone the repository:
    ```bash
        git clone https://github.com/AmrithNiyogi/nl-sql-interface.git
        cd nl-sql-interface
    ```
2. Create a virtual environment and install dependencies:
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # Linux/Mac
    venv\Scripts\activate     # Windows
    pip install -r requirements.txt
    ```
3. Set up the database:
   - Download the Chinook database and place it in `data/chinook.db`.
   - Or use another PostgreSQL/BigQuery database.

4. Run the FastAPI backend:
    ```bash
    uvicorn src.api:app --reload
    ```
   
5. Launch the Gradio frontend:
    ```bash
    python frontend/gradio_app.py
    ```

---

## How to Contribute
1. Fork the repository
2. Clone your fork locally
3. Create a new branch for your changes
4. Commit your changes and create a pull request