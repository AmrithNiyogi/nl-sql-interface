#!/bin/bash

# Setup environment for model training
echo "Setting up the environment..."

# Update pip
pip install --upgrade pip

# Install necessary libraries
pip install gradio pandas~=2.2.3 langchain scikit-learn~=1.6.1 fastapi sqlalchemy~=2.0.40 uvicorn haystack matplotlib~=3.10.1 seaborn~=0.13.2 psycopg2 torch transformers networkx~=3.4.2 sweetviz~=2.3.1 llama-index

echo "Environment setup complete."
