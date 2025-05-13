import json
import argparse
from typing import List, Tuple
from pipeline import run_query_pipeline
from database import execute_query

def load_test_data(file_path: str) -> List[Tuple[str, str]]:
    """Load NL-SQL pairs from a JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return [(item["question"], item["sql"]) for item in data]

def normalize_sql(sql: str) -> str:
    """Normalize SQL for comparison (basic)"""
    # Using regular expressions to handle multiple spaces and inconsistent formatting better.
    import re
    sql = sql.strip().lower()
    sql = re.sub(r'\s+', ' ', sql)  # Replace multiple spaces with a single space
    sql = sql.replace(";", "")  # Remove semicolons
    return sql

def evaluate(test_data: List[Tuple[str, str]], execute=False) -> None:
    total = len(test_data)
    exact_match = 0
    exec_correct = 0

    for i, (question, gold_sql) in enumerate(test_data):
        print(f"\n[{i+1}] Question: {question}")
        pred_sql = run_query_pipeline(question)
        print(f"Predicted SQL: {pred_sql}")
        print(f"Expected SQL:  {gold_sql}")

        # Exact Match Evaluation
        if normalize_sql(pred_sql) == normalize_sql(gold_sql):
            exact_match += 1
            print("Exact Match ✅")
        else:
            print("Exact Match ❌")

        # Execution Match Evaluation
        if execute:
            try:
                gold_res = execute_query(gold_sql)
                pred_res = execute_query(pred_sql)
                if gold_res == pred_res:
                    exec_correct += 1
                    print("Execution Match ✅")
                else:
                    print("Execution Match ❌")
            except Exception as e:
                print(f"Execution Error: {e}")

    # Final Summary
    print("\n📊 Benchmark Summary:")
    print(f"Total Samples:       {total}")
    print(f"Exact Match Accuracy: {exact_match}/{total} = {exact_match / total:.2%}")
    if execute:
        print(f"Execution Accuracy:   {exec_correct}/{total} = {exec_correct / total:.2%}")
