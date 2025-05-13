import pandas as pd
from sqlalchemy import create_engine, inspect
import time

# Create a connection to PostgreSQL using SQLAlchemy
DATABASE_URL = "postgresql://postgres:postgres@localhost:5432/chinook"  # Replace with your actual connection details
engine = create_engine(DATABASE_URL)


# Function to preprocess the data and create question-answer pairs
def preprocess_data():
    preprocessed_data = []

    # Get the list of tables in the PostgreSQL database
    inspector = inspect(engine)
    tables = inspector.get_table_names()

    # Iterate through each table and fetch data
    for table in tables:
        print(f"Processing table: {table}")

        # Load table schema
        columns = [column['name'] for column in inspector.get_columns(table)]

        # Query all rows from the table (Consider adding LIMIT or pagination for large tables)
        query = f"SELECT * FROM {table} LIMIT 100;"  # Adjust LIMIT as needed
        df = pd.read_sql(query, engine)

        # Create question-answer pairs
        for _, row in df.iterrows():
            question = f"What is {table} with values {', '.join([f'{columns[i]}: {str(row[i])}' for i in range(len(row))])}?"
            answer = f"{table} details are {', '.join([f'{columns[i]}: {str(row[i])}' for i in range(len(row))])}."
            preprocessed_data.append({'question': question, 'answer': answer})

        # Optionally, add a small delay to avoid overloading the database
        time.sleep(1)

    return preprocessed_data


# Preprocess the data and save it to CSV
data = preprocess_data()
df = pd.DataFrame(data)

# Save the preprocessed data to CSV file for model training
df.to_csv('preprocessed_data.csv', index=False)

print("Preprocessing complete and data saved to 'preprocessed_data.csv'.")
