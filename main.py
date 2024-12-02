import sqlite3
import gradio as gr
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import load_dataset, Dataset
import logging
from typing import Optional

# Configuring logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Function for SQLite database connection
def create_connection(db_file):
    conn = sqlite3.connect(db_file)
    return conn

# Function to create a table
def create_table(conn):
    try:
        sql_create_table = """CREATE TABLE IF NOT EXISTS predictions (
                                id INTEGER PRIMARY KEY AUTOINCREMENT,
                                text TEXT NOT NULL,
                                label INTEGER
                             );"""
        cursor = conn.cursor()
        cursor.execute(sql_create_table)
    except sqlite3.Error as e:
        logging.error(f"Error creating table: {e}")

# Function to insert prediction into the database
def insert_prediction(conn, text, label):
    sql = '''INSERT INTO predictions(text, label) VALUES(?,?)'''
    cursor = conn.cursor()
    cursor.execute(sql, (text, label))
    conn.commit()
    return cursor.lastrowid

def load_data(dataset_name: str) -> Optional[Dataset]:
    # Load dataset implementation (same as before)
    ...

def fine_tune_model(model_name: str, dataset: Dataset, num_samples: int = 1000, 
                    num_epochs: int = 3, batch_size: int = 8, output_dir: str = "./results") -> Optional[AutoModelForSequenceClassification]:
    # Fine-tune model implementation (same as before)
    ...

def classify_text(model: Optional[AutoModelForSequenceClassification], 
                  tokenizer: AutoTokenizer, text: str) -> Optional[int]:
    # Classify text implementation (same as before)
    ...

def main():
    # Main function implementation
    ...

    # SQLite database setup
    database = "predictions.db"
    conn = create_connection(database)
    create_table(conn)

    def predict(text: str) -> Optional[int]:
        label = classify_text(model, tokenizer, text)
        insert_prediction(conn, text, label)  # Save prediction to SQLite
        return label

    gr.Interface(
        fn=predict,
        inputs=gr.Textbox(lines=2, placeholder="Enter text here..."),
        outputs="label",
        live=True
    ).launch()

    conn.close()  # Close the connection when done

if __name__ == "__main__":
    main()