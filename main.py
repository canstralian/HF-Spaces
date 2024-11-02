import gradio as gr
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import load_dataset, Dataset
import logging

# Configuring logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(dataset_name: str) -> Dataset:
    """
    Load the dataset by name.

    Parameters:
    - dataset_name: str : Name of the dataset to load

    Returns:
    - dataset: Dataset : Loaded dataset
    """
    try:
        dataset = load_dataset(dataset_name)
        logging.info(f"Loaded dataset: {dataset_name}")
        return dataset
    except Exception as e:
        logging.error(f"Error loading dataset: {e}")
        raise

def fine_tune_model(
    model_name: str, 
    dataset: Dataset, 
    num_samples: int = 1000, 
    num_epochs: int = 3, 
    batch_size: int = 8, 
    output_dir: str = "./results"
) -> AutoModelForSequenceClassification:
    """
    Fine-tune a pre-trained model on a provided dataset.

    Parameters:
    - model_name: str : Name of the pre-trained model
    - dataset: Dataset : Dataset for training and evaluation
    - num_samples: int : Number of samples to use for training (default 1000)
    - num_epochs: int : Number of training epochs (default 3)
    - batch_size: int : Batch size for training and evaluation (default 8)
    - output_dir: str : Directory to save training results (default './results')

    Returns:
    - model: AutoModelForSequenceClassification : Trained model
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        logging.info(f"Loaded model and tokenizer: {model_name}")
    except Exception as e:
        logging.error(f"Error loading model or tokenizer: {e}")
        raise

    def tokenize_function(examples):
        return tokenizer(examples['text'], padding="max_length", truncation=True)

    try:
        # Use streaming and caching for efficient processing
        tokenized_dataset = dataset.map(tokenize_function, batched=True, num_proc=4, cache_file_name="cached_tokenized_dataset.arrow")
        tokenized_dataset = tokenized_dataset.shuffle(seed=42).select(range(num_samples))
        logging.info("Tokenized, shuffled, and selected dataset samples.")
    except Exception as e:
        logging.error(f"Error during tokenization or dataset preparation: {e}")
        raise

    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        weight_decay=0.01,
        logging_dir=f"{output_dir}/logs",
        logging_strategy="steps",
        logging_steps=10,
    )

    try:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset['train'],
            eval_dataset=tokenized_dataset['test'],
            tokenizer=tokenizer,
        )
        trainer.train()
        logging.info("Model training completed.")
    except Exception as e:
        logging.error(f"Error during training: {e}")
        raise

    return model

def classify_text(model: AutoModelForSequenceClassification, tokenizer: AutoTokenizer, text: str) -> int:
    """
    Classify the input text using the fine-tuned model.

    Parameters:
    - model: AutoModelForSequenceClassification : Fine-tuned model
    - tokenizer: AutoTokenizer : Tokenizer for the model
    - text: str : Input text to classify

    Returns:
    - label: int : Predicted label
    """
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        outputs = model(**inputs)
        return outputs.logits.argmax().item()
    except Exception as e:
        logging.error(f"Error during text classification: {e}")
        raise

def main():
    """
    Main function to run the Gradio interface for text classification.
    """
    dataset_name = "imdb"
    model_name = "distilbert-base-uncased"

    try:
        dataset = load_data(dataset_name)
        model = fine_tune_model(model_name, dataset, num_samples=1000, num_epochs=3, batch_size=8, output_dir="./results")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        logging.info("Model is ready for inference.")
    except Exception as e:
        logging.error(f"Setup failed: {e}")
        return

    def predict(text: str) -> int:
        return classify_text(model, tokenizer, text)

    gr.Interface(
        fn=predict,
        inputs=gr.Textbox(lines=2, placeholder="Enter text here..."),
        outputs="label"
    ).launch()

if __name__ == "__main__":
    main()