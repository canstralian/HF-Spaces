import kagglehub
from kagglehub import KaggleDatasetAdapter

# Set the path to the file you'd like to load
file_path = ""  # Ensure this is set before running

try:
    hf_dataset = kagglehub.load_dataset(
        KaggleDatasetAdapter.HUGGING_FACE,
        "atifaliak/youtube-comments-dataset",
        file_path,
        # Additional arguments if needed
    )
    print("Hugging Face Dataset:", hf_dataset)
except Exception as e:
    print(f"An error occurred while loading the dataset: {e}")
