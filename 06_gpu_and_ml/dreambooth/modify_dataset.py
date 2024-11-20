import huggingface_hub
from datasets import Dataset, load_dataset

# Authenticate with Hugging Face (replace "YOUR_API_TOKEN" with your actual token)
huggingface_hub.login("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")

# Load the dataset (replace "your_dataset_name" with the dataset name)
# e.g., "your_username/your_dataset"
dataset = load_dataset("yirenlu/heroicons-subset-100-images")

# Display the first row to understand the structure (especially column names)
print(dataset["train"][0])


# Define a function to modify captions in the "text" column
def modify_caption(example):
    # Example modification (appends " - modified" to each caption)
    example[
        "text"
    ] = f"an HCON, a black and white minimalist icon of {example['text'].replace('an icon of', '')}"
    return example


# Apply the modification to each split in the dataset
modified_dataset = dataset.map(modify_caption)

# Check the modifications
print(modified_dataset["train"][0])

# Grab only the first 25 rows
my_modified_dataset = Dataset.from_dict(modified_dataset["train"][:25])

# Save the modified dataset to re-upload it
# You may need to change "your_modified_dataset_name" to an appropriate unique name
my_modified_dataset.push_to_hub("heroicons-subset-25-images")

print("Dataset re-uploaded successfully!")
