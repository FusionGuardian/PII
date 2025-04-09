from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from datasets import load_dataset

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("iiiorg/piiranha-v1-detect-personal-information")
model = AutoModelForTokenClassification.from_pretrained("iiiorg/piiranha-v1-detect-personal-information")

# Set up pipeline
pii_detector = pipeline("token-classification", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

# Load a small subset of the dataset
dataset = load_dataset("ai4privacy/pii-masking-400k", split="train[:20]")

# Run the model on each example's source_text
for i, example in enumerate(dataset):
    text = example["source_text"]
    print(f"\nExample {i + 1}:\n{text}")
    results = pii_detector(text)
    print("Detected PII:")
    for ent in results:
        print(ent)
