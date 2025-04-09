from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("iiiorg/piiranha-v1-detect-personal-information")
model = AutoModelForTokenClassification.from_pretrained("iiiorg/piiranha-v1-detect-personal-information")

# Set up pipeline
pii_detector = pipeline("token-classification", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

# Function to detect PII in user input text
def detect_pii_in_text(user_input):
    print("\nUser Input Text:")
    print(user_input)
    
    # Get PII results from the model
    results = pii_detector(user_input)
    
    print("\nDetected PII:")
    if results:
        for ent in results:
            print(f"Entity: {ent['word']}, Label: {ent['entity_group']}, Score: {ent['score']:.4f}")
    else:
        print("No PII detected.")

# Get input from the user
user_input = input("Enter a sentence or text to detect PII: ")

# Detect PII in the input text
detect_pii_in_text(user_input)
