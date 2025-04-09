import os
import time
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from collections import Counter
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from datasets import load_dataset

# Start timing the process
start_time = time.time()

# Create a time-based identifier for filenames
time_id = datetime.now().strftime("%Y%m%d_%H%M%S")

# Set folder to save images
output_folder = "saved_graphs"
os.makedirs(output_folder, exist_ok=True)

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("iiiorg/piiranha-v1-detect-personal-information")
model = AutoModelForTokenClassification.from_pretrained("iiiorg/piiranha-v1-detect-personal-information")

# Setup pipeline
pii_detector = pipeline("token-classification", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

# Load dataset and select a random subset of 100 examples
dataset = load_dataset("ai4privacy/pii-masking-400k", split="train")
num_examples = 20000
random_subset = dataset.shuffle(seed=42).select(range(num_examples))  # Shuffle and select first n after shuffle

# Initialize lists for actual and predicted labels
all_true_labels = []
all_pred_labels = []
all_texts = []

# Initialize a Counter to count frequencies of each category
category_counter = Counter()

# Process each example
for example in random_subset:
    text = example["source_text"]
    true_labels = [mask['label'] for mask in example['privacy_mask']]  # Actual labels (from privacy_mask)

    # Run the model on the text
    predictions = pii_detector(text)
    
    # Extract the predicted labels from 'entity_group' (corrected field)
    pred_labels = [pred['entity_group'] for pred in predictions]
    
    # Ensure the number of true labels matches the predicted labels
    if len(true_labels) == len(pred_labels):
        # Update category frequency counter
        category_counter.update(pred_labels)

        # Append true and predicted labels for evaluation
        all_true_labels.extend(true_labels)
        all_pred_labels.extend(pred_labels)
        
        # Add text to list
        all_texts.append(text)

# Calculate Precision and Recall per category
labels = list(category_counter.keys())  # Unique categories
precision, recall, _, _ = precision_recall_fscore_support(all_true_labels, all_pred_labels, labels=labels, average=None)

# Create a DataFrame to store the metrics for easier visualization
import pandas as pd
metrics_df = pd.DataFrame({
    'Category': labels,
    'Frequency': [category_counter[label] for label in labels],
    'Precision': precision,
    'Recall': recall
})

# Sort by frequency
metrics_df_sorted = metrics_df.sort_values(by='Frequency', ascending=False)

# Plot Histogram of the 10 most frequent categories
plt.figure(figsize=(10, 6))
sns.barplot(x='Category', y='Frequency', data=metrics_df_sorted.head(10))
plt.xticks(rotation=45)
plt.title('Top 10 Most Frequent PII Categories')
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "top_10_most_frequent_categories.png"))
plt.close()

# Plot Precision and Recall for the 10 most frequent categories
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

sns.barplot(x='Category', y='Precision', data=metrics_df_sorted.head(10), ax=axes[0])
axes[0].set_title('Precision of the Top 10 Categories')
axes[0].set_xticklabels(metrics_df_sorted['Category'].head(10), rotation=45)

sns.barplot(x='Category', y='Recall', data=metrics_df_sorted.head(10), ax=axes[1])
axes[1].set_title('Recall of the Top 10 Categories')
axes[1].set_xticklabels(metrics_df_sorted['Category'].head(10), rotation=45)

plt.tight_layout()
plt.savefig(os.path.join(output_folder, "precision_recall_top_10_categories.png"))
plt.close()

# Create another set of graphs for Precision and Recall with adjusted scale (to better see differences when values are close to 1)
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Plot Precision with adjusted scale
sns.barplot(x='Category', y='Precision', data=metrics_df_sorted.head(10), ax=axes[0])
axes[0].set_ylim(0.9, 1)  # Adjusted y-axis to zoom in on high precision values
axes[0].set_title('Precision of the Top 10 Categories (Adjusted Scale)')
axes[0].set_xticklabels(metrics_df_sorted['Category'].head(10), rotation=45)

# Plot Recall with adjusted scale
sns.barplot(x='Category', y='Recall', data=metrics_df_sorted.head(10), ax=axes[1])
axes[1].set_ylim(0.9, 1)  # Adjusted y-axis to zoom in on high recall values
axes[1].set_title('Recall of the Top 10 Categories (Adjusted Scale)')
axes[1].set_xticklabels(metrics_df_sorted['Category'].head(10), rotation=45)

plt.tight_layout()
plt.savefig(os.path.join(output_folder, "precision_recall_adjusted_scale.png"))
plt.close()

# Plot the 10 most precise categories
top_10_precise = metrics_df_sorted.nlargest(10, 'Precision')
plt.figure(figsize=(10, 6))
sns.barplot(x='Category', y='Precision', data=top_10_precise)
plt.xticks(rotation=45)
plt.title('Top 10 Most Precise PII Categories')
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "top_10_most_precise_categories.png"))
plt.close()

# Plot the 10 categories with the best recall
top_10_recall = metrics_df_sorted.nlargest(10, 'Recall')
plt.figure(figsize=(10, 6))
sns.barplot(x='Category', y='Recall', data=top_10_recall)
plt.xticks(rotation=45)
plt.title('Top 10 Categories with Best Recall')
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "top_10_best_recall_categories.png"))
plt.close()

# Plot the 10 least precise categories
top_10_least_precise = metrics_df_sorted.nsmallest(10, 'Precision')
plt.figure(figsize=(10, 6))
sns.barplot(x='Category', y='Precision', data=top_10_least_precise)
plt.xticks(rotation=45)
plt.title('Top 10 Least Precise PII Categories')
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "top_10_least_precise_categories.png"))
plt.close()

# Plot the 10 categories with the worst recall
top_10_worst_recall = metrics_df_sorted.nsmallest(10, 'Recall')
plt.figure(figsize=(10, 6))
sns.barplot(x='Category', y='Recall', data=top_10_worst_recall)
plt.xticks(rotation=45)
plt.title('Top 10 Categories with Worst Recall')
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "top_10_worst_recall_categories.png"))
plt.close()

# Confusion Matrix (Predicted vs Actual)
cm = confusion_matrix(all_true_labels, all_pred_labels, labels=labels)
plt.figure(figsize=(12, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.title('Confusion Matrix: Predicted vs Actual PII Categories')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "confusion_matrix.png"))
plt.close()

category_counts = dict(Counter(all_true_labels))
plt.figure(figsize=(8, 8))
plt.pie(category_counts.values(), labels=category_counts.keys(), autopct='%1.1f%%', startangle=90)
plt.title('PII Category Distribution\nn='+str(num_examples))
plt.axis('equal')
plt.tight_layout()
plt.savefig(os.path.join(output_folder, f"category_distribution_{time_id}.png"))
plt.close()

# Assuming all_true_labels and all_pred_labels are your true and predicted labels
confusion = confusion_matrix(all_true_labels, all_pred_labels)

# Calculate false positives
false_positive_counts = np.sum(confusion, axis=0) - np.diagonal(confusion)

# Get categories that have false positives > 0
categories = np.unique(all_true_labels)
categories_with_false_positives = {category: count for category, count in zip(categories, false_positive_counts) if count > 0}

# Sort categories by false positive counts in descending order and limit to top 10
sorted_categories = sorted(categories_with_false_positives.items(), key=lambda x: x[1], reverse=True)
top_categories = sorted_categories[:10]

# Separate the categories and their corresponding false positive counts
top_categories_labels = [category for category, count in top_categories]
top_false_positives = [count for category, count in top_categories]

# Plot the top false positives
plt.figure(figsize=(10, 6))
plt.bar(top_categories_labels, top_false_positives, color='orange')
plt.xlabel('Categories')
plt.ylabel('False Positives')
plt.title('Top False Positives per Category')
plt.tight_layout()

# Save the figure with a time-based filename
plt.savefig(os.path.join(output_folder, f"false_positives_{time_id}.png"))
plt.close()

categories_per_example = [len(set(example)) for example in all_true_labels]

# Plot the histogram
plt.figure(figsize=(10, 6))
plt.hist(categories_per_example, bins=range(1, max(categories_per_example) + 2), edgecolor='black')
plt.xlabel('Number of Categories per Example')
plt.ylabel('Frequency')
plt.title('Histogram of Number of Categories per Example')
plt.tight_layout()

# Save the figure with a time-based filename
plt.savefig(os.path.join(output_folder, f"categories_per_example_{time_id}.png"))
plt.close()

print("Graphs saved in the folder:", output_folder)
