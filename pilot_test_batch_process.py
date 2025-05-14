"""
update: 2025-May-14
This script uses the batch process to solve the problem of input exceeding the max length of model input. It also stores the raw output for better diagnosis.

Fallacy Detection and Classification using Qwen3-14B

This script processes a balanced pilot dataset of statements extracted from MMUSEDFallacy, using the Qwen3-14B model to:
1. Detect whether each statement contains a logical fallacy
2. If fallacious, classify it into one of 6 fallacy types

~The script takes a single-prompt approach, processing all examples at once~ <- see update,
and evaluates the model's performance against ground truth labels.

Author: Hongxu Zhou
""" 

import argparse
import time
import re
import os
import json
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import confusion_matrix, classification_report

# Define fallacy type labels for reference and plotting
FALLACY_LABELS = [
    "Appeal to Emotion",
    "Appeal to Authority",
    "Ad Hominem",
    "False Cause",
    "Slippery Slope",
    "Slogans",
    "Sound Argument"
] # The types of fallacies are consistent with the dataset and the original paper

def parse_arguments():
    """
    Parse command line arguments.
    Add new ones as needed.
    The options to test are:
    1. prompt variation
    2. prompt format
    3. prompting method
    4. additional information (meta-data / context)
    5. few shots
    """
    parser = argparse.ArgumentParser(description="Fallacy detection using Qwen3-14B")
    
    parser.add_argument(
        "--model_name", 
        type=str, 
        default="Qwen/Qwen3-14B",
        help="Name or path of the model to use"
    )
    
    parser.add_argument(
        "--cache_dir", 
        type=str, 
        default="/scratch/s5788668/models/qwen3",
        help="Directory where model cache is stored"
    )
    
    parser.add_argument(
        "--dataset_path", 
        type=str, 
        default="/scratch/s5788668/projects/com_arg/pilot_study/fallacy_pilot_dataset_with_sound.tsv",
        help="Path to the fallacy dataset"
    )
    
    parser.add_argument(
        "--max_new_tokens", 
        type=int, 
        default=2048,
        help="Maximum number of new tokens to generate"
    )
    
    parser.add_argument(
        "--output_file", 
        type=str, 
        default="fallacy_detection_results.csv",
        help="Path to save the results"
    )
    
    parser.add_argument(
        "--plot_file", 
        type=str, 
        default="confusion_matrix.png",
        help="Path to save the confusion matrix plot"
    )
    
    parser.add_argument(
        "--raw_output_dir", 
        type=str, 
        default="raw_outputs",
        help="Directory to save raw model outputs"
    )
    
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=5,
        help="Number of examples to process in each batch"
    )
    
    return parser.parse_args()

def load_dataset(dataset_path):
    """Load the fallacy dataset from a TSV file."""
    df = pd.read_csv(
        dataset_path, 
        sep='\t', 
        names=['text', 'label_type', 'is_fallacy']
    )
    
    print(f"Loaded dataset with {len(df)} examples")
    
    if df.empty:
        raise ValueError("Dataset is empty")
        
    return df

def create_prompt(examples):
    """
    For this script, I used a 'two-step' approach to prompt the model.
    The first step is a binary classification.
    The second step is a multi-class classification for the fallacies detected.
    
    The mapping of fallacies to numbers is provided by the TA. He said we can just use it this way.
    
    Known issues:
    1. The model output is not always in the expected format.
    2. The description of the fallacies may not be comprehensive enough.
    """
    instructions = """You are an expert in logic and critical thinking. Your task is to analyze statements to determine if they contain logical fallacies.

For each statement, determine:
1. Whether it contains a logical fallacy or is a sound argument
2. If it's a fallacy, identify which type from the following categories:
   - Appeal to Emotion (0): Using emotion instead of logic
   - Appeal to Authority (1): Using authority figures to justify claims
   - Ad Hominem (2): Attacking the person instead of their argument
   - False Cause (3): Assuming correlation implies causation
   - Slippery Slope (4): Claiming one event leads to extreme consequences
   - Slogans (5): Using catchy phrases instead of substantive argument
   - Sound Argument (6): Not a fallacy

For each statement, provide a brief analysis, then end with a classification in this exact format:
CLASSIFICATION: [NUMBER]

Where [NUMBER] is a single digit (0-6) representing your classification.
"""

    # Add each example with a number prefix
    prompt = instructions + "\n\n" # Insert a blank line between instructions and examples
    for i, example in enumerate(examples, 1): #careful with the index
        prompt += f"STATEMENT {i}: {example}\n\n"
    
    return prompt

def parse_classifications(output_text, num_examples):
    """Parse model output to extract classifications for examples."""
    # Split the output by statement numbers to separate analyses
    parts = re.split(r'STATEMENT \d+:', output_text)
    
    # The first part is just the instructions echoed back, so remove it
    if len(parts) > 1:
        parts = parts[1:]
    
    classifications = []
    pattern = r"CLASSIFICATION:\s*(\d)"
    
    # Try to find classification in each part
    for part in parts:
        match = re.search(pattern, part)
        if match:
            classification = int(match.group(1))
            # Validate that it's in the expected range
            if 0 <= classification <= 6:
                classifications.append(classification)
            else:
                # If outside valid range, append None
                classifications.append(None)
        else:
            # If no classification found, append None
            classifications.append(None)
    
    # Handle mismatch in expected vs. actual classifications
    if len(classifications) < num_examples:
        # Fill in missing classifications with None
        classifications.extend([None] * (num_examples - len(classifications)))
    elif len(classifications) > num_examples:
        # Trim excess classifications
        classifications = classifications[:num_examples]
    
    return classifications

def process_batch(model, tokenizer, examples, batch_indices, max_new_tokens, device):
    """Process a batch of examples and return both parsed results and raw outputs."""
    # Create prompt for this batch
    prompt = create_prompt(examples)
    
    # Prepare input for the model with thinking mode enabled
    messages = [{"role": "user", "content": prompt}]
    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True # Enable thinking mode
    )
    
    # Tokenize input
    inputs = tokenizer([input_text], return_tensors="pt").to(device)
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.6,
            do_sample=True,
            top_p=0.95,
            top_k=20
        )
    
    # Extract the model's response
    output_ids = outputs[0][len(inputs.input_ids[0]):].tolist()
    
    # Parse thinking content and final response following the offical doc
    try:
        # Find index of </think> token (151668 for Qwen3)
        index = len(output_ids) - output_ids[::-1].index(151668)
        thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
        response_content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
    except ValueError:
        # If </think> not found, assume everything is the response
        thinking_content = ""
        response_content = tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")
    
    # Extract classifications
    classifications = parse_classifications(response_content, len(examples))
    
    # Prepare results
    results = {
        'batch_indices': batch_indices,
        'parsed_classifications': classifications,
        'thinking_content': thinking_content,
        'response_content': response_content,
        'raw_output': tokenizer.decode(output_ids, skip_special_tokens=True)
    }
    
    return results

def save_raw_outputs(batch_results, raw_output_dir, batch_num):
    """
    This function does not work so well mosty because the model output is not 100% stable.
    Please always check the raw output before using it.
    """
    # Create directory if it doesn't exist
    if not os.path.exists(raw_output_dir):
        os.makedirs(raw_output_dir)
    
    # Save batch results as JSON
    with open(f"{raw_output_dir}/batch_{batch_num}_results.json", 'w') as f:
        # Convert indices to regular Python list for JSON serialization
        serializable_results = batch_results.copy()
        serializable_results['batch_indices'] = batch_results['batch_indices'].tolist()
        json.dump(serializable_results, f, indent=2)
    
    # Save thinking content
    with open(f"{raw_output_dir}/batch_{batch_num}_thinking.txt", 'w') as f:
        f.write(batch_results['thinking_content'])
    
    # Save response content
    with open(f"{raw_output_dir}/batch_{batch_num}_response.txt", 'w') as f:
        f.write(batch_results['response_content'])

def evaluate_and_visualize(true_labels, predicted_labels, examples, output_file, plot_file):
    """Evaluate model performance and visualize results."""
    # Replace None values with -1 for evaluation
    predicted_clean = [label if label is not None else -1 for label in predicted_labels]
    
    # Calculate overall accuracy (excluding parsing failures)
    valid_predictions = [p for p in predicted_clean if p != -1]
    if len(valid_predictions) > 0:
        valid_indices = [i for i, p in enumerate(predicted_clean) if p != -1]
        valid_true = [true_labels[i] for i in valid_indices]
        accuracy = sum(p == t for p, t in zip(valid_predictions, valid_true)) / len(valid_predictions)
        print(f"Overall accuracy (excluding parsing failures): {accuracy:.4f}")
    
    # Include parsing failures in total accuracy calculation
    total_accuracy = sum(p == t for p, t in zip(predicted_clean, true_labels)) / len(true_labels)
    print(f"Total accuracy (including parsing failures): {total_accuracy:.4f}")
    
    # Count parsing failures
    parsing_failures = predicted_clean.count(-1)
    if parsing_failures > 0:
        print(f"Parsing failures: {parsing_failures} ({parsing_failures/len(predicted_clean):.2%})")
    
    # Classification report (excluding parsing failures)
    if len(valid_predictions) > 0:
        # Replace label values with readable names
        label_names = [FALLACY_LABELS[i] for i in range(7)]
        
        # Print classification report
        report = classification_report(
            [true_labels[i] for i in valid_indices],
            valid_predictions,
            target_names=label_names,
            labels=range(7)
        )
        print("\nClassification Report:")
        print(report)
    
    # Create confusion matrix
    # Include a row/column for parsing failures (-1)
    cm = confusion_matrix(
        true_labels, 
        predicted_clean,
        labels=list(range(-1, 7))  # Include -1 for parsing failures
    )
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    labels = ["Parsing Error"] + FALLACY_LABELS
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=labels,
        yticklabels=FALLACY_LABELS
    )
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Fallacy Detection Confusion Matrix')
    plt.tight_layout()
    plt.savefig(plot_file)
    print(f"Confusion matrix saved to {plot_file}")
    
    # Save detailed results to CSV
    results_df = pd.DataFrame({
        'statement': examples,
        'true_label': true_labels,
        'predicted_label': predicted_clean,
    })
    
    # Save the results dataframe
    results_df.to_csv(output_file, index=False)
    print(f"Detailed results saved to {output_file}")

def main():
    """Main function for the fallacy detection pipeline."""
    args = parse_arguments()
    
    # Load dataset
    df = load_dataset(args.dataset_path)
    examples = df['text'].tolist()
    true_labels = df['label_type'].tolist()
    
    # Start timing
    start_time = time.time()
    
    # Load model and tokenizer
    print(f"Loading tokenizer from {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir=args.cache_dir)
    
    print(f"Loading model from {args.model_name} with 4-bit quantization...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        cache_dir=args.cache_dir,
        torch_dtype=torch.float16,
        device_map="auto",
        load_in_4bit=True,
        low_cpu_mem_usage=True
    )
    
    model_load_time = time.time() - start_time
    print(f"Model loaded in {model_load_time:.2f} seconds")
    
    # Process examples in batches
    batch_size = args.batch_size
    num_examples = len(examples)
    parsed_classifications = [None] * num_examples
    
    print(f"Processing {num_examples} examples in batches of {batch_size}...")
    
    batch_start_time = time.time()
    num_batches = (num_examples + batch_size - 1) // batch_size  # Ceiling division
    
    for i in range(0, num_examples, batch_size):
        batch_num = i // batch_size + 1
        print(f"\nProcessing batch {batch_num}/{num_batches}...")
        
        # Get batch of examples and their indices
        batch_indices = np.arange(i, min(i + batch_size, num_examples))
        batch_examples = [examples[j] for j in batch_indices]
        
        # Process batch
        try:
            batch_results = process_batch(
                model, 
                tokenizer, 
                batch_examples, 
                batch_indices, 
                args.max_new_tokens,
                model.device
            )
            
            # Save raw outputs
            save_raw_outputs(batch_results, args.raw_output_dir, batch_num)
            
            # Update parsed classifications
            for idx, classification in zip(batch_indices, batch_results['parsed_classifications']):
                parsed_classifications[idx] = classification
                
            # Print batch stats
            batch_parsing_failures = batch_results['parsed_classifications'].count(None)
            print(f"Batch {batch_num} completed: {len(batch_indices) - batch_parsing_failures}/{len(batch_indices)} successfully parsed")
            
        except Exception as e:
            print(f"Error processing batch {batch_num}: {str(e)}")
            # Continue with next batch
    
    batch_end_time = time.time()
    batch_processing_time = batch_end_time - batch_start_time
    print(f"\nAll batches processed in {batch_processing_time:.2f} seconds")
    
    # Combine thinking and response content from all batches
    all_thinking_content = ""
    all_response_content = ""
    
    for i in range(1, num_batches + 1):
        with open(f"{args.raw_output_dir}/batch_{i}_thinking.txt", 'r') as f:
            all_thinking_content += f"--- BATCH {i} ---\n" + f.read() + "\n\n"
        with open(f"{args.raw_output_dir}/batch_{i}_response.txt", 'r') as f:
            all_response_content += f"--- BATCH {i} ---\n" + f.read() + "\n\n"
    
    # Save combined thinking and response content
    with open(args.output_file.replace('.csv', '_thinking.txt'), 'w') as f:
        f.write(all_thinking_content)
    print(f"Combined thinking content saved to {args.output_file.replace('.csv', '_thinking.txt')}")
    
    with open(args.output_file.replace('.csv', '_response.txt'), 'w') as f:
        f.write(all_response_content)
    print(f"Combined response content saved to {args.output_file.replace('.csv', '_response.txt')}")
    
    # Evaluate model performance
    evaluate_and_visualize(true_labels, parsed_classifications, examples, args.output_file, args.plot_file)
    
    # Print total time
    total_time = time.time() - start_time
    print(f"Total execution time: {total_time:.2f} seconds")

if __name__ == "__main__":
    main()