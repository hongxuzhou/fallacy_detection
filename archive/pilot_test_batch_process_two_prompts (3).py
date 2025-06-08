"""
This script uses batch process to solve the problem of input exceeding the max length of model input. It also stores the raw output for better diagonisis.
This script is based on `pilot_test_batch_process.py` and adds the two prompts based on Pragma-Dialectical Approach and Periodic Table of Argument" 

23/May update:
    1. Regarding the output format issue, adjustments are made to:
        - The prompts are modified to specify only one primary fallacy type in the output.
        - Trying to implement `Outlines` to achieve rigid output format, but need to consider the interface with the context. 
        
20-May Adjustments:
    1. Two theoretical prompts added. The thinking part of the output shows the model tries to follow the thinking path defined by the prompt. 
    2. The default batch size is changed to **1** from **5** for 1 prompt + 1 context + 1 example format. CAUTION: This causes the processing time to be much longer. You need at least 5 hours on Habrok

Known issues:
    1. format of model output: The more detailed prompts may exacerbate model output format confusion. This causes the evaluation method useless since results can't be correctly parsed. Specifically, thinking content may leak to response content. 
    2. infinity thinking loop: as pragma-dialectical results 161 and 16 shows, the model may fall into the infinite thinking process until the output exceeds the max length and gets truncated. 
    3. overthinking: as pragma-dialectical 1 shows, the model can be more hesitate than with the original prompt, resulting in violating the format requirements and adding a "final classification" which is not helpful. 
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
]

def parse_arguments():
    """Parse command line arguments."""
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
        default=1, # Using 1 prompt + 1 context + 1 example format
        help="Number of examples to process in each batch"
    )

    # This is the new argument for prompt type
    parser.add_argument(
        "--prompt_type",
        type=str,
        choices=["original", "p_d", "pta"], # Original keeps the original prompt used in pilot batch test unchanged, p_d means pragma-dialectics, pta stands for periodic table of arguments
        default="original", 
        help="Type of prompt to use for fallacy detection"
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

def create_prompt(examples, prompt_type="original"):
    """
    Create a prompt containing example statements for the model to analyse.

    Args: 
        examples: list of statements to anlayse.
        prompt_type: we have three types so far.
    
    
    """
    if prompt_type == "original": 

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
        prompt = instructions + "\n\n"
        for i, example in enumerate(examples, 1):
            prompt += f"STATEMENT {i}: {example}\n\n"
        
    elif prompt_type == "p_d":
        prompt = """You are an expert in argumentation analysis using the pragma-dialectical framework. Your task is to analyze statements and identify the PRIMARY fallacy that hinders the resolution of disputes in critical discussion.

## TASK OVERVIEW
Analyze each statement to:
1. Determine whether it violates any rules of critical discussion (fallacy detection)
2. If fallacious, identify the MOST SIGNIFICANT rule violation and classify the PRIMARY fallacy type
3. Provide a brief justification for your analysis

## THEORETICAL FOUNDATION: PRAGMA-DIALECTICAL RULES

In pragma-dialectics, fallacies are violations of rules for critical discussion. Analyze each statement against these ten rules:

1. FREEDOM RULE: Parties must not prevent each other from advancing or questioning standpoints
   - Violations: ad hominem attacks, threats (ad baculum), appeals to pity (ad misericordiam), declaring topics taboo

2. BURDEN OF PROOF RULE: A party who advances a standpoint must defend it when asked
   - Violations: evading the burden of proof by presenting claims as self-evident, shifting the burden to the other party

3. STANDPOINT RULE: Attacks must address the actual standpoint advanced by the other party
   - Violations: straw man arguments, distorting the opponent's position

4. RELEVANCE RULE: Standpoints must be defended with relevant argumentation
   - Violations: irrelevant arguments (ignoratio elenchi), appealing to emotion (pathos) or authority without proper reasoning

5. UNEXPRESSED PREMISE RULE: Parties cannot falsely attribute implicit premises or deny responsibility for their own implicit premises
   - Violations: exaggerating unexpressed premises, denying implied commitments

6. STARTING POINT RULE: Parties cannot falsely present premises as accepted starting points or deny established starting points
   - Violations: begging the question (petitio principii), denying agreed premises

7. VALIDITY RULE: Reasoning that is presented as formally conclusive must be logically valid
   - Violations: formal logical fallacies, invalid deductive reasoning

8. ARGUMENT SCHEME RULE: Standpoints must be defended using appropriate argument schemes applied correctly
   - Violations: hasty generalization, false analogy, false causality, slippery slope

9. CONCLUDING RULE: Failed defenses require withdrawing the standpoint; successful defenses require withdrawing doubts
   - Violations: refusing to accept the outcome, claiming absolute victory from limited success

10. LANGUAGE USE RULE: Formulations must be clear and unambiguous
    - Violations: vagueness, ambiguity, equivocation

## ANALYSIS PROCEDURE
For each statement:

STEP 1: DETECTION
- Carefully read the statement and determine if it violates any of the ten rules
- If no rules are violated, label it as "SOUND ARGUMENT"
- If one or more rules are violated, proceed to classification

STEP 2: CLASSIFICATION
- Identify the PRIMARY rule violation (the most significant one)
- While multiple violations may exist, focus on the most prominent fallacy
- Secondary violations can be mentioned in your analysis but not in the classification

STEP 3: JUSTIFICATION
- Provide a brief explanation of why the statement violates the primary rule
- You may note other violations in your analysis, but keep the focus on the main fallacy

## OUTPUT FORMAT
For each statement, provide your analysis in this format:

Statement: [Original statement]
Analysis: [Your pragma-dialectical analysis focusing on the primary violation, though you may briefly note secondary issues]
Classification: [NUMBER]

Where [NUMBER] is a SINGLE DIGIT representing the PRIMARY fallacy:
0 - Appeal to Emotion (violations of relevance rule with emotional appeals, ad baculum, ad misericordiam)
1 - Appeal to Authority (violations of relevance rule with inappropriate appeals to authority)
2 - Ad Hominem (violations of freedom rule through personal attacks)
3 - False Cause (violations of argument scheme rule with causal fallacies)
4 - Slippery Slope (violations of argument scheme rule with hasty slippery slope reasoning)
5 - Slogans (violations of language use rule through empty phrases, equivocation)
6 - Sound Argument (no violations of rules)

## IMPORTANT CONSIDERATIONS
- Focus on the statement itself, not surrounding context
- When multiple fallacies exist, classify based on the MOST SIGNIFICANT violation
- Be careful not to over-interpret - identify only clear violations
- For borderline cases, explain your reasoning for choosing the primary fallacy
- Maintain consistency in your analysis across different statements
"""

        # Again, add examples
        for i, example in enumerate(examples, 1):
            prompt += f"\n\nStatement: {example}"

    elif prompt_type == "pta":
        # Periodic table of arguments prompt 
        prompt = """You are an expert in argument analysis using the Periodic Table of Arguments framework. Your task is to analyze statements, determine their argument structure, and identify the PRIMARY fallacy based on invalid argument patterns.

## TASK OVERVIEW
For each statement, you will:
1. Deconstruct the statement into its argumentative components
2. Identify the argument's structural properties (form, substance, and lever)
3. Determine if the argument follows a valid pattern or contains fallacies
4. If fallacious, identify the PRIMARY pattern violation

## THEORETICAL FOUNDATION: PERIODIC TABLE OF ARGUMENTS
The PTA classifies arguments based on three parameters:

1. ARGUMENT FORM (the configuration of subjects and predicates):
   - ALPHA: "a is X, because a is Y" (same subject, different predicates)
   - BETA: "a is X, because b is X" (different subjects, same predicate)
   - GAMMA: "a is X, because b is Y" (different subjects, different predicates)
   - DELTA: "q [is A], because q is Z" (second-order predicate arguments)

2. ARGUMENT SUBSTANCE (types of statements used):
   - Statement of FACT (F): Descriptions of observable or verifiable reality
   - Statement of VALUE (V): Evaluative judgments based on criteria
   - Statement of POLICY (P): Advocating actions or decisions

3. ARGUMENT LEVER (relationship between non-common elements):
   - In ALPHA form: Relationship between predicates X and Y
   - In BETA form: Relationship between subjects a and b
   - In GAMMA form: Relationship between "a relates to b" and "X relates to Y"
   - In DELTA form: Relationship between Z and acceptability

## FALLACY DETECTION PROCEDURE
STEP 1: STATEMENT DECONSTRUCTION
- Identify the conclusion (the claim being supported)
- Identify the premise (the reason given to support the conclusion)
- If multiple arguments exist in one statement, separate them

STEP 2: STRUCTURAL ANALYSIS
- For the conclusion and premise:
  * Identify the subject(s) and predicate(s)
  * Determine the statement type(s) (F, V, or P)
- Based on subject/predicate configuration, identify the argument form (alpha, beta, gamma, or delta)
- Determine the argument substance (combination of statement types)

STEP 3: LEVER IDENTIFICATION
- Identify what connects the non-common elements
- Determine what type of relationship is being claimed (causal, analogical, etc.)

STEP 4: PATTERN EVALUATION
- Determine if the identified lever is valid for this form and substance combination
- Check if the lever follows an established valid argument pattern
- A fallacy exists when:
  * The lever doesn't establish a legitimate connection
  * The wrong type of lever is used for the given form and substance
  * The lever makes an unwarranted logical leap
  * Multiple incompatible levers are combined

STEP 5: FALLACY CLASSIFICATION
- If pattern violations exist, determine the PRIMARY (most significant) fallacy
- While multiple violations may be present, focus on the most prominent one
- Secondary violations can be noted in analysis but not in final classification

## COMMON FALLACIOUS PATTERNS
- False Causality: Claiming Y causes X without sufficient evidence (in alpha FF arguments)
- False Analogy: Claiming a and b are similar when they're fundamentally different (in beta arguments)
- False Equivalence: Treating different subjects as identical (in beta arguments)
- Equivocation: Using the same term with different meanings (misidentified form)
- Ad Hominem: Using personal attributes to attack a claim (invalid delta form)
- Circular Reasoning: Using the conclusion as a premise (disguised as alpha form)
- False Dilemma: Presenting only two options when more exist (invalid lever in gamma form)
- Appeal to Emotion: Using emotional response instead of valid lever
- Appeal to Authority: Using inappropriate authority (invalid lever in delta form)

## OUTPUT FORMAT
For each statement, provide your analysis in this format:

Statement: [Original statement]
[Your PTA structure analysis, focusing on the primary violation]
Classification: [NUMBER]

Where [NUMBER] is a SINGLE DIGIT representing the PRIMARY fallacy:
0 - Appeal to Emotion (invalid emotional lever)
1 - Appeal to Authority (invalid authority lever)
2 - Ad Hominem (attacks on character rather than arguments)
3 - False Cause (invalid causal lever)
4 - Slippery Slope (invalid chain of consequences)
5 - Slogans (statements without proper argumentative structure)
6 - Valid Argument (valid argument pattern)

Note: If multiple fallacies are present, classify based on the most significant violation only.
"""

        for i, example in enumerate(examples, 1):
            prompt += f"\n\nStatement: {example}"

    else:
        raise ValueError(f"Unknown prompt type: {prompt_type}")

    return prompt

def parse_classifications(output_text, num_examples, prompt_type="original"):
    """Parse model output to extract classifications for examples based on prompt type."""
    
    if prompt_type == "original":
        # Original parsing logic
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
    
    elif prompt_type == "p_d":
        # Parsing logic for pragma-dialectical prompt
        parts = re.split(r'Statement:', output_text)[1:]  # Skip the first part (instructions)
        classifications = []
        pattern = r"Classification:.*?(\d)"  # Look for a digit after "Classification:"
        
        for part in parts:
            match = re.search(pattern, part)
            if match:
                classification = int(match.group(1))
                # Validate that it's in the expected range
                if 0 <= classification <= 6:
                    classifications.append(classification)
                else:
                    classifications.append(None)
            else:
                # Check if it's explicitly labeled as a sound argument (6)
                if "Classification: SOUND ARGUMENT" in part:
                    classifications.append(6)
                else:
                    classifications.append(None)
    
    elif prompt_type == "pta":
        # Parsing logic for PTA prompt
        parts = re.split(r'Statement:', output_text)[1:]  # Skip the first part (instructions)
        classifications = []
        pattern = r"Fallacy Type.*?(\d)"  # Look for a digit after "Fallacy Type"
        
        for part in parts:
            match = re.search(pattern, part)
            if match:
                classification = int(match.group(1))
                # Validate that it's in the expected range
                if 0 <= classification <= 6:
                    classifications.append(classification)
                else:
                    classifications.append(None)
            else:
                # Check if it's explicitly labeled as a valid argument (6)
                if "Classification: VALID ARGUMENT" in part:
                    classifications.append(6)
                else:
                    classifications.append(None)
    
    else:
        raise ValueError(f"Unknown prompt type: {prompt_type}")
    
    # Handle mismatch in expected vs. actual classifications
    if len(classifications) < num_examples:
        # Fill in missing classifications with None
        classifications.extend([None] * (num_examples - len(classifications)))
    elif len(classifications) > num_examples:
        # Trim excess classifications
        classifications = classifications[:num_examples]
    
    return classifications

def process_batch(model, tokenizer, examples, batch_indices, max_new_tokens, device, prompt_type="original"):
    """Process a batch of examples and return both parsed results and raw outputs."""
    # Create prompt for this batch
    prompt = create_prompt(examples, prompt_type)
    
    # Prepare input for the model with thinking mode enabled
    messages = [{"role": "user", "content": prompt}]
    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True
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
    
    # Parse thinking content and final response
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
    """Save raw outputs from a batch to files."""
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
                model.device,
                args.prompt_type, # Add prompt choice here
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