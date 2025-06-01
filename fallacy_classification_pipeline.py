import argparse
import csv
import json
import os
import re
import time
from calendar import month_abbr

import bitsandbytes as bnb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from nltk.tokenize import sent_tokenize
from sklearn.metrics import classification_report, confusion_matrix
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

import nltk
nltk.download('punkt_tab')


# Define fallacy type labels for reference and plotting (only classification, so no Sound Arguments)
FALLACY_LABELS = [
    "Appeal to Emotion",
    "Appeal to Authority",
    "Ad Hominem",
    "False Cause",
    "Slippery Slope",
    "Slogans",
    # "Sound Argument"
]
CACHE_DIR = "cache/"


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Fallacy detection using Qwen3-14B")
    
    parser.add_argument(
        "--model_name", 
        type=str, 
        default="Qwen/Qwen3-8B",
        help="Name or path of the model to use"
    )
    
    parser.add_argument(
        "--dataset_path", 
        type=str, 
        default="dataset/mamkit_dataset.csv",
        help="Path to the fallacy dataset"
    )

    parser.add_argument(
        "--output_file", 
        type=str, 
        default="outputs/predictions.csv",
        help="File to save the results"
    )
    
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=1,
        help="Number of examples to process in each batch"
    )

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
        sep=",", 
        header=0    
    )
    
    print(f"Loaded dataset with {len(df)} examples")
    
    if df.empty:
        raise ValueError("Dataset is empty")
        
    return df

def create_prompt(prompt_type="original"):
    """
    Create a prompt containing example statements for the model to analyse.

    Args: 
        examples: list of statements to anlayse.
        prompt_type: we have three types so far.
    
    
    """
    if prompt_type == "original": 

        prompt = """You are an expert in logic and critical thinking. Your task is to analyze statements to determine if they contain logical fallacies.

For the following statement, determine:
1. The type of fallacy from the following categories:
- Appeal to Emotion (0): Using emotion instead of logic
- Appeal to Authority (1): Using authority figures to justify claims
- Ad Hominem (2): Attacking the person instead of their argument
- False Cause (3): Assuming correlation implies causation
- Slippery Slope (4): Claiming one event leads to extreme consequences
- Slogans (5): Using catchy phrases instead of substantive argument

All statements are taken from United States presidential debates between 1960 and 2020.
"""
        
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

    else:
        raise ValueError(f"Unknown prompt type: {prompt_type}")

    return prompt


def format_date_string(date_str):
    """
    Convert '1984_07Oct_1' → '1984-10-07'
    """

    year_str, dd_mon, _ = date_str.split("_")
    year = int(year_str)
    day = dd_mon[:2]
    month = dd_mon[2:].title()

    month_num = list(month_abbr).index(month)

    return f"{year:04d}-{month_num:02d}-{day}"


def add_context(prompt, sample, add_date=True, add_context_flag=True, context_window=3):
    """Parse statement and add context"""

    output_format = """Respond in this exact JSON format:
{
    "analysis": "Your reasoning here.",
    "classification": NUMBER
}"""    
    
    statement = sample["snippet"]
    context = sample["dialogue"]
    context_date = sample["filename"]

    sentences = sent_tokenize(context)

    # Find the index of the sentence that contains the statement
    match_index = None
    for i, sent in enumerate(sentences):
        if statement in sent:
            match_index = i
            break

    if match_index is None:
        print("Sentence not found in context!")
        prompt += f"\nSTATEMENT: \"{statement.strip()}\"\n\n"
        prompt += f"\n{output_format}"
        return prompt
        
    # Get up to *context_window* sentences before
    before = " ".join(sentences[max(0, match_index - context_window):match_index]).strip()
    # Get up to *context_window* sentences after
    after = " ".join(sentences[match_index + 1:match_index + 1 + context_window]).strip()

    context_combined = f"{before} [STATEMENT] {after}".strip()

    formatted_date = format_date_string(context_date)

    # Final prompt
    prompt += f"\nSTATEMENT: \"{statement.strip()}\"\n"
    if add_date:
        prompt += f"\nDATE: {formatted_date}\n"
    if add_context_flag and context_window > 0:
        prompt += f"\nThe context below includes sentences before and after the statement, with [STATEMENT] marking where it originally appeared.\n"
        prompt += f"CONTEXT: {context_combined}\n"
    prompt += f"\n{output_format}"

    return prompt


def process_sample(model, tokenizer, sample, device, prompt_type="original"):
    """Process a batch of examples and return both parsed results and raw outputs."""

    # Generate prompt
    prompt = create_prompt(prompt_type)
    # Add context to prompt
    prompt_with_context_and_statement = add_context(prompt, sample, False, False, 3)
    
    print(prompt_with_context_and_statement)
    
    print("Created Prompt")
    
    # Apply chat template
    messages = [{"role": "user", "content": prompt_with_context_and_statement}]
    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True
    )
    
    print("Applied Chat Template")

    # Tokenize
    inputs = tokenizer(input_text, return_tensors="pt").to(device)

    input_num_tokens = inputs['input_ids'].shape[1]
    max_new_tokens = input_num_tokens + 4096 

    print(f"max_new_tokens: {max_new_tokens}")
    
    print("Tokenized")

    print("Inference Started")
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.6,
            do_sample=True,
            top_p=0.95,
            top_k=20,
            eos_token_id=tokenizer.eos_token_id
        )

    print("Inference Completed")
    
    return outputs
    

def extract_output_and_store(filename, text_output):
    """
    Extracts <think>...</think> content into a .txt file
    and the trailing JSON block into .json file.
    """
    # Extract <think>...</think>
    think_match = re.search(r"<think>(.*?)</think>", text_output, re.DOTALL)
    think_content = think_match.group(1).strip() if think_match else ""

    # Extract JSON (after </think>)
    try:
        json_part = text_output.split("</think>")[-1].strip()
        parsed_json = json.loads(json_part)
    except Exception as e:
        print(f"Error parsing JSON: {e}")
        parsed_json = {}

    with open(f"{filename}.txt", "w", encoding="utf-8") as f:
        f.write(think_content)

    with open(f"{filename}.json", "w", encoding="utf-8") as f:
        json.dump(parsed_json, f, indent=4)

    return parsed_json.get("classification")


def evaluate_and_visualize(output_file):
    """Evaluate model performance and visualize results."""
    df = pd.read_csv("outputs/predictions.csv")

    predicted_labels= df["classification"].astype(int).tolist()
    true_labels = df["true_value"].astype(int).tolist()

    report = classification_report(true_labels, predicted_labels)
    print(report)
    
    # Create confusion matrix
    cm = confusion_matrix(
        true_labels, 
        predicted_labels,
        labels=list(range(len(FALLACY_LABELS)))
    )
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=FALLACY_LABELS,
                yticklabels=FALLACY_LABELS)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Fallacy Detection Confusion Matrix")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_file)
    print(f"Confusion matrix saved to {output_file}")


def map_single_fallacy(fallacy):
    label_map = {
        label.replace(" ", "").lower(): idx
        for idx, label in enumerate(FALLACY_LABELS)
    }
    return label_map[fallacy.strip().lower()]


def main():
    """Main function for the fallacy classification pipeline."""
    args = parse_arguments()

    # Load dataset
    df = load_dataset(args.dataset_path)
        
    # Start timing
    start_time = time.time()

    # Create directory if it doesn't exist
    model_cache_dir = CACHE_DIR + args.model_name
    if not os.path.exists(model_cache_dir):
        os.makedirs(model_cache_dir)

    # Load model and tokenizer
    print(f"Loading tokenizer from {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir=model_cache_dir)

    # bnb_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_use_double_quant=True,
    #     bnb_4bit_quant_type="nf4",
    #     bnb_4bit_compute_dtype=torch.bfloat16
    # )
    
    # print(f"Loading model from {MODEL_NAME} with 4-bit quantization...")
    print(f"Loading model from {args.model_name}...")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        cache_dir=model_cache_dir,
        device_map="auto",
        torch_dtype=torch.float16,
        # quantization_config=bnb_config,
        low_cpu_mem_usage=True
    )

    model_load_time = time.time() - start_time
    print(f"Model loaded in {model_load_time:.2f} seconds")

    # Process samples
    num_samples = len(df)
    print(f"Processing {num_samples} samples individually...")

    if not os.path.exists(args.output_file):
        with open(args.output_file, mode="w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["filename", "statement", "classification", "true_value"])

    inference_start_time = time.time()

    for i in range(num_samples):
        val_X = df.iloc[i]
        val_y = map_single_fallacy(df['fallacy'].iloc[i])

        sample_num = i + 1
        print(f"Processing sample {sample_num}/{num_samples}...")
        
        # Process sample
        try:
            sample_result = process_sample(model, tokenizer, val_X, model.device, args.prompt_type)
            decoded_result = tokenizer.decode(sample_result[0], skip_special_tokens=True)

            print(decoded_result)

            filename = f"outputs/{val_X['filename']}_{i}"
            statement = val_X["snippet"]
            
            classification = extract_output_and_store(filename, decoded_result)

            # Append result to CSV
            with open(args.output_file, mode="a", newline="", encoding="utf-8") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([filename, statement, classification, val_y])
            
        except Exception as e:
            print(f"Error processing sample {sample_num}: {e}")
            continue

    inference_end_time = time.time()
    inference_processing_time = inference_end_time - inference_start_time
    print(f"All samples processed in {inference_processing_time:.2f} seconds")

    # Evaluate model performance
    evaluate_and_visualize("outputs/confusion_matrix.png")

    # Print total computation time
    total_time = time.time() - start_time
    print(f"Total execution time: {total_time:.2f} seconds")


if __name__ == "__main__":
    main()