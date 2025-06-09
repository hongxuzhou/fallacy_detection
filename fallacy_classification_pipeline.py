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


FALLACY_LABELS = [
    "Appeal to Emotion",
    "Appeal to Authority",
    "Ad Hominem",
    "False Cause",
    "Slippery Slope",
    "Slogans",
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

        prompt = """You are an expert in logic and critical thinking. Your task is to analyze fallacious statements to classify their type of fallacy.

For the following statement, determine:
1. The type of fallacy from the following categories:
- Appeal to Emotion (0): The unessential loading of the argument with emotional language to exploit the audience emotional instinct. Sub-categories: appeal to pity, appeal to fear, loaded language (i.e., increasing the intensity of a phrase by using emotionally loaded descriptive phrases - either positive or negative) and flag waving, which appeals to the emotion of a group of people by referring to their identity.
- Appeal to Authority (1): When the arguer mentions the name of an authority or a group of people who agreed with her claim either without providing any relevant evidence, or by mentioning popular non-experts, or the acceptance of the claim by the majority.
- Ad Hominem (2): When the argument becomes an excessive attack on an arguer's position. It covers three sub-types: general ad hominem (an attack on the character of the opponent), tu quoque ad hominem (the “You did it first” attack) and bias ad hominem (an attack in which the arguer implies that the opponent is personally benefiting from his stance in the argument); and Name-calling, Labeling, i.e., when the arguer calls the opponent by an offensive label.
- False Cause (3): The misinterpretation of the correlation of two events for causation. Politicians tend to apply this technique when they affiliate the cause of an improvement to their party, or the failure to their opponent's party.
- Slippery Slope (4): It suggests that an unlikely exaggerated  outcome may follow an act. The intermediate premises are usually omitted and a starting premise is usually used as the first step leading to an exaggerated claim.
- Slogans (5): A brief and striking phrase used to provoke excitement of the audience, often accompanied by argument by repetition.

All statements are taken from United States presidential debates between 1960 and 2020.
"""
        
    elif prompt_type == "p_d": 
        
        prompt = """You are an expert in pragma-dialectical argumentation analysis. Your task is to analyze statements, identify the PRIMARY fallacy from a given list hindering critical discussion, and explain via Chain-of-Thought (CoT).

## TASK:
1. Analyze the statement against Pragma-Dialectical (PD) rules.
2. Identify the primary PD rule violation & specific PD fallacy.
3. Classify this into one of the 6 Output Categories.
4. Provide structured CoT justification based on Detection, Classification, and Justification.

## PRAGMA-DIALECTICAL RULES (Violations = Fallacies):

In pragma-dialectics, fallacies are violations of rules for critical discussion. Analyze each statement against these ten rules:

1. FREEDOM RULE: Parties must not prevent each other from advancing or questioning standpoints
- Violations: aiming to discredit/silence: Ad Hominem (2) (direct attack, or undermining competence/consistency/right to speak), threats (ad baculum), appeals to pity (ad misericordiam), emotional pressure to silence (can be Appeal to Emotion (0)), declaring taboo.

2. BURDEN OF PROOF RULE: A party who advances a standpoint must defend it when asked
- Violations: evading the burden of proof by presenting claims as self-evident, shifting the burden to the other party.

3. STANDPOINT RULE: Attacks must address the actual standpoint advanced by the other party
- Violations: straw man arguments, distorting the opponent's position.

4. RELEVANCE RULE: Standpoints must be defended with relevant argumentation
- Violations: irrelevant arguments (ignoratio elenchi), appealing to emotion (pathos) or authority without proper reasoning.

5. UNEXPRESSED PREMISE RULE: Parties cannot falsely attribute implicit premises or deny responsibility for their own implicit premises
- Violations: exaggerating unexpressed premises, denying implied commitments.

6. STARTING POINT RULE: Parties cannot falsely present premises as accepted starting points or deny established starting points
- Violations: begging the question (petitio principii), denying agreed premises.

7. VALIDITY RULE: Reasoning that is presented as formally conclusive must be logically valid
- Violations: formal logical fallacies, invalid deductive reasoning.

8. ARGUMENT SCHEME RULE: Standpoints must be defended using appropriate argument schemes applied correctly
- Violations: Arguing that one action will lead to an extreme consequence through an implied or unstated chain of events. The intermediate steps are typically omitted (Slippery Slope (4)), drawing broad conclusions from few examples (hasty generalization), misinterpreting correlation as causation (false analogy), misinterpreting correlation as causation (False Cause (3)).

9. CONCLUDING RULE: Failed defenses require withdrawing the standpoint; successful defenses require withdrawing doubts
- Violations: refusing to accept the outcome, claiming absolute victory from limited success.

10. LANGUAGE USE RULE: Formulations must be clear and unambiguous
- Violations: vagueness, ambiguity, equivocation.

## OUTPUT FALLACY CATEGORIES (Primary Classification):
- Appeal to Emotion (0): The unessential loading of the argument with emotional language to exploit the audience emotional instinct. Sub-categories: appeal to pity, appeal to fear, loaded language (i.e., increasing the intensity of a phrase by using emotionally loaded descriptive phrases - either positive or negative) and flag waving, which appeals to the emotion of a group of people by referring to their identity.
- Appeal to Authority (1): When the arguer mentions the name of an authority or a group of people who agreed with her claim either without providing any relevant evidence, or by mentioning popular non-experts, or the acceptance of the claim by the majority.
- Ad Hominem (2): When the argument becomes an excessive attack on an arguer's position. It covers three sub-types: general ad hominem (an attack on the character of the opponent), tu quoque ad hominem (the “You did it first” attack) and bias ad hominem (an attack in which the arguer implies that the opponent is personally benefiting from his stance in the argument); and Name-calling, Labeling, i.e., when the arguer calls the opponent by an offensive label.
- False Cause (3): The misinterpretation of the correlation of two events for causation. Politicians tend to apply this technique when they affiliate the cause of an improvement to their party, or the failure to their opponent's party.
- Slippery Slope (4): It suggests that an unlikely exaggerated  outcome may follow an act. The intermediate premises are usually omitted and a starting premise is usually used as the first step leading to an exaggerated claim.
- Slogans (5): A brief and striking phrase used to provoke excitement of the audience, often accompanied by argument by repetition.

## CoT ANALYSIS PROCEDURE (Follow this 3-Step Structure):

STEP 1: DETECTION
- Carefully read the statement. 
- What aspect of the statement seems problematic for critical discussion?
- Which of the 10 Pragma-Dialectical (PD) rules are violated? Identify the single *most significant* PD rule violated.

STEP 2: CLASSIFICATION
- Briefly restate the primary PD rule violation from Step 1 that is the basis for classification.
- Based on this primary PD violation, select the single most appropriate fallacy from the "OUTPUT FALLACY CATEGORIES" list (e.g., Ad Hominem (2)). Consider the violation examples under the PD rules for guidance.

STEP 3: JUSTIFICATION
- Provide a brief explanation. Why does the statement violate the identified primary PD rule? 
- How does this PD rule violation justify the chosen Output Category from Step 2? 
- If multiple PD rules were violated, explain why the chosen one is primary for classification into your 6 categories.

## KEY GUIDELINES:
-  Focus on the statement in isolation.
-  The *primary* classification must be one of the 6 Output Categories.
-  If multiple PD rules are violated, the "Classification" (Step 2) should focus on the violation that best maps to one of the 6 Output Categories as the *primary fallacy*.
-  Avoid over-interpretation; identify clear violations.

## STATEMENT (AND CONTEXT):
All statements are taken from United States presidential and vice-presidential debates between 1960 and 2020.
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

## OUTPUT FALLACY CATEGORIES (Primary Classification)
- Appeal to Emotion (0): The unessential loading of the argument with emotional language to exploit the audience emotional instinct. Sub-categories: appeal to pity, appeal to fear, loaded language (i.e., increasing the intensity of a phrase by using emotionally loaded descriptive phrases - either positive or negative) and flag waving, which appeals to the emotion of a group of people by referring to their identity.
- Appeal to Authority (1): When the arguer mentions the name of an authority or a group of people who agreed with her claim either without providing any relevant evidence, or by mentioning popular non-experts, or the acceptance of the claim by the majority.
- Ad Hominem (2): When the argument becomes an excessive attack on an arguer's position. It covers three sub-types: general ad hominem (an attack on the character of the opponent), tu quoque ad hominem (the “You did it first” attack) and bias ad hominem (an attack in which the arguer implies that the opponent is personally benefiting from his stance in the argument); and Name-calling, Labeling, i.e., when the arguer calls the opponent by an offensive label.
- False Cause (3): The misinterpretation of the correlation of two events for causation. Politicians tend to apply this technique when they affiliate the cause of an improvement to their party, or the failure to their opponent's party.
- Slippery Slope (4): It suggests that an unlikely exaggerated  outcome may follow an act. The intermediate premises are usually omitted and a starting premise is usually used as the first step leading to an exaggerated claim.
- Slogans (5): A brief and striking phrase used to provoke excitement of the audience, often accompanied by argument by repetition.

## STATEMENT (AND CONTEXT)
All statements are taken from United States presidential and vice-presidential debates between 1960 and 2020.
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


def classify_level(value):
    """
    Classify a value between -1 and 1 into low, moderate, or high.
    """
    
    if value < -0.33:
        return "low"
    elif value > 0.33:
        return "high"
    else:
        return "moderate"


def interpret_emotion(arousal, dominance, valence):
    tone = []

    if arousal > 0.33:
        tone.append("energetic")
    elif arousal < -0.33:
        tone.append("lethargic")
    else:
        tone.append("calm")

    if dominance > 0.33:
        tone.append("assertive")
    elif dominance < -0.33:
        tone.append("submissive")
    else:
        tone.append("neutral in control")

    if valence > 0.33:
        tone.append("positive or pleasant")
    elif valence < -0.33:
        tone.append("negative or unpleasant")
    else:
        tone.append("emotionally neutral")

    return tone


def add_context(prompt, sample, add_audio_context_flag=True, add_context_flag=True, context_window=3):
    """Parse statement and add context"""

    output_format = """
## OUTPUT FORMAT:

For the statement, provide your analysis in this format:
{
    "analysis": "Your reasoning here.",
    "classification": NUMBER
}
"""    
    
    statement = sample["snippet"]
    context = sample["dialogue"]
    context_date = sample["filename"]

    sentences = sent_tokenize(context)

    # Find the start and end indexes of the sentences that contain the statement
    best_span = None
    for i in range(len(sentences)):
        for j in range(i, len(sentences)):
            joined = " ".join(sentences[i:j+1]).strip()
            
            # Use exact match to avoid extra sentences leaking in
            if joined == statement.strip():
                best_span = (i, j)
                break

        if best_span:
            break

    if best_span is None:
        print("Snippet not found in context!")
        prompt += f"\nSTATEMENT: \"{statement}\"\n\n"
        prompt += f"\n{output_format}"
        return prompt
    
    ## Audio Modality
    arousal = sample["arousal"]
    dominance = sample["dominance"]
    valence = sample["valence"]

    arousal_level = classify_level(arousal)
    dominance_level = classify_level(dominance)
    valence_level = classify_level(valence)

    tone_descriptions = interpret_emotion(arousal, dominance, valence)

    audio_prompt = (
        f"The speaker speaks with an emotional tone characterized by "
        f"{arousal_level} arousal ({arousal:.2f}), "
        f"{dominance_level} dominance ({dominance:.2f}), and "
        f"{valence_level} valence ({valence:.2f}). "
        f"These values, ranging from -1 to 1, suggest the speech is "
        f"{', '.join(tone_descriptions[:-1])}, and {tone_descriptions[-1]}."
    )

    start_index, end_index = best_span
    # Get up to *context_window* sentences before
    before = " ".join(sentences[max(0, start_index - context_window):start_index]).strip()
    # Get up to *context_window* sentences after
    after = " ".join(sentences[end_index + 1:min(len(sentences), end_index + 1 + context_window)]).strip()

    context_combined = f"{before} [STATEMENT] {after}".strip()
    formatted_date = format_date_string(context_date)

    # Final prompt
    prompt += f"\nSTATEMENT: \"{statement.strip()}\"\n"
    if add_context_flag and context_window > 0:
        prompt += f"\nDATE: {formatted_date}\n"
        prompt += f"\nThe context below includes sentences before and after the statement, with [STATEMENT] marking where it originally appeared.\n"
        prompt += f"CONTEXT: {context_combined}\n"
    if add_audio_context_flag:
        prompt += f"\n{audio_prompt}"
    prompt += f"\n{output_format}"

    return prompt


def process_sample(model, tokenizer, sample, device, prompt_type="original", add_audio_context_flag=True, add_context_flag=True, context_window=3):
    """Process a batch of examples and return both parsed results and raw outputs."""

    # Generate prompt
    prompt = create_prompt(prompt_type)
    # Add context to prompt
    prompt_with_context_and_statement = add_context(prompt, sample, add_audio_context_flag, add_context_flag, context_window)
    
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
    max_new_tokens = input_num_tokens + 5120 

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


def evaluate_and_visualize(output_file, inference_processing_time):
    """Evaluate model performance and visualize results."""
    df = pd.read_csv("outputs/predictions.csv")

    predicted_labels= df["classification"].astype(int).tolist()
    true_labels = df["true_value"].astype(int).tolist()

    report = classification_report(true_labels, predicted_labels)
    with open('outputs/classification_report.txt', 'w') as f:
        f.write(f"Total inference time: {inference_processing_time:.2f} seconds\n\n")
        f.write(report)

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
    
    print(f"Loading model from {args.model_name}...")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        cache_dir=model_cache_dir,
        device_map="auto",
        torch_dtype=torch.float16,
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
            sample_result = process_sample(model, tokenizer, val_X, model.device, args.prompt_type, False, False, 3)
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
    evaluate_and_visualize("outputs/confusion_matrix.png", inference_processing_time)

    # Print total computation time
    total_time = time.time() - start_time
    print(f"Total execution time: {total_time:.2f} seconds")


if __name__ == "__main__":
    main()