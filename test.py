from fallacy_classification_pipeline import *

DATASET_PATH = "dataset/test_set.csv"
MODEL_NAME = "Qwen/Qwen3-8B"
MODEL_RESULTS_SAVE_NAME = "qwen"

PROMPT_TYPE = "p_d"

BATCH_SIZE = 1

CSV_OUTPUT_PATH = "outputs/predictions.csv"

def main():
    # Load dataset
    df = load_dataset(DATASET_PATH)
        
    # Start timing
    start_time = time.time()

    # Create directory if it doesn't exist
    model_cache_dir = CACHE_DIR + MODEL_NAME
    if not os.path.exists(model_cache_dir):
        os.makedirs(model_cache_dir)

    # Load model and tokenizer
    print(f"Loading tokenizer from {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=model_cache_dir)

    print(f"Loading model from {MODEL_NAME}...")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
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

    if not os.path.exists(CSV_OUTPUT_PATH):
        with open(CSV_OUTPUT_PATH, mode="w", newline="", encoding="utf-8") as csvfile:
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
            sample_result = process_sample(model, tokenizer, val_X, model.device, PROMPT_TYPE, True, True, 3)
            decoded_result = tokenizer.decode(sample_result[0], skip_special_tokens=True)

            print(decoded_result)

            filename = f"outputs/{val_X['filename']}_{i}"
            statement = val_X["snippet"]
            
            classification = extract_output_and_store(filename, decoded_result)

            # Append result to CSV
            with open(CSV_OUTPUT_PATH, mode="a", newline="", encoding="utf-8") as csvfile:
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