import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import os


# Fallacy labels definition
FALLACY_LABELS = [
    "Appeal to Emotion",
    "Appeal to Authority", 
    "Ad Hominem",
    "False Cause",
    "Slippery Slope",
    "Slogans",
]


def evaluate_predictions(csv_path, output_dir="evaluation_outputs"):
    """
    Evaluate predictions from a CSV file and generate confusion matrix and classification report.
    
    Args:
        csv_path (str): Path to the predictions CSV file
        output_dir (str): Directory to save output files
    """
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Load predictions
    print(f"Loading predictions from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Strip whitespace from column names
    df.columns = df.columns.str.strip()
    
    # Extract labels
    predicted_labels = df["classification"].astype(int).tolist()
    true_labels = df["true_value"].astype(int).tolist()
    
    # Generate classification report
    print("Generating classification report...")
    report = classification_report(
        true_labels, 
        predicted_labels,
        target_names=FALLACY_LABELS,
        digits=4
    )
    
    # Save classification report
    report_path = os.path.join(output_dir, "classification_report.txt")
    with open(report_path, 'w') as f:
        f.write(f"Evaluation Results for: {csv_path}\n")
        f.write(f"Total samples evaluated: {len(df)}\n")
        f.write("="*60 + "\n\n")
        f.write(report)
    print(f"Classification report saved to: {report_path}")
    
    # Create confusion matrix
    print("Generating confusion matrix...")
    cm = confusion_matrix(
        true_labels, 
        predicted_labels,
        labels=list(range(len(FALLACY_LABELS)))
    )
    
    # Plot confusion matrix with green colormap
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=FALLACY_LABELS,
        yticklabels=FALLACY_LABELS,
        cbar_kws={'label': 'Count'},
        square=True
    )
    
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.title("Fallacy Detection Confusion Matrix", fontsize=14, pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # Save confusion matrix
    matrix_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(matrix_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to: {matrix_path}")
    
    # Print accuracy
    accuracy = sum(p == t for p, t in zip(predicted_labels, true_labels)) / len(df)
    print(f"\nOverall Accuracy: {accuracy:.4f}")


def main():
    """Main function to run the evaluation script."""
    parser = argparse.ArgumentParser(
        description="Evaluate fallacy detection predictions from CSV file"
    )
    
    parser.add_argument(
        "csv_path",
        type=str,
        help="Path to the predictions CSV file"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="evaluation_outputs",
        help="Directory to save output files"
    )
    
    args = parser.parse_args()
    
    # Run evaluation
    evaluate_predictions(args.csv_path, args.output_dir)


if __name__ == "__main__":
    main()