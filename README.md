# LTP Final Assignment: Fallacy Classification
Final Assignment for Language Technology Project

## Table of Contents
1. [Project Overview](#project-overview)  
2. [Dataset](#dataset)  
3. [File Structure](#file-structure)  
4. [Setup and Environment](#setup-and-environment)  
5. [References](#references)

---

## Project Overview
This project investigates how contextual information and structured reasoning prompts affect a Large Language Model's (LLM) ability to classify fallacies. We tested a Qwen3-8B model by enhancing prompts with two types of information: textual context (surrounding dialogue) and audio-based emotional context. We also compared base prompts against two theoretically-grounded Chain-of-Thought (CoT) frameworks: Pragma-Dialectics and the Periodic Table of Arguments (PTA).

Our key finding is that supplemental context often acts as noise, degrading performance rather than improving it. Simpler prompts focused on the core argument consistently yielded better results. Notably, emotional audio cues created a strong bias, causing the model to over-classify "Appeal to Emotion," while the combination of a base prompt with the Pragma-Dialectics framework achieved the highest overall accuracy. This work highlights the LLM's sensitivity to prompt framing and suggests that for logical reasoning tasks, more information is not always better.

---

## Dataset
The dataset used in this study is MM-USED-Fallacy, a multimodal corpus designed for fallacy detection. It is derived from the Unfair Speeches in English Debates (USED) dataset, which contains transcripts and audio from political debates.

---

## File Structure

```plaintext
├── archive
├── dataset
│   ├── test_set.csv
│   ├── validation_set.csv
├── outputs
│   ├── basic
│   ├── pd
│   ├── pta
├── fallacy_classification_pipeline.py
├── test.py
├── requirements.txt
└── README.md
```

**Folder / File Descriptions:**
- **archive**: Old code or files we are no longer using.
- **dataset**: Folder containing our validation and test sets.
- **outputs**: The results of our different prompt types and the different conditions
- **fallacy_classification_pipeline.py**: Our fallacy classification pipeline.
- **test.py**: Simple Python file to run the `fallacy_classification_pipeline.py` without needing to add arguments.
- **requirements.txt**: Lists the Python dependencies for the project.
- **README.md**: This README file.

---

## Setup and Environment

### Creating a Conda Environment
It is recommended to use a Conda environment to manage dependencies. Follow these steps:

1. **Create the Conda environment:**
   ```bash
   conda create -n ltp python=3.10
   ```

2. **Activate the environment:**
   ```bash
   conda activate ltp
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
---

---

## References
- **Official MM-ArgFallacy2025 Competition Documentation:** [MM-ArgFallacy2025](https://nlp-unibo.github.io/mm-argfallacy/2025/)
- **Official Qwen3 Documentation:** [Qwen3](https://qwenlm.github.io/blog/qwen3/)
---