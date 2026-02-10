
import os
import datasets
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# --- Configuration ---
SFT_SIZE = 1000  # Number of synthetic SFT examples to generate
SEED = 42

def make_gsm8k_prompt(question):
    return f"Question: {question}\nLet's think step by step to solve this.\n"

def process_gsm8k():
    print("Downloading GSM8K dataset...")
    # Load GSM8K (main config)
    ds = datasets.load_dataset("gsm8k", "main")
    
    train_data = []
    for example in ds['train']:
        prompt = make_gsm8k_prompt(example['question'])
        # Extract numerical answer or keep full solution for reward model reference
        ground_truth = example['answer'] 
        
        train_data.append({
            "prompt": prompt,
            "reward_model": {"ground_truth": ground_truth, "target": ground_truth.split("#### ")[-1].strip()},
            "id": f"gsm8k_train_{len(train_data)}"
        })

    test_data = []
    for example in ds['test']:
        prompt = make_gsm8k_prompt(example['question'])
        ground_truth = example['answer']
        
        test_data.append({
            "prompt": prompt,
            "reward_model": {"ground_truth": ground_truth, "target": ground_truth.split("#### ")[-1].strip()},
            "id": f"gsm8k_test_{len(test_data)}"
        })

    return pd.DataFrame(train_data), pd.DataFrame(test_data)

def generate_synthetic_sft():
    print(f"Generating {SFT_SIZE} synthetic SFT examples...")
    
    # Template for CoDA format (Extend Phase)
    # Teaching the model to use <Planner> and <Executor> tags
    
    templates = [
        {
            "q": "What is 25 * 4 + 10?",
            "a": """<Planner>
I need to calculate 25 * 4 first, then add 10.
<run>
<Executor>
print(25 * 4 + 10)
</Executor>
</run>
The answer is 110.
</Planner>"""
        },
        {
            "q": "Solve 150 / 3 - 20",
            "a": """<Planner>
First I divide 150 by 3, which is 50. Then I subtract 20.
<run>
<Executor>
print(150 / 3 - 20)
</Executor>
</run>
The result is 30.
</Planner>"""
        },
        {
             "q": "Who is the current president of France?",
             "a": """<Planner>
I need to search for the current president of France.
<run>
<Executor>
search("current president of France")
</Executor>
</run>
Based on the search results, Emmanuel Macron is the president.
</Planner>"""
        }
    ]
    
    sft_data = []
    for i in range(SFT_SIZE):
        # Determine strict SFT formatting
        # Randomly pick a template to simulate variety (in real scenario, use diverse distilled data)
        tmpl = templates[i % len(templates)]
        
        entry = {
            "prompt": f"Question: {tmpl['q']}\nAnswer:",
            "response": tmpl['a'],
            "id": f"sft_syn_{i}"
        }
        sft_data.append(entry)
        
    return pd.DataFrame(sft_data)

def save_parquet(df, filename):
    table = pa.Table.from_pandas(df)
    os.makedirs("data", exist_ok=True)
    pq.write_table(table, f"data/{filename}")
    print(f"Saved {len(df)} rows to data/{filename}")

if __name__ == "__main__":
    # 1. RL Data (Recall Phase)
    train_df, item_df = process_gsm8k()
    
    # Split some train for validation if needed, or use test set
    # Using full train for RL training
    save_parquet(train_df, "train.parquet")
    
    # Use valid/test set for evaluation
    save_parquet(item_df, "valid.parquet")
    # Also save a smaller subset for quick validation
    save_parquet(item_df.head(500), "valid_500.parquet")

    # 2. SFT Data (Extend Phase)
    sft_df = generate_synthetic_sft()
    save_parquet(sft_df, "sft_train.parquet")

    print("\n✅ All datasets prepared successfully!")
    print("1. data/train.parquet       (RL Training - GSM8K)")
    print("2. data/valid.parquet       (RL Validation)")
    print("3. data/sft_train.parquet   (SFT / RED Extend - Synthetic CoDA Format)")
