
import os
import datasets
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# --- Configuration ---
SFT_SIZE = 2000  # Increased SFT size for better coverage
SEED = 42

def make_gsm8k_prompt(question):
    return f"Question: {question}\nLet's think step by step to solve this.\n"

def make_qa_prompt(question):
    return f"Question: {question}\nAnswer this question by thinking step by step and using search tools if necessary.\n"

def process_gsm8k():
    print("Processing GSM8K dataset...")
    ds = datasets.load_dataset("gsm8k", "main")
    
    data = []
    for example in ds['train']:
        prompt = make_gsm8k_prompt(example['question'])
        ground_truth = example['answer']
        data.append({
            "prompt": prompt,
            "reward_model": {"ground_truth": ground_truth, "target": ground_truth.split("#### ")[-1].strip()},
            "id": f"gsm8k_{len(data)}",
            "source": "gsm8k"
        })
    return pd.DataFrame(data)

def process_hotpotqa():
    print("Processing HotpotQA (distractor) dataset...")
    # Load distractor config (smaller than fullwiki but good for multi-hop)
    ds = datasets.load_dataset("hotpot_qa", "distractor")
    
    data = []
    # Filter for 'hard' or 'bridge' questions if available, but distractor is already multihop
    for example in ds['train']:
        # Only take questions with answers (not yes/no for now to simplify reward)
        if example['answer'] and example['answer'].lower() not in ['yes', 'no']:
            prompt = make_qa_prompt(example['question'])
            ground_truth = example['answer']
            data.append({
                "prompt": prompt,
                "reward_model": {"ground_truth": ground_truth, "target": ground_truth},
                "id": f"hotpotqa_{example['id']}",
                "source": "hotpotqa"
            })
            
            if len(data) >= 5000: # Limit to 5k to balance with GSM8K
                break
    return pd.DataFrame(data)

def process_musique():
    print("Processing Musique dataset...")
    try:
        ds = datasets.load_dataset("musique", "answerable") # 'answerable' subset
    except Exception as e:
        print(f"Skipping Musique due to load error (might require manual download): {e}")
        return pd.DataFrame()

    data = []
    for example in ds['train']:
        if len(example['answer']) < 50: # Filter short answers
            prompt = make_qa_prompt(example['question'])
            ground_truth = example['answer']
            data.append({
                "prompt": prompt,
                "reward_model": {"ground_truth": ground_truth, "target": ground_truth},
                "id": f"musique_{example['id']}",
                "source": "musique"
            })
            
            if len(data) >= 2000: # Limit count
                break
    return pd.DataFrame(data)

def generate_adv_synthetic_sft():
    print(f"Generating {SFT_SIZE} Advanced Synthetic SFT examples...")
    
    # 1. Math Reasoning Template (GSM8K Style)
    math_templates = [
        {"q": "What is 25 * 4 + 10?", "a": "<Planner>\nI need to calculate 25 * 4 first, then add 10.\n<run>\n<Executor>\nprint(25 * 4 + 10)\n</Executor>\n</run>\nThe answer is 110.\n</Planner>"},
        {"q": "Solve 150 / 3 - 20", "a": "<Planner>\nFirst I divide 150 by 3, which is 50. Then I subtract 20.\n<run>\n<Executor>\nprint(150 / 3 - 20)\n</Executor>\n</run>\nThe result is 30.\n</Planner>"}
    ]
    
    # 2. Multi-hop Search Template (HotpotQA/Musique Style)
    # Teaching: Planner -> Search -> Observation -> Planner -> Answer
    search_templates = [
        {
            "q": "Which city is the birthplace of the painter of 'Mona Lisa'?",
            "a": """<Planner>
I need to find out who painted the 'Mona Lisa' first.
<run>
<Executor>
search("painter of Mona Lisa")
</Executor>
</run>
Observation: Leonardo da Vinci painted the Mona Lisa.
Now I need to find the birthplace of Leonardo da Vinci.
<run>
<Executor>
search("birthplace of Leonardo da Vinci")
</Executor>
</run>
Observation: Leonardo da Vinci was born in Vinci, Italy.
The answer is Vinci.
</Planner>"""
        },
        {
            "q": "Who is the director of the movie starring Tom Hanks as Forrest Gump?",
            "a": """<Planner>
I need to check the cast of the movie 'Forrest Gump' to confirm the role, then find the director.
<run>
<Executor>
search("director of movie Forrest Gump classification")
</Executor>
</run>
Observation: Forrest Gump is a 1994 film directed by Robert Zemeckis.
The answer is Robert Zemeckis.
</Planner>"""
        },
        {
             "q": "What represents the capital of the country where the Eiffel Tower is located?",
             "a": """<Planner>
First, I need to locate the Eiffel Tower.
<run>
<Executor>
search("location of Eiffel Tower")
</Executor>
</run>
Observation: The Eiffel Tower is located in Paris, France.
Now I need to find the capital of France.
<run>
<Executor>
search("capital of France")
</Executor>
</run>
Observation: Paris is the capital of France.
The answer is Paris.
</Planner>"""
        }
    ]

    sft_data = []
    
    # Generate Math examples (30%)
    for i in range(int(SFT_SIZE * 0.3)):
        tmpl = math_templates[i % len(math_templates)]
        entry = {"prompt": f"Question: {tmpl['q']}\nAnswer:", "response": tmpl['a'], "id": f"sft_math_{i}"}
        sft_data.append(entry)

    # Generate Search examples (70%) - Critical for HotpotQA/Musique skills
    for i in range(int(SFT_SIZE * 0.7)):
        tmpl = search_templates[i % len(search_templates)]
        entry = {"prompt": f"Question: {tmpl['q']}\nAnswer:", "response": tmpl['a'], "id": f"sft_search_{i}"}
        sft_data.append(entry)
        
    # Shuffle
    df = pd.DataFrame(sft_data)
    return df.sample(frac=1, random_state=SEED).reset_index(drop=True)

def save_parquet(df, filename):
    table = pa.Table.from_pandas(df)
    os.makedirs("data", exist_ok=True)
    pq.write_table(table, f"data/{filename}")
    print(f"Saved {len(df)} rows to data/{filename}")

if __name__ == "__main__":
    # 1. RL Data Collection (Recall Phase)
    print("--- 1. Gathering Recall Phase Data (RL) ---")
    gsm8k_df = process_gsm8k()
    hotpot_df = process_hotpotqa()
    # musique_df = process_musique() # Uncomment if you want to try downloading Musique
    
    # Combine Datasets
    combined_df = pd.concat([gsm8k_df, hotpot_df], ignore_index=True)
    combined_df = combined_df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    
    print(f"Total RL Training Data: {len(combined_df)} samples")
    print(f"- GSM8K: {len(gsm8k_df)}")
    print(f"- HotpotQA: {len(hotpot_df)}")
    
    # Split Train/Valid (95/5)
    valid_size = int(len(combined_df) * 0.05)
    train_df = combined_df[:-valid_size]
    valid_df = combined_df[-valid_size:]
    
    save_parquet(train_df, "train.parquet")
    save_parquet(valid_df, "valid.parquet")
    save_parquet(valid_df.head(500), "valid_500.parquet")

    # 2. SFT Data Generation (Extend Phase)
    print("\n--- 2. Generating Extend Phase Data (SFT) ---")
    sft_df = generate_adv_synthetic_sft()
    save_parquet(sft_df, "sft_train.parquet")

    print("\n✅ All datasets prepared successfully! (Professional Research Ready)")
    print("1. data/train.parquet       (Mixed GSM8K + HotpotQA for RL)")
    print("2. data/sft_train.parquet   (Multi-hop Reasoning + Math SFT)")
    print("3. data/valid.parquet       (Validation Set)")
