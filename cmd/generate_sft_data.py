#!/usr/bin/env python3
"""Generate SFT training data for CoDA.
Creates expert-style trajectories showing the CoDA XML workflow format.
"""
import pandas as pd
import json
import random

def create_sft_examples_from_train(train_path='data/train.parquet', output_path='data/sft_train.parquet', n_samples=500):
    """Create SFT examples from training data to teach the model XML format."""
    
    df = pd.read_parquet(train_path)
    print(f"Loaded {len(df)} rows from {train_path}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Sample a subset
    if len(df) > n_samples:
        df = df.sample(n=n_samples, random_state=42)
    
    sft_data = []
    
    for idx, row in df.iterrows():
        # Extract question from prompt (list of chat messages)
        prompt_data = row.get('prompt', '')
        if isinstance(prompt_data, list):
            # Chat format: [{'content': '...', 'role': 'user'}]
            question = prompt_data[0]['content'] if prompt_data else ''
        elif isinstance(prompt_data, str):
            question = prompt_data
        else:
            question = str(prompt_data)
        
        if not question:
            continue
        
        # Extract target answer from reward_model.ground_truth.target
        reward_model = row.get('reward_model', {})
        if isinstance(reward_model, str):
            try:
                reward_model = json.loads(reward_model)
            except:
                reward_model = {}
        
        target = ''
        if isinstance(reward_model, dict):
            gt = reward_model.get('ground_truth', {})
            if isinstance(gt, dict):
                t = gt.get('target', gt.get('answer', ''))
                if isinstance(t, list):
                    target = t[0] if t else ''
                else:
                    target = str(t)
        
        # Fallback: try golden_answers
        if not target:
            golden = row.get('golden_answers', [])
            if isinstance(golden, list) and golden:
                target = golden[0]
            elif isinstance(golden, str):
                target = golden
        
        if not target:
            continue
            
        # Create expert trajectory in CoDA XML format
        search_query = question[:80]
        
        # 1. Construct Full Prompt with System Prompt (Must match rl_dataset.py)
        system_prompt = """You are a helpful assistant excel at answering questions with multi-turn search engine calling.
To answer questions, you must first reason through the available information using <think> and </think>.
If you identify missing knowledge, you may issue a search request using <search> query </search> at any time.
The retrieval system will provide you with the most relevant documents enclosed in <documents> and </documents>.
After each search, you need to summarize and refine the existing documents in <refine> and </refine>.
You may send multiple search requests if needed.
Once you have sufficient information, provide a concise final answer using <answer> and </answer>. For example, <answer> Donald Trump </answer>."""
        
        full_prompt = f"{system_prompt}\nQuestion: {question}\nAssistant: "

        # 2. Construct Response with Mock Documents (Environment simulation)
        response = f"""<think>
I need to find information to answer this question. Let me search for relevant documents.
</think>
<search>{search_query}</search>
<documents>
Title: {search_query}
Content: {target} is the answer to the question "{question}". This document confirms that {target} matches the query.
</documents>
<refine>
Based on the retrieved documents, I found that the answer is {target}.
</refine>
<think>
Now I have enough information to answer the question.
</think>
<answer>{target}</answer>"""
        
        sft_data.append({
            'prompt': full_prompt,
            'response': response
        })
    
    sft_df = pd.DataFrame(sft_data)
    sft_df.to_parquet(output_path, index=False)
    print(f"Created {len(sft_df)} SFT examples → {output_path}")
    return sft_df

if __name__ == '__main__':
    create_sft_examples_from_train()
