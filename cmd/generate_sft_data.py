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
    
    # Sample a subset
    if len(df) > n_samples:
        df = df.sample(n=n_samples, random_state=42)
    
    sft_data = []
    
    for _, row in df.iterrows():
        prompt_text = row.get('prompt', '')
        
        # Extract the question from the prompt
        # The prompt usually contains the question in the system message
        question = prompt_text if isinstance(prompt_text, str) else str(prompt_text)
        
        # Get the ground truth answer
        extra_info = row.get('extra_info', {})
        if isinstance(extra_info, str):
            try:
                extra_info = json.loads(extra_info)
            except:
                extra_info = {}
        
        if isinstance(extra_info, dict):
            target = extra_info.get('target', extra_info.get('answer', ''))
        else:
            target = str(extra_info) if extra_info else ''
        
        if not target:
            continue
            
        # Create expert trajectory in CoDA XML format
        response = f"""<think>
I need to find information to answer this question. Let me search for relevant documents.
</think>
<search>
{target[:50]}
</search>
<information>
Based on the retrieved documents, I found relevant information about the topic.
</information>
<think>
Now I have enough information to answer the question. The answer is {target}.
</think>
<finish>
{target}
</finish>"""
        
        sft_data.append({
            'prompt': question,
            'response': response
        })
    
    sft_df = pd.DataFrame(sft_data)
    sft_df.to_parquet(output_path, index=False)
    print(f"Created {len(sft_df)} SFT examples → {output_path}")
    return sft_df

if __name__ == '__main__':
    create_sft_examples_from_train()
