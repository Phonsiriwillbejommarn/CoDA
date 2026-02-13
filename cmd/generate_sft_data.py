#!/usr/bin/env python3
"""Generate SFT training data for CoDA.
Creates expert-style trajectories showing the CoDA XML workflow format.
Supports multiple patterns: single-search, multi-hop, noisy-documents, direct-answer.
"""
import pandas as pd
import json
import random

SYSTEM_PROMPT = """You are a helpful assistant excel at answering questions with multi-turn search engine calling.
To answer questions, you must first reason through the available information using <think> and </think>.
If you identify missing knowledge, you may issue a search request using <search> query </search> at any time.
The retrieval system will provide you with the most relevant documents enclosed in <documents> and </documents>.
After each search, you need to summarize and refine the existing documents in <refine> and </refine>.
You may send multiple search requests if needed.
Once you have sufficient information, provide a concise final answer using <answer> and </answer>. For example, <answer> Donald Trump </answer>."""

# Noisy document templates to mix with real info
NOISE_DOCS = [
    "Title: Popular Culture\nContent: This article discusses various topics in popular culture and entertainment that are not directly related to the query.",
    "Title: Historical Overview\nContent: A broad overview of historical events spanning multiple centuries. Further research may be needed for specific details.",
    "Title: General Knowledge\nContent: This document contains general knowledge information that may or may not be relevant to the specific question asked.",
]

THINK_TEMPLATES = {
    'plan_search': [
        "I need to find information to answer this question. Let me search for relevant documents.",
        "Let me search for specific information about this topic.",
        "I don't have enough information yet. Let me search for this.",
        "To answer accurately, I should search for the relevant facts.",
    ],
    'need_more': [
        "The search results gave me partial information, but I need to search for more details.",
        "I found some relevant information, but I need another search to confirm.",
        "Let me search for additional details to answer this completely.",
        "I have some clues but need to refine my search with a different query.",
    ],
    'ready_answer': [
        "Now I have enough information to answer the question.",
        "Based on the retrieved documents, I can now provide the answer.",
        "I've gathered sufficient information. Let me provide the final answer.",
        "The search results confirm the answer. Let me respond.",
    ],
    'direct_know': [
        "This is a straightforward question that I can answer directly based on the information given.",
        "I already know the answer to this question from the context provided.",
        "This question can be answered directly without additional searching.",
    ],
}

def pick(templates):
    return random.choice(templates)


def make_pattern_single_search(question, target, search_query):
    """Pattern 1: Single search → refine → answer (60% of data)"""
    response = f"""<think>
{pick(THINK_TEMPLATES['plan_search'])}
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
{pick(THINK_TEMPLATES['ready_answer'])}
</think>
<answer>{target}</answer>"""
    return response


def make_pattern_multi_hop(question, target, search_query):
    """Pattern 2: Two searches → multi-hop reasoning (20% of data)"""
    # Split query for two-step search
    words = search_query.split()
    mid = max(len(words) // 2, 2)
    query1 = ' '.join(words[:mid])
    query2 = search_query  # Full query for second search
    
    noise = random.choice(NOISE_DOCS)
    
    response = f"""<think>
{pick(THINK_TEMPLATES['plan_search'])}
</think>
<search>{query1}</search>
<documents>
{noise}
Title: Background on {query1}
Content: This article provides background information related to {query1}. For more specific details, additional research may be needed.
</documents>
<refine>
The initial search provided some background, but I need more specific information to answer the question.
</refine>
<think>
{pick(THINK_TEMPLATES['need_more'])}
</think>
<search>{query2}</search>
<documents>
Title: {search_query}
Content: According to reliable sources, {target} is the answer. The document states that regarding "{question}", the correct answer is {target}.
</documents>
<refine>
After two searches, I now have the complete information. The answer to the question is {target}.
</refine>
<think>
{pick(THINK_TEMPLATES['ready_answer'])}
</think>
<answer>{target}</answer>"""
    return response


def make_pattern_noisy_docs(question, target, search_query):
    """Pattern 3: Search with noisy/irrelevant docs mixed in (10% of data)"""
    noise1 = random.choice(NOISE_DOCS)
    noise2 = random.choice(NOISE_DOCS)
    
    response = f"""<think>
{pick(THINK_TEMPLATES['plan_search'])}
</think>
<search>{search_query}</search>
<documents>
{noise1}
{noise2}
Title: {search_query}
Content: {target} is confirmed as the answer to "{question}". Multiple sources verify this information.
</documents>
<refine>
The search returned several documents. After filtering out irrelevant results, the key finding is that the answer is {target}.
</refine>
<think>
{pick(THINK_TEMPLATES['ready_answer'])}
</think>
<answer>{target}</answer>"""
    return response


def make_pattern_direct_answer(question, target, search_query):
    """Pattern 4: Direct answer without search (10% of data)"""
    response = f"""<think>
{pick(THINK_TEMPLATES['direct_know'])}
</think>
<answer>{target}</answer>"""
    return response


def create_sft_examples_from_train(train_path='data/train.parquet', output_path='data/sft_train.parquet', n_samples=500):
    """Create SFT examples from training data with multiple patterns."""
    
    df = pd.read_parquet(train_path)
    print(f"Loaded {len(df)} rows from {train_path}")
    
    if len(df) > n_samples:
        df = df.sample(n=n_samples, random_state=42)
    
    # Pattern distribution
    patterns = [
        (make_pattern_single_search, 0.60),  # 60% single search
        (make_pattern_multi_hop,     0.20),  # 20% multi-hop
        (make_pattern_noisy_docs,    0.10),  # 10% noisy docs
        (make_pattern_direct_answer, 0.10),  # 10% direct answer
    ]
    
    sft_data = []
    pattern_counts = {fn.__name__: 0 for fn, _ in patterns}
    
    for idx, row in df.iterrows():
        # Extract question
        prompt_data = row.get('prompt', '')
        if isinstance(prompt_data, list):
            question = prompt_data[0]['content'] if prompt_data else ''
        elif isinstance(prompt_data, str):
            question = prompt_data
        else:
            question = str(prompt_data)
        
        if not question:
            continue
        
        # Extract target answer
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
        
        if not target:
            golden = row.get('golden_answers', [])
            if isinstance(golden, list) and golden:
                target = golden[0]
            elif isinstance(golden, str):
                target = golden
        
        if not target:
            continue
        
        # Select pattern by weighted random
        r = random.random()
        cumulative = 0
        selected_fn = make_pattern_single_search
        for fn, weight in patterns:
            cumulative += weight
            if r <= cumulative:
                selected_fn = fn
                break
        
        search_query = question[:80]
        full_prompt = f"{SYSTEM_PROMPT}\nQuestion: {question}\nAssistant: "
        response = selected_fn(question, target, search_query)
        
        sft_data.append({
            'prompt': full_prompt,
            'response': response,
        })
        pattern_counts[selected_fn.__name__] += 1
    
    sft_df = pd.DataFrame(sft_data)
    sft_df.to_parquet(output_path, index=False)
    
    print(f"\nCreated {len(sft_df)} SFT examples → {output_path}")
    print("Pattern distribution:")
    for name, count in pattern_counts.items():
        pct = count / len(sft_df) * 100 if sft_df.shape[0] > 0 else 0
        print(f"  {name}: {count} ({pct:.0f}%)")
    
    return sft_df

if __name__ == '__main__':
    random.seed(42)
    create_sft_examples_from_train()
