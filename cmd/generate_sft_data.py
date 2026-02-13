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





def make_pattern_single_search(question, target, search_query):
    """Pattern 1: Single search → refine → answer (60% of data)"""
    cot_plan = random.choice([
        f'Let me break down this question: "{question}"\nI need to identify the key entity and find specific information about it. The most effective approach would be to search directly for the core topic.',
        f'To answer "{question}", I need to:\n1. Identify what specific information is being asked\n2. Search for relevant documents\n3. Extract the answer from reliable sources',
        f'This question asks about "{question}". I should search for the main subject to find the relevant facts. Let me construct an effective search query.',
        f'Analyzing the question: "{question}"\nThis requires factual knowledge that I should verify. Let me search for the key terms to find accurate information.',
    ])
    
    cot_conclude = random.choice([
        f'The search results clearly indicate that {target} is the correct answer. The retrieved document directly addresses the question with supporting evidence.',
        f'After reviewing the documents, I can confirm the answer is {target}. The evidence from the search results is consistent and reliable.',
        f'Based on my analysis of the retrieved information, {target} answers the question. The source is credible and the information is specific.',
        f'The documents confirm that {target} is the answer. Let me provide this as my final response.',
    ])
    
    response = f"""<think>
{cot_plan}
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
{cot_conclude}
</think>
<answer>{target}</answer>"""
    return response


def make_pattern_multi_hop(question, target, search_query):
    """Pattern 2: Two searches → multi-hop reasoning (20% of data)"""
    words = search_query.split()
    mid = max(len(words) // 2, 2)
    query1 = ' '.join(words[:mid])
    query2 = search_query
    
    noise = random.choice(NOISE_DOCS)
    
    cot_plan = random.choice([
        f'This question seems to require multiple pieces of information: "{question}"\nLet me start by searching for the broader topic, then narrow down to specific details.',
        f'To answer "{question}", I may need to gather information from multiple sources.\nStep 1: Search for background information\nStep 2: Search for specific details\nLet me start with a broad search.',
        f'Analyzing this question, it likely requires connecting multiple facts.\nI\'ll approach this systematically - first search for general context, then drill down into specifics.',
    ])
    
    cot_middle = random.choice([
        f'The first search gave me some background context, but the documents didn\'t contain the specific answer I need.\nI should refine my search query to be more targeted and include the full question.',
        f'Hmm, the initial results are too general. I have some context but need more specific information.\nLet me try a more detailed search query that captures the full intent of the question.',
        f'The background information is helpful but not sufficient to answer the question directly.\nI need to search again with a more specific query to find the exact answer.',
    ])
    
    cot_conclude = random.choice([
        f'Now I have the complete picture. After combining information from both searches:\n- First search provided background context\n- Second search confirmed that {target} is the answer\nI\'m confident in this answer.',
        f'After two rounds of searching, I\'ve gathered enough evidence.\nThe key finding from my research is that the answer is {target}. Multiple sources support this conclusion.',
        f'Excellent! The second search provided the definitive answer. By connecting the background from my first search with the specific details from the second, I can confirm that {target} is correct.',
    ])
    
    response = f"""<think>
{cot_plan}
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
{cot_middle}
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
{cot_conclude}
</think>
<answer>{target}</answer>"""
    return response


def make_pattern_noisy_docs(question, target, search_query):
    """Pattern 3: Search with noisy/irrelevant docs mixed in (10% of data)"""
    noise1 = random.choice(NOISE_DOCS)
    noise2 = random.choice(NOISE_DOCS)
    
    cot_plan = random.choice([
        f'I need to find information about: "{question}"\nI should search and then carefully evaluate the relevance of each returned document.',
        f'Let me search for the answer to "{question}". I\'ll need to be careful to distinguish relevant information from noise in the results.',
    ])
    
    cot_conclude = random.choice([
        f'The search returned several documents, but not all are relevant.\nLet me filter:\n- Document 1: Not directly relevant (general topic)\n- Document 2: Not directly relevant (general topic)\n- Document 3: Directly answers the question → {target}\nI\'ll use the most relevant document for my answer.',
        f'I need to evaluate the quality of these search results:\n- Some documents are too general and don\'t address the specific question\n- The last document specifically confirms that {target} is the answer\nI should rely on the most specific and relevant source.',
    ])
    
    response = f"""<think>
{cot_plan}
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
{cot_conclude}
</think>
<answer>{target}</answer>"""
    return response


def make_pattern_direct_answer(question, target, search_query):
    """Pattern 4: Direct answer with CoT reasoning, no search (10% of data)"""
    cot_direct = random.choice([
        f'Let me think about this question: "{question}"\n\nBreaking it down:\n1. The question asks for a specific piece of information\n2. Based on the context and information available, the answer is {target}\n3. I\'m confident enough to answer directly without searching.',
        f'Analyzing: "{question}"\n\nI can reason through this:\n- The question is asking for a factual answer\n- From what I know, {target} is the correct answer\n- No additional search is needed as this is well-established information.',
        f'This is a question I can answer through reasoning: "{question}"\n\nMy thought process:\n- Identifying what\'s being asked\n- The answer should be {target} based on available knowledge\n- I don\'t need to search for additional verification.',
    ])
    
    response = f"""<think>
{cot_direct}
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
