#!/usr/bin/env python3
"""Generate SFT training data for CoDA.
Creates expert-style trajectories with type-aware patterns:
- Math questions (GSM8K) → Direct CoT reasoning, no search
- QA questions (HotpotQA) → Search-based patterns with CoT
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

NOISE_DOCS = [
    "Title: Popular Culture\nContent: This article discusses various topics in popular culture and entertainment that are not directly related to the query.",
    "Title: Historical Overview\nContent: A broad overview of historical events spanning multiple centuries. Further research may be needed for specific details.",
    "Title: General Knowledge\nContent: This document contains general knowledge information that may or may not be relevant to the specific question asked.",
]


# =============================================================================
# MATH PATTERNS (for GSM8K — no search needed)
# =============================================================================

def make_math_direct_cot(question, target, search_query):
    """Math: Direct CoT reasoning → answer (70% of math)"""
    cot = random.choice([
        f'Let me solve this step by step.\n\nQuestion: "{question}"\n\nI need to work through the math carefully:\n- First, identify the key numbers and operations\n- Then calculate step by step\n- The answer should be {target}',
        f'This is a math problem. Let me reason through it:\n\nGiven: "{question}"\n\nStep 1: Identify what we need to find\nStep 2: Set up the calculation\nStep 3: Compute the result\n\nAfter working through the math, the answer is {target}.',
        f'Let me break down this math problem: "{question}"\n\nTo solve this:\n1. Parse the numbers and relationships\n2. Apply the correct operations\n3. Calculate carefully\n\nThe final answer is {target}.',
        f'I can solve this mathematically without searching.\n\nProblem: "{question}"\n\nWorking through the calculations:\n- Analyzing the given information\n- Performing the arithmetic\n- Verifying the result\n\nThe answer is {target}.',
    ])
    
    response = f"""<think>
{cot}
</think>
<answer>{target}</answer>"""
    return response


def make_math_with_verification(question, target, search_query):
    """Math: Think → answer → verify (30% of math)"""
    cot_solve = random.choice([
        f'Let me work through this problem: "{question}"\n\nI need to:\n1. Understand what\'s being asked\n2. Identify the relevant numbers\n3. Perform the calculation step by step',
        f'This math problem requires careful calculation.\n\nQuestion: "{question}"\n\nLet me identify the approach and solve it systematically.',
    ])
    
    cot_verify = random.choice([
        f'Let me verify my answer of {target}.\n\nDouble-checking the calculation:\n- Re-reading the question to make sure I understood correctly\n- Verifying each step of my computation\n- The answer {target} is confirmed correct.',
        f'I should double-check: is {target} really the right answer?\n\nVerification:\n- Going back through each step\n- The math checks out\n- I\'m confident the answer is {target}.',
    ])
    
    response = f"""<think>
{cot_solve}
</think>
<think>
{cot_verify}
</think>
<answer>{target}</answer>"""
    return response


# =============================================================================
# QA PATTERNS (for HotpotQA — search-based)
# =============================================================================

def make_qa_single_search(question, target, search_query):
    """QA: Single search → refine → answer (50% of QA)"""
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


def make_qa_multi_hop(question, target, search_query):
    """QA: Two searches → multi-hop reasoning (30% of QA)"""
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


def make_qa_noisy_docs(question, target, search_query):
    """QA: Search with noisy docs, must filter (20% of QA)"""
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


# =============================================================================
# Main
# =============================================================================

MATH_PATTERNS = [
    (make_math_direct_cot,        0.70),  # 70% direct CoT
    (make_math_with_verification, 0.30),  # 30% CoT + verify
]

QA_PATTERNS = [
    (make_qa_single_search, 0.50),  # 50% single search
    (make_qa_multi_hop,     0.30),  # 30% multi-hop
    (make_qa_noisy_docs,    0.20),  # 20% noisy docs
]


def detect_question_type(row):
    """Detect if question is math (GSM8K) or QA (HotpotQA)."""
    qid = str(row.get('id', row.get('extra_info', {}).get('id', '')))
    if 'gsm8k' in qid.lower():
        return 'math'
    # Check if answer looks numeric
    reward_model = row.get('reward_model', {})
    if isinstance(reward_model, str):
        try:
            reward_model = json.loads(reward_model)
        except:
            reward_model = {}
    if isinstance(reward_model, dict):
        gt = reward_model.get('ground_truth', {})
        if isinstance(gt, dict):
            t = gt.get('target', gt.get('answer', ''))
            if isinstance(t, list):
                t = t[0] if t else ''
            t = str(t).strip()
            # Numeric answers are likely math
            try:
                float(t.replace(',', '').replace('$', '').replace('%', ''))
                return 'math'
            except:
                pass
    return 'qa'


def select_pattern(patterns):
    """Weighted random selection from pattern list."""
    r = random.random()
    cumulative = 0
    for fn, weight in patterns:
        cumulative += weight
        if r <= cumulative:
            return fn
    return patterns[0][0]


def create_sft_examples_from_train(train_path='data/train.parquet', output_path='data/sft_train.parquet', n_samples=1500):
    """Create SFT examples with type-aware patterns."""
    
    df = pd.read_parquet(train_path)
    print(f"Loaded {len(df)} rows from {train_path}")
    
    if len(df) > n_samples:
        df = df.sample(n=n_samples, random_state=42)
    
    sft_data = []
    type_counts = {'math': 0, 'qa': 0}
    pattern_counts = {}
    
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
        
        # Detect type and select appropriate pattern
        q_type = detect_question_type(row)
        type_counts[q_type] += 1
        
        if q_type == 'math':
            selected_fn = select_pattern(MATH_PATTERNS)
        else:
            selected_fn = select_pattern(QA_PATTERNS)
        
        fn_name = selected_fn.__name__
        pattern_counts[fn_name] = pattern_counts.get(fn_name, 0) + 1
        
        search_query = question[:80]
        full_prompt = f"{SYSTEM_PROMPT}\nQuestion: {question}\nAssistant: "
        response = selected_fn(question, target, search_query)
        
        sft_data.append({
            'prompt': full_prompt,
            'response': response,
        })
    
    sft_df = pd.DataFrame(sft_data)
    sft_df.to_parquet(output_path, index=False)
    
    print(f"\n✅ Created {len(sft_df)} SFT examples → {output_path}")
    print(f"\nType distribution:")
    for t, count in type_counts.items():
        pct = count / len(sft_df) * 100 if len(sft_df) > 0 else 0
        print(f"  {t}: {count} ({pct:.0f}%)")
    print(f"\nPattern distribution:")
    for name, count in sorted(pattern_counts.items()):
        pct = count / len(sft_df) * 100 if len(sft_df) > 0 else 0
        print(f"  {name}: {count} ({pct:.0f}%)")
    
    return sft_df

if __name__ == '__main__':
    random.seed(42)
    create_sft_examples_from_train()
