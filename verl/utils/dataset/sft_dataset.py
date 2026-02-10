
# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from typing import List, Union

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

class SFTDataset(Dataset):
    def __init__(self,
                 parquet_files: Union[str, List[str]],
                 tokenizer: PreTrainedTokenizer,
                 prompt_key='prompt',
                 response_key='response',
                 max_length=1024,
                 truncation=True):
        
        if isinstance(parquet_files, str):
            parquet_files = [parquet_files]

        self.parquet_files = parquet_files
        self.tokenizer = tokenizer
        self.prompt_key = prompt_key
        self.response_key = response_key
        self.max_length = max_length
        self.truncation = truncation

        self._read_files()

    def _read_files(self):
        dataframes = []
        for parquet_file in self.parquet_files:
            # Check if file exists to avoid crashes
            if os.path.exists(parquet_file):
                df = pd.read_parquet(parquet_file)
                dataframes.append(df)
        
        if not dataframes:
            self.data = pd.DataFrame()
            print(f"Warning: No data loaded from {self.parquet_files}")
        else:
            self.data = pd.concat(dataframes, ignore_index=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        prompt = row[self.prompt_key]
        response = row[self.response_key]

        # Tokenize prompt and response
        # Note: We don't add special tokens here to control concatenation manually
        # But for chat models, we might need apply_chat_template if raw text.
        # Assuming prepare_data.py handles formatting or we use raw text.
        
        # For base models or simple SFT, just concat text? 
        # Better to tokenize separately to know lengths for masking.
        
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        response_ids = self.tokenizer.encode(response, add_special_tokens=False)
        response_ids.append(self.tokenizer.eos_token_id)

        input_ids = prompt_ids + response_ids
        
        # Create masks
        # 1 for response (calculate loss), 0 for prompt
        loss_mask = [0] * len(prompt_ids) + [1] * len(response_ids)
        attention_mask = [1] * len(input_ids)

        # Truncation / Padding
        if len(input_ids) > self.max_length:
            if self.truncation:
                # Truncate from the end? Or keep prompt?
                # Usually keep prompt and truncate response? 
                # For simplicity, just truncate end.
                input_ids = input_ids[:self.max_length]
                loss_mask = loss_mask[:self.max_length]
                attention_mask = attention_mask[:self.max_length]
        else:
            # Pad to max_length (right padding)
            pad_len = self.max_length - len(input_ids)
            input_ids += [self.tokenizer.pad_token_id] * pad_len
            loss_mask += [0] * pad_len
            attention_mask += [0] * pad_len

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'loss_mask': torch.tensor(loss_mask, dtype=torch.long),
            # Position IDs typically handled by model or collator, but we can provide simple ones
            'position_ids': torch.arange(len(input_ids), dtype=torch.long)
        }
