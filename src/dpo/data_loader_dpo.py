# src/dpo/data_loader_dpo.py

from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import pandas as pd
from src.dpo.utils.config_dpo import CONFIG

class DPODataset(Dataset):
    def __init__(self, file_path, tokenizer):
        self.tokenizer = tokenizer
        self.data = pd.read_json(file_path, lines=True)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        return self.preprocess_dpo_data(row)
    
    def preprocess_dpo_data(self, row):
        # 1) Build the prompt prefix from a template in CONFIG
        prefix_template = CONFIG.get(
            "prompt_template",
            "Premise: {premise}\nInitial: {initial}\nCounterfactual: {counterfactual}\nGenerate an ending:"
        )
        prefix = prefix_template.format(
            premise=row["premise"],
            initial=row["initial"],
            counterfactual=row["counterfactual"],
        )

        # 2) Tokenize the prefix alone to get its true length
        context_enc = self.tokenizer(
            prefix,
            max_length=CONFIG["max_length"],
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        context_input_ids      = context_enc["input_ids"].squeeze(0)
        context_attention_mask = context_enc["attention_mask"].squeeze(0)
        prefix_len = int(context_attention_mask.sum().item())

        # 3) Helper to tokenize (prefix, completion) and build labels
        def tokenize_completion(text: str):
            enc = self.tokenizer(
                prefix,
                text,
                max_length=CONFIG["max_length"],
                truncation="only_second",
                padding="max_length",
                return_tensors="pt"
            )
            input_ids      = enc["input_ids"].squeeze(0)
            attention_mask = enc["attention_mask"].squeeze(0)

            # Mask out prefix tokens and padding in labels
            labels = input_ids.clone()
            labels[:prefix_len] = -100            # ignore prefix
            labels[attention_mask == 0] = -100    # ignore padding

            return input_ids, attention_mask, labels

        # 4) Tokenize both chosen (edited) and rejected (original) endings
        chosen_input_ids, chosen_attention_mask, chosen_labels = tokenize_completion(str(row["edited_ending"]))
        rej_input_ids,   rej_attention_mask,   rej_labels   = tokenize_completion(str(row["original_ending"]))

        # 5) Return dict with clear keys
        return {
            # for generation (prefix only)
            "input_ids":      context_input_ids,
            "attention_mask": context_attention_mask,

            # for DPO loss (prefix + completion)
            "chosen_input_ids":      chosen_input_ids,
            "chosen_attention_mask": chosen_attention_mask,
            "chosen_labels":         chosen_labels,
            "rejected_input_ids":      rej_input_ids,
            "rejected_attention_mask": rej_attention_mask,
            "rejected_labels":         rej_labels,

            # raw fields for downstream metrics/CSV
            "premise":          row["premise"],
            "initial":          row["initial"],
            "counterfactual":   row["counterfactual"],
            "original_ending":  row["original_ending"],
            "edited_ending":    row["edited_ending"],
        }

def create_dataloaders(data_path, tokenizer, batch_size, num_workers):
    dataloaders = {}
    for split in ["train", "dev", "test"]:
        file_name = CONFIG[f"{split}_file"]
        dataset = DPODataset(Path(data_path) / file_name, tokenizer)
        dataloaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=(split == "train")
        )
    return dataloaders
