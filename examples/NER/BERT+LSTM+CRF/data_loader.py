import torch
import numpy as np

from torch.utils.data import Dataset

class NerDataset(Dataset):
    def __init__(self, data, args, tokenizer):
        self.data = data
        self.args = args
        self.tokenizer = tokenizer
        self.label2id = args.label2id
        self.max_seq_len = args.max_seq_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        examples=self.data[item]
        tokenized_exmaples = self.tokenizer(examples["tokens"], max_length=self.max_seq_len,truncation=True,padding="max_length",
                                       is_split_into_words=True)
        word_ids = tokenized_exmaples.word_ids(batch_index=0)
        label_ids = []
        for word_id in word_ids:
            if word_id is None:
                label_ids.append(0)
            else:
                label_ids.append(examples["label"][word_id])
        tokenized_exmaples["labels"] = label_ids

        for k,v in tokenized_exmaples.items():
            tokenized_exmaples[k]=torch.tensor(np.array(v))

        return tokenized_exmaples
