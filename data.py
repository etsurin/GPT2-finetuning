import torch
import numpy as np
import pandas as pd

class ConversationDataset(Dataset):
    def __init__(self, tokenizer, raw_data, max_length):

        self.data = []
        self.labels = []
        self.gensrc = []
        self.gentgt = []
        for item in raw_data:
            tmp_list = item.split('<|endoftext|>')
            tmp_data = tokenizer.encode(item + '<|endoftext|>', max_length = max_length, padding = 'max_length')
            tmp_tgt = tmp_list[-1]
            tmp_src = tokenizer.encode('<|endoftext|>'.join(tmp_list[:-1]), max_length = max_length, padding = 'max_length')
            self.data.append(tmp_data)
            self.labels.append([-100 if x == tokenizer.pad_token_id else x for x in tmp_data])
            self.gentgt.append(tmp_tgt)
            self.gensrc.append(tmp_src)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return torch.tensor(self.data[item], dtype=torch.long), torch.tensor(self.labels[item], dtype=torch.long), torch.tensor(self.gensrc[item], dtype=torch.long), self.gentgt[item]

def setup_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)

setup_seed(123)

all_rick = pd.read_csv('RickAndMortyScripts.csv')
rawtext = all_rick['line']
n = 8
all_data = []
for i in range(len(rawtext)-n):
    all_data.append('<|endoftext|>'.join(rawtext[i:i+n]))