import tiktoken
from GPTDataset import GPTDatasetV1
from torch.utils.data import Dataset, DataLoader
import torch
from SelfAttention import SelfAttention_v1, SelfAttention_v2, CasualAttention, MultiHeadAttentionWrapper, MultiHeadAttention
from GPTModel import GPTModel
from config import GPT_CONFIG_124M 
def create_dataloader_v1(txt, batch_size=4, max_length=256, stride=128, 
                          shuffle=True, drop_last=True, num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2")  
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride) 
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, 
                            drop_last=drop_last, num_workers=num_workers)
    
    return dataloader 
def generate_text_simple(model, idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        idx_cont = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cont)
        logits = logits[:, -1, :]
        probas = torch.softmax(logits, dim=-1)
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=-1)
    return idx
f = open("the-verdict.txt", "r")
text = f.read()

data_loader = create_dataloader_v1(text, max_length=4, stride=1, shuffle=False)


batch = []
text_1 = "Every effort moves you"
text_2 = "Every day holds a"
batch.append(torch.tensor(data_loader.dataset.tokenizer.encode(text_1)))
batch.append(torch.tensor(data_loader.dataset.tokenizer.encode(text_2)))
batch = torch.stack(batch, dim=0)
model = GPTModel(GPT_CONFIG_124M)
print([data_loader.dataset.tokenizer.decode(x.flatten().tolist()) for x in list(generate_text_simple(model, batch, 10, 4))])