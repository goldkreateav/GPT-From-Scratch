import tiktoken
from GPTDataset import GPTDatasetV1
from torch.utils.data import Dataset, DataLoader
import torch
from SelfAttention import SelfAttention_v1, SelfAttention_v2, CasualAttention, MultiHeadAttensionWrapper
def create_dataloader_v1(txt, batch_size=4, max_length=256, stride=128, 
                          shuffle=True, drop_last=True, num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2")  
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride) 
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, 
                            drop_last=drop_last, num_workers=num_workers)
    
    return dataloader 
f = open("the-verdict.txt", "r")
text = f.read()

data_loader = create_dataloader_v1(text, max_length=4, stride=1, shuffle=False)
data_iter = iter(data_loader)
first_batch = next(data_iter)


inputs = torch.tensor([
    [0.43, 0.15, 0.89],  # Your (x^1)
    [0.55, 0.87, 0.66],  # journey (x^2)
    [0.57, 0.85, 0.64],  # starts (x^3)
    [0.22, 0.58, 0.33],  # with (x^4)
    [0.77, 0.25, 0.10],  # one (x^5)
    [0.05, 0.80, 0.55]   # step (x^6)
])

d_in = inputs.shape[-1]
d_out = 1
torch.manual_seed(123)
batch = torch.stack((inputs, inputs), dim=0)
context_length = batch.shape[1]
print(d_in, d_out, context_length)
multihead = MultiHeadAttensionWrapper(d_in, d_out, context_length, 0.0, num_head=2)
print(multihead(batch))