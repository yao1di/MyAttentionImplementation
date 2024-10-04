import math
import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    def __init__(self, hidden_dim, dropOut=1e-1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.Query = nn.Linear(hidden_dim, hidden_dim)
        self.Key = nn.Linear(hidden_dim, hidden_dim)
        self.Value = nn.Linear(hidden_dim, hidden_dim)
        self.Dropout = nn.Dropout(p=dropOut)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, X, mask=None):
        ## 首先得到q,k,v
        q, k, v = self.Query(X), self.Key(X), self.Value(X)
        # q@k^T => (batch_size,seq_len,hiddendim) *(batch_size,hiddendim,seq_len)
        # (batch_size,seq_len,seq)  
        attention_weight = q @ k.transpose(-1, -2) / math.sqrt(self.hidden_dim)

        if mask is not None:
            attention_weight = attention_weight.masked_fill(mask==0,float("-inf"))

        attention_weight = self.softmax(attention_weight)


        attention_weight = self.Dropout(attention_weight)
        print(attention_weight)
        return attention_weight@v
if __name__ == '__main__':
    X = torch.randn(3,4,5)
    model = SelfAttention(5)
    mask = torch.tensor(
        [
            [1,1,1,0],
            [1,1,0,0],
            [1,0,0,0],
        ]
    )
    ## mask应该是(batch_size,seq_len,seq_len)的shape
    mask = mask.unsqueeze(dim=1).repeat(1,4,1)
    y = model(X, mask)
    print(y.shape)
