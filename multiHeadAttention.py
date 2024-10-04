import torch.nn as nn
import torch
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads,dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.head_dim = d_model // num_heads

        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=dropout)

        self.softmax = nn.Softmax(dim=-1)

        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self,X,mask):
        batch_size,seq_len,_ = X.shape

        q,k,v = self.query(X),self.key(X),self.value(X)
        # 此时得到的qkv是(batch_size,seq_len,d_model)
        # 要变成h个单头注意力，故要将他们变成
        # (batch_size,num_heads,seq_len,head_dim)

        q = q.view(batch_size,seq_len,self.num_heads,self.head_dim).transpose(1,2)
        k = k.view(batch_size,seq_len,self.num_heads,self.head_dim).transpose(1,2)
        v = v.view(batch_size,seq_len,self.num_heads,self.head_dim).transpose(1,2)

        ## 然后进进行scaled Dot-product Attention
        attention_weight = q@k.transpose(-1,-2)/math.sqrt(self.head_dim)
        if mask is not None:
            attention_weight = attention_weight.masked_fill(mask==0,float('-inf'))
        attention_weight = self.softmax(attention_weight)
        attention_weight = self.dropout(attention_weight)
        attention_score = attention_weight@v
        ##接下来要将几个单头concat起来
        attention_score = attention_score.transpose(1,2).contiguous()
        attention_score = attention_score.view(batch_size,seq_len,-1)
        #print(attention_score.shape)
        output = self.out_proj(attention_score)
        return output

if __name__ == '__main__':
    X = torch.rand(3,2,128)
    # 多头的attention_mask应该是(batch_size,num_heads,seq_len,seq_len)
    # (3,8,2,2)
    attention_mask = torch.tensor([
        [0,1],
        [1,0],
        [1,1]
    ]).unsqueeze(1).unsqueeze(2).repeat(1,8,2,1)
    print(attention_mask.shape)
    net = MultiHeadAttention(128,8)
    y = net(X,attention_mask)
    print(y.shape)