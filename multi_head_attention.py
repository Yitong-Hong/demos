import torch
import math

torch.manual_seed(42)
input_dim=5
input_len=4
n_dim=6
n_heads=3
per_dim=n_dim//n_heads
I=torch.rand(input_len,input_dim)
W_Q=torch.rand(input_dim,n_dim)
W_K=torch.rand(input_dim,n_dim)
W_V=torch.rand(input_dim,n_dim)
Q=I@W_Q
K=I@W_K
V=I@W_V
Q=Q.reshape(input_len,n_heads,per_dim)
K=K.reshape(input_len,n_heads,per_dim)
V=V.reshape(input_len,n_heads,per_dim)
Q=Q.transpose(0,1)
K=K.transpose(0,1)
V=V.transpose(0,1)
attention=Q@K.transpose(-2,-1)/math.sqrt(per_dim)
attention=torch.softmax(attention,dim=-1)
# V: (n_heads,input_len,per_dim)
# attention: (n_heads,input_len,input_len)
attention=attention@V
# attention: (n_heads,input_len,per_dim)
attention=attention.transpose(0,1).reshape(input_len,n_dim)
print(attention)


