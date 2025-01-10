import torch
import math

torch.manual_seed(42)

input_len=4
n_dim=6
n_heads=3
per_dim=n_dim//n_heads
I=torch.rand(input_len,n_dim)
W_Q=torch.rand(n_dim,n_dim)
W_K=torch.rand(n_dim,n_dim)
W_V=torch.rand(n_dim,n_dim)
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
attention=attention@V
attention=attention.transpose(0,1).reshape(input_len,n_dim)
FC=torch.rand(n_dim,n_dim)
attention=attention@FC
print(attention)


