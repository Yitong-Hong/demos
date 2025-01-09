import torch
import math
torch.manual_seed(42)
input_dim=3
input_len=4
n_dim=2
I=torch.rand(input_len,input_dim)
W_Q=torch.rand(input_dim,n_dim)
W_K=torch.rand(input_dim,n_dim)
W_V=torch.rand(input_dim,n_dim)
Q=I@W_Q
K=I@W_K
V=I@W_V
attention=Q@K.T/math.sqrt(n_dim)
attention=torch.softmax(attention,dim=1)
print(attention@V)
