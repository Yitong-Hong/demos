import torch
import math
torch.manual_seed(42)
input_len=4
n_dim=2
I=torch.rand(input_len,n_dim)
W_Q=torch.rand(n_dim,n_dim)
W_K=torch.rand(n_dim,n_dim)
W_V=torch.rand(n_dim,n_dim)
Q=I@W_Q
K=I@W_K
V=I@W_V
attention=Q@K.T/math.sqrt(n_dim)
attention=torch.softmax(attention,dim=1)
print(attention@V)
