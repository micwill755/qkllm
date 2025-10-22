import torch

tensor1 = torch.tensor([1., 2., 3.])
tensor2 = torch.tensor([1., 2., 3.])
print(tensor1 + tensor2)

tensor1 = torch.tensor([1., 2., 3.]).to("cuda")
tensor2 = torch.tensor([1., 2., 3.]).to("cuda")
print(tensor1 + tensor2)

tensor1 = torch.tensor([1., 2., 3.]).to("cuda:1")
tensor2 = torch.tensor([1., 2., 3.]).to("cuda:1")
print(tensor1 + tensor2)