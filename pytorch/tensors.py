import torch

tensor0d = torch.tensor(1) # scalar e.g: 2
tensor1d = torch.tensor([0, 1, 2]) # vector e.g: [0, 1, 2]
tensor2d = torch.tensor([[0, 1], 
                         [3, 5]])
tensor3d = torch.tensor([[[0, 1], [2, 3]],
                          [[4, 5], [6, 7]]])

print(tensor3d.dtype) # default 64-bit integer

floatvec = torch.tensor([1.0, 2.0, 3.0]) # default 32-bit float
print(floatvec.dtype)

float3d = tensor3d.to(dtype=floatvec.dtype)
print(float3d.dtype)

# .view()

# Contiguous tensor
x = torch.randn(3, 4)
print(x.is_contiguous())  # True
x.view(12) 
print(x.view(12))

# Non-contiguous after transpose
y = x.transpose(0, 1)
print(y)
print(y.is_contiguous())  # False
#y.view(12)  # will fail

'''Original x (contiguous):

Memory: [1][2][3][4][5][6]
         └─row1─┘ └─row2─┘  ← sequential

Copy
Transposed y (non-contiguous):

Memory: [1][2][3][4][5][6]
         ↑  ↑  ↑  ↑  ↑  ↑
         │  │  │  │  │  └─ row3, col2
         │  │  │  │  └──── row2, col2  
         │  │  │  └─────── row1, col2
         │  │  └────────── row3, col1
         │  └───────────── row2, col1
         └──────────────── row1, col1
         
.view() just changes the "reading instructions" (the shape/stride metadata), not the actual data in memory.

'''

#.reshape() is smarter - it tries to avoid copying, but will copy if necessary.
# Case 1: Contiguous tensor - NO COPY (acts like .view())
x = torch.tensor([[1, 2, 3], [4, 5, 6]])
y = x.reshape(6)

print(x.data_ptr() == y.data_ptr())  # True - same memory!
y[0] = 999
print(x)  # tensor([[999, 2, 3], [4, 5, 6]]) - x changed too!

# Case 2: Non-contiguous tensor - COPIES DATA
x = torch.tensor([[1, 2, 3], [4, 5, 6]])
z = x.transpose(0, 1)  # Non-contiguous
w = z.reshape(6)

print(z.data_ptr() == w.data_ptr())  # False - different memory!
w[0] = 999
print(z)  # tensor([[1, 4], [2, 5], [3, 6]]) - z unchanged!

