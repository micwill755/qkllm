import random
from itertools import product

class Tensor:
    def __init__(self, data, v=0, use_rand=False):
        if use_rand:
            self._shape = data
            self.tensor = self._create_rand_tensor(data, v)
        elif isinstance(data, tuple):  # Shape tuple
            self._shape = data
            self.tensor = self._create_tensor(data, v)
        else:  # Array data
            self.tensor = data
            self._shape = self._create_shape(data)
        
    def __getitem__(self, index):
        # TODO avoid copying
        if isinstance(index, tuple):
            result = self.tensor
            for idx in index:
                result = result[idx]
            return result if not isinstance(result, list) else Tensor(result)
        
        if isinstance(self.tensor[index], list):
            return Tensor(self.tensor[index])
        return self.tensor[index]

    def __setitem__(self, index, value):
        if isinstance(index, tuple):
            result = self.tensor
            for idx in index[:-1]:
                result = result[idx]
            result[index[-1]] = value
        else:
            self.tensor[index] = value

    def __truediv__(self, scalar):
        result = Tensor(self.shape)
        batch_indices = product(*[range(d) for d in self.shape])
        for idx in batch_indices:
            result[idx] = self[idx] / scalar
        return result

    def __len__(self):
        return len(self.tensor)
    
    def __iter__(self):
        return iter(self.tensor)

    def _print(self, s, t):
        if len(s) == 1:
            return f'{t}'
        str = ''
        for i in range(s[0]):
            if i == 0:
                str += f'{self._print(s[1:], t[i])}\n'
            elif i > 0 and i < s[0] - 1:
                str += f'   {self._print(s[1:], t[i])}\n'
            elif i == s[0] - 1:
                str += f'   {self._print(s[1:], t[i])}'

        return f'Tensor({str})'
    
    def __str__(self):
        return self._print(self.shape, self.tensor)
    
    def __mul__(self, other):
        result = Tensor(self.shape)
        for idx in product(*[range(d) for d in self.shape]):
            result[idx] = self[idx] * other
        return result

    def __sub__(self, other):
        result = Tensor(self.shape)
        for idx in product(*[range(d) for d in self.shape]):
            result[idx] = self[idx] - other[idx]
        return result

    def __add__(self, other):
        result = Tensor(self.shape)
        for idx in product(*[range(d) for d in self.shape]):
            result[idx] = self[idx] + other[idx]
        return result
    
    def _create_shape(self, arr):
        if not isinstance(arr, list):
            return ()
        
        shape = (len(arr),)
        if len(arr) > 0:
            shape += self._create_shape(arr[0])
        return shape
    
    def update_shape(self, s):
        self._shape = s
    
    def _create_rand_tensor(self, s, v=0):
        if len(s) == 1:
            return [round(random.gauss(0, 1), 4) for i in range(s[0])]
        
        tensor = []
        for _ in range(s[0]):
            tensor.append(self._create_rand_tensor(s[1:], v))
        return tensor
    
    def _create_tensor(self, s, v=0):
        if len(s) == 1:
            return [v for i in range(s[0])]
        
        tensor = []
        for _ in range(s[0]):
            tensor.append(self._create_tensor(s[1:], v))
        return tensor
        
    def append(self, item):
        self.tensor.append(item)
        self._shape = self._create_shape(self.tensor)
    
    def get_value_at(self, tensor, indices):
        for idx in indices:
            tensor = tensor[idx]
        return tensor

    def set_value_at(self, tensor, indices, value):
        for idx in indices[:-1]:
            tensor = tensor[idx]
        tensor[indices[-1]] = value
    
    # recursive approach
    def iterate(self, d1, d2, newTensor, current_indices):
        # Base case: we have a complete index pointing to an element
        if len(current_indices) == len(self._shape):
            # Swap d1 and d2 positions in the index
            new_indices = list(current_indices)
            new_indices[d1], new_indices[d2] = new_indices[d2], new_indices[d1]
            
            # Copy element from old position to new position
            value = self.get_value_at(self.tensor, current_indices)
            self.set_value_at(newTensor.tensor, new_indices, value)
            return
        
        # Recursive case: add next dimension index
        current_dim = len(current_indices)
        for i in range(self.shape[current_dim]):
            self.iterate(d1, d2, newTensor, current_indices + [i])

    def _transpose(self, d1=-1, d2=-1):
        # TODO first implementation creating new Tensor, need to improve
        # convert shape
        shape_list = list(self._shape)
        shape_list[d1], shape_list[d2] = shape_list[d2], shape_list[d1]
        new_shape = tuple(shape_list)
        # new shape will be created when instantiated
        newTensor = Tensor(new_shape)
        
        # using recursion
        # self.iterate(d1, d2, newTensor, [])
        # using iter tools
        for indices in product(*[range(d) for d in self.shape]):
            new_indices = list(indices)
            new_indices[d1], new_indices[d2] = new_indices[d2], new_indices[d1]
            value = self.get_value_at(self.tensor, indices)
            self.set_value_at(newTensor.tensor, new_indices, value)

        return newTensor
            
    def transpose(self, d1=-1, d2=-1):
        return self._transpose(d1, d2)
    
    def squeeze(self, dim=0):
        # TODO: make this dynamic
        if dim == 0 and len(self.shape) > 0 and self.shape[0] == 1:
            return Tensor(self.tensor[0])
    
    def unsqueeze(self, dim=0):
        # TODO: make this dynamic
        if dim == 0:
            return Tensor([self.tensor])

    @property
    def shape(self):
        return self._shape
    
    @property
    def T(self):
        return self._transpose(-2, -1)