import random

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
        if isinstance(self.tensor[index], list):
            return Tensor(self.tensor[index])
        return self.tensor[index]
    
    def __setitem__(self, index, value):
        self.tensor[index] = value
    
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
    
    def _create_shape(self, arr):
        if not isinstance(arr, list):
            return ()
        
        shape = (len(arr),)
        if len(arr) > 0:
            shape += self._create_shape(arr[0])
        return shape
    
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
    
    def _transpose(self, d1=-1, d2=-1):
        # we arent providing the dimensions to tranpose
        if d1 == -1:
            if len(self.shape) == 1:
                return Tensor(self.tensor)
            if len(self.shape) == 2: 
                t = Tensor(self.shape[::-1])
                for i in range(self.shape[0]):
                    for j in range(self.shape[1]):
                        t[j][i] = self.tensor[i][j]
                return t
    
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
        return self._transpose()