''' # 1. DeviceMesh
mesh = DeviceMesh(dim_names=("pp", "tp"), dim_shapes=(2, 2))

dim_shapes = tuple of len(2)
total number of elements = dim_shapes[0] * dim_shapes[1]

-> print(mesh.mesh) -> [[0,1], [2,3]]
0 1
2 3

mesh = DeviceMesh(dim_names=("pp", "tp"), dim_shapes=(2, 4))
-> print(mesh.mesh) -> [[0,1, 2, 3], [4,5,6,7]]

0 1 2 3
4 5 6 7

# m x n pp -> m, tp -> n,

[0, 1]
[2, 3]

# 2 can you also add # 2. Shape
    mesh.shape() = (2, 2)
    mesh.shape("tp") = (2, )
    mesh.shape("pp") = (2, )

# 3. Get the group corresponding to a rank for a given dim
    mesh.get("tp", rank=0) -> [0, 1] GPUs in the group
    mesh.get("tp", rank=1) -> [0, 1]

    mesh.get("tp", rank=2) -> [2, 3]
    mesh.get("tp", rank=3) -> [2, 3]

    mesh.get("pp", rank=0) -> [0, 2]
    mesh.get("pp", rank=1) -> [1, 3] '''

class DeviceMesh:
    def __init__(self, dim_names, dim_shapes):
        self.dim_names = dim_names
        self.dim_shapes = dim_shapes
        self.total_ranks = 1
        self.mesh = self._create_mesh()
        self.coords = self._create_coords()

    def _create_mesh(self):
        for shape in self.dim_shapes:
            self.total_ranks *= shape
        
        elements = []
        for i in range(self.total_ranks):
            elements.append(i)

        return self._reshape(elements, self.dim_shapes)

    # recursive - think of sorting algorithms
    def _reshape(self, arr, shape):
        if len(shape) == 1:
            return arr
        
        chunk_size = len(arr) // shape[0]
        result = []
        for i in range(shape[0]):
            start_chunk = i * chunk_size
            end_chunk = start_chunk + chunk_size
            chunk = arr[start_chunk:end_chunk]
            result.append(self._reshape(chunk, shape[1:]))
            
        return result
    
    def _create_coords(self):
        coords = []
        for r in range(self.total_ranks):
            c = []
            for s in self.dim_shapes:
                c.insert(0, r % self.dim_shapes[s])
                r //= self.dim_shapes[s]
            coords.append(c)
        return coords

    # question 2
    def shape(self, dim_name=None):
        if not dim_name:
            return self.dim_shapes
        
        i = self.dim_names.index(dim_name)
        return (self.dim_shapes[i],)
    
    # rank is just the device/GPU number, 
    # and get() finds all devices that share 
    # the same "slice" along the specified dimension.
    def get(self, dim_name, rank):
        dim_name_i = self.dim_names.index(dim_name)
        coords = []
        t = rank

        # step 1 build out coordinates
        # we need the coordinate system because 
        # you need to know which position along 
        # each dimension a rank occupies to find its group
        for s in self.dim_shapes:
            coords.insert(0, t % self.dim_shapes[s])
            t //= self.dim_shapes[s]
        
        target_coord = coords[dim_name_i]

        result = []

        for r in range(self.total_ranks):
            # Check if this rank has the same coordinate for target dimension
            if self.coords[r][dim_name_i] == target_coord:
                result.append(r)

        return result

mesh = DeviceMesh(dim_names=('tp', 'dp', 'pp'), dim_shapes=(2, 2, 2))
print(mesh.mesh)

print(mesh.shape("tp"))
print(mesh.get('pp', rank=1))
print(mesh.get('tp', rank=1))
print(mesh.get('dp', rank=1))