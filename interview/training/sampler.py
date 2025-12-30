"""
Implement a sampler:
index_mapping={0: "tag1", 1: "tag1", 2: "tag2", ...}
tags_mapping={"tag1": 1, "tag2": 2, "tag3": 4..}  mapping from tag to weight, weith >= 1
"""
import random

class Sampler:
    def __init__(self, index_mapping, tags_mapping, size):
        random.seed(42)
        #assert size > 0
        if size <= 0:
            raise ValueError("Sampler requires size > 0")
        
        self.size = size
        self.index_mapping = index_mapping
        self.tags_mapping = tags_mapping
        self.weights = []
        self.indicies = []
        for idx, tag in self.index_mapping.items():
            if tag in self.tags_mapping:
                self.weights.append(self.tags_mapping[tag])
                self.indicies.append(idx)
        
    def __iter__(self) -> int:
        count = 0
        while count < self.size:
            yield random.choices(self.indicies, weights=self.weights)[0]
            count += 1

# unit tests

index_mapping={0: "tag1", 1: "tag1", 2: "tag2"}
tags_mapping={"tag1": 1, "tag2": 2, "tag3": 4}

sampler1 = Sampler(index_mapping, tags_mapping, 5)

# test sample return valid indicies
samples = list(sampler1)
for sample in samples:
    assert sample in [0, 1, 2]
    
# sample size limit
sampler2 = Sampler(index_mapping, tags_mapping, size=5)
samples = list(sampler2)
assert len(samples) == 5
