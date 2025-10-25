#!/bin/python3

import math
import os
import random
import re
import sys

#
# Complete the 'findMinBatchSize' function below.
#
# The function is expected to return an INTEGER.
# The function accepts following parameters:
#  1. INTEGER_ARRAY dataSamples
#  2. INTEGER maxBatches
#

# [1, 5, 7]
# batch size = 1 
# dataSamples[0] = 1 / batchsize = 1
# dataSamples[1] = 5 / batchsize = 5
# batch size = 2
# dataSamples[0] = 1 / batchsize = 1
# dataSamples[1] = 5 / batchsize = 3 batches = [2, 2, 1]
# batch size = 3
# dataSamples[0] = 1 / batchsize = 1
# dataSamples[1] = 5 / batchsize = 2 batches = [3, 2]

# [2, 4, 5]
# b size = 1
# samples[0] = 2 / bsize = 2
# samples[1] = 4 / b size = 4
# samples[2] = 5 / b size = 5
# 11 > 10
# [2, 4, 5]
# b size = 2
# samples[0] = 2 / bsize = 1
# samples[1] = 4 / b size = 2
# samples[2] = 5 / b size = 3
# 8 < 10
# [2, 4, 5, 8, 9], m = 5
# b size = 2
# samples[0] = 2 / bsize = 1
# samples[1] = 4 / b size = 2
# samples[2] = 5 / b size = 3
# 1, 2, 3, 4, 5 = 15
# 8 < 10

'''
m = 8
1 + 5 + 7 = 13 - b size of 1
ceil(1) = 1 + ceil(5 / 2) = 3 + ceil(7 / 2) == 4 = 13 - b size of 2

[2, 5, 8, 9, 13, 18, 24]
[2, 5, 8, 9]

t = 5
mid = 3

[1, 2, 3, 4, ]
bs = 2
res = 46
max_b = 48

for i

'''

def findMinBatchSize(dataSamples: list[int], maxBatches: int):
    left, right = 1, max(dataSamples)
    iterations = 0
    while left < right:
        mid = (left + right) // 2
        total = sum(math.ceil(sample / mid) for sample in dataSamples)
        if total <= maxBatches:
            right = mid 
        else:
            left = mid + 1
        iterations += 1
    
    return left

# binary search but memory is large 
# how can we avoid creating this for i in range(1, max(dataSamples)): b_sizes.append(i)
'''def findMinBatchSize(dataSamples: list[int], maxBatches: int):
    b_sizes = []
    for i in range(1, max(dataSamples)):
        b_sizes.append(i)
        
    left, right = 0, len(b_sizes)
    
    min_b_size = float('inf')
    iterations = 0

    while left < right:
        mid = (right + left) // 2
        s = 0
        b_size = b_sizes[mid]

        for i in range(len(dataSamples)):
            s += math.ceil(dataSamples[i] / b_size)

        if s <= maxBatches:
            right = mid
        else:
            left = mid + 1
        
        min_b_size = min(min_b_size, b_size)
        iterations += 1
    
    print(iterations)
    return min_b_size'''

# brute force
'''def findMinBatchSize(dataSamples: list[int], maxBatches: int):
    while result > maxBatches:
        s = 0
        for i in range(len(dataSamples)):
            s += math.ceil(dataSamples[i] / b_size)
        result = s
        b_size += 1
            
    return b_size - 1'''
        
if __name__ == '__main__':
    # For testing with large_test_input.txt
    with open('large_test_input.txt', 'r') as f:
        dataSamples_count = int(f.readline().strip())
        dataSamples = []
        
        for _ in range(dataSamples_count):
            dataSamples_item = int(f.readline().strip())
            dataSamples.append(dataSamples_item)
        
        maxBatches = int(f.readline().strip())
    
    result = findMinBatchSize(dataSamples, maxBatches)
    print(f"Result: {result}")