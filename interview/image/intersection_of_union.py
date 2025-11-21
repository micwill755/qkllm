'''

IoU = Intersection over Union means:

Intersection: The area where boxes overlap (4 in your example)

Union: The total area covered by both boxes combined (28 in your example)

IoU: The ratio = intersection / union = 4/28 ≈ 0.143

So IoU is not the overlapping area itself, but rather how much of the total combined area is overlapping.

'''

def iou_2d(box1, box2):
    """Calculate IoU for 2D boxes [x1, y1, x2, y2]"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union

# Test with overlapping boxes
box1 = [0, 0, 4, 4]  # 4x4 box from (0,0) to (4,4)
box2 = [2, 2, 6, 6]  # 4x4 box from (2,2) to (6,6)

result = iou_2d(box1, box2)
print(f"Box1: {box1}")
print(f"Box2: {box2}")
print(f"IoU: {result}")

# Manual calculation:
# Intersection: (4-2) * (4-2) = 2*2 = 4
# Area1: 4*4 = 16
# Area2: 4*4 = 16  
# Union: 16 + 16 - 4 = 28
# IoU: 4/28 = 0.143
print(f"Expected: {4/28:.3f}")
