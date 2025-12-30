'''Description:
Determine if a bot can reach a specified destination point on a grid.

The bot starts at coordinates (x, y) and must reach a target location. It can make unlimited moves but is restricted to only 
two types of movements:

1. Move from (x, y) to (x + y, y)
2. Move from (x, y) to (x, x + y)

For example, starting at (1, 1), a possible sequence is: (1, 1) → (1, 2) → (3, 2) → (5, 2).

Function signature:
python
def canReach(x1, y1, x2, y2):
    # Complete the 'canReach' function below.
    # The function is expected to return a STRING.
    # The function accepts following parameters:
    # 1. INTEGER x1
    # 2. INTEGER y1  
    # 3. INTEGER x2
    # 4. INTEGER y2
    # Write your code here


Input:
The function receives four integers representing the starting coordinates (x1, y1) and target coordinates (x2, y2).

Output:
Return "Yes" if the bot can reach the target, "No" otherwise.'''

# time complexity 2^n
def canReach(x1, y1, x2, y2):
    if x1 == x2 and y1 == y2:
        return True

    if x1 > x2 or y1 > y2:
        return False
    
    return (canReach(x1 + y1, y1, x2, y2) 
        or canReach(x1, y1 + x1, x2, y2))

# time complexity 
def canReachReverse(x1, y1, x2, y2):
    while x2 >= x1 and y2 >= y1:
        # here we enforce only 1 move a time
        # the order in which we step back from the destination
        # doesn't matter if the path exists
        if x1 == x2 and y1 == y2:
            return True
        if x2 > x1:
            x2 -= y1
        elif y2 > y1:
            y2 -= x1 
    
    return False

if __name__ == '__main__':
    print(canReachReverse(1, 1, 3, 5))
    print(canReachReverse(1, 1, 2, 3))
    print(canReachReverse(3, 4, 3, 4))
    print(canReachReverse(5, 7, 2, 3))