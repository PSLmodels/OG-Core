# function
def math_L (L):
    return min(L), max(L), sum(L)/len(L)

#Calling the function

L = [3,6,5]
min_L, max_L, mn_L = math_L(L)

print('The min, max and mean of L =' , math_L(L))
