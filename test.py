import numpy as np
l = [(1, 2), (1, 2), (1, 3),]
l.reverse()
print(l)

from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
x = np.array([[1,1], [2,2], [3,3], [4,4], [5,5]])
y = np.array([[2,2], [3,3], [4,4]])
distance, path = fastdtw(x, y, dist=euclidean)
print(distance)
print(path)


word = "lines."
test_list = list()
test_list.extend(word)
print(test_list)
grapheme_set = {"a"}
grapheme_set = grapheme_set.union(word)
print(grapheme_set)
grapheme_dict = dict()
print(grapheme_dict)

print(1//5)
print(1/5)

p = [1, 2, 3]
print(len(p))
p.copy().insert(0, 100)
print(p)

from itertools import combinations, permutations, combinations_with_replacement
combines = [c for c in  combinations(range(4), 2)]
combines_with_replace = [c for c in  combinations_with_replacement(range(4), 2)]
permutes = [p for p in  permutations(range(4), 2)]
print(combines)
print(combines_with_replace)
print(permutes)

for locations in combines_with_replace:
    print("locations:"+str(locations))
    for i in range(len(locations)):
        print(locations[i] + i)
        pass
    pass
print(combines_with_replace)

func = lambda x: x[1]
print("###############")
print(str(func((0, 1))))



import numpy as np
array = np.zeros(shape=(5, 7))
new_array = np.ones(shape=(5, 7))

import time
begin_time = time.time()
print(np.mean(array-new_array))
end_time = time.time()
print(end_time - begin_time)

begin_time = time.time()
print(np.mean(np.subtract(array, new_array)))
end_time = time.time()
print(end_time - begin_time)