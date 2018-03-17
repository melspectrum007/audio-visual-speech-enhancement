
import numpy as np

def split_and_concat(array, axis, split):
	slc = [slice(None)] * array.ndim
	slc[axis] = slice(0, -(array.shape[axis] % split))

	return np.concatenate(np.split(array[slc], split, axis))


a = np.zeros([3, 80, 248])

b = split_and_concat(a, -1, 4)

print b.shape