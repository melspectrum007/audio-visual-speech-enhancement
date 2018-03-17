
import numpy as np

def split_and_concat(array, axis, split):
	slc = [slice(None)] * array.ndim
	mod = -(array.shape[axis] % split)
	end = None if mod == 0 else mod
	slc[axis] = slice(0, end)

	return np.concatenate(np.split(array[slc], split, axis))


a = np.zeros([3, 80, 248])

b = split_and_concat(a, -1, 6)

print b.shape