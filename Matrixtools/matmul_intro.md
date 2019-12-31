

MATRIX MULTIPLICATION:
=====================

Matrix multiplication is core to our operations in DL, ML and AI in general.

It is also, comparably, an expensive operation. And, investigations into optimization has been ongoing.

A more optimized approach, called `Strassen algorithm`, attains a time complexity of O(n^2.8...). However, it is not as useful for small matrices and some other types of matrices.

The fastest algorithm, `Coppersmith–Winograd algorithm`, attains O(n^2.3...). In recent years improvements have resulted in additional small gains in speed. However, it is rarely useful in industry, as it is thought to be practical for very large matrices, which may not be reslistic for our systems now.

The standard operation, which we demonstrated, is more widely used, as it fits more used cases than the optimized options. Even though, we get a complexity of O(n^3)


Key thing to note
-----------------

Optimizing matrix multiplication requires knowledge of when to use which algorithms. The most practical algorithms now are the standard or naive algorithm, and the Strassen algorithm. For smaller matrices, the standard algorithm outperforms the Strassen. Then, as the size of matrices grow, the Strassen shines - for given types of matrices.


THE STANDARD ALGORITHM
======================

Two matrices can only be multiplied if the number of columns of the first matrix equals the number of rows of the second matrix.

For, a matrix represented as an n-dimesional array, the number of arrays in the parent array is the number of rows of the matrix. Then, the number of items in each sub-array is the number of columns in the matrix.

We are only concerned with the naive implementation for our use case. This can be made more elegant. But, for the sake of time we wrap it up with this.

```
def mm(X, Y):

	# We get the number of rows and columns of the matrices.

	# the number of columns of matrices
	X_cols = len(X[0])
	Y_cols = len(Y[0])

	# the number of rows of both matrices
	X_rows = len(X)
	Y_rows = len(Y)

	# Number of columns of X, should be same as the number of rows of Y, to be able to perform the operation.
	if(X_cols == Y_rows):
		new_matrix = [] 

		for i in range(X_rows):
			new_row = []
			for j in range(Y_cols):
				new_val = 0
				for k in range(Y_rows):
					new_val += (X[i][k] * Y[k][j])

				new_row.append(new_val)

			new_matrix.append(new_row)

		return new_matrix
	else:
		return "Arrays can NOT be multiplied"

```

We test this out for the following matrices:
```
P = [[1, 2, 1], [2, 1, 2]]
Q = [[1, 1], [2, 2], [3, 3]]
R = [[3, 3], [1, 1]]
```

OUTCOMES:
```
>>> mm(P, Q)
[[8, 8], [10, 10]]
>>> 
>>> mm(R, P)
[[9, 9, 9], [3, 3, 3]]
>>> 
>>> mm(P, R)
Arrays can NOT be multiplied
```


