"""

(c) Copyright, Victor Mawusi Ayi. 2019
All Rights Reserved!

"""


from vectorkit import Vector, isovector

class Matrix():

	def __init__(self, array_of_arrays):

		self.column_size = 0
		self.longest_value = 0
		self.row_size = 0
		self.rows = []

		try:
			for array in array_of_arrays:
				arr_to_vec = Vector(array)
				self.rows.append(arr_to_vec)
				
				self.row_size += 1
				if arr_to_vec.dimensions > self.column_size:
					self.column_size = arr_to_vec.dimensions

		except ValueError:
			raise ValueError(
				"Matrices must contain arrays of numbers"
			)
			
		self.__smoothen__()
		self.shape = (self.row_size, self.column_size)

	def __repr__(self):

		return "Matrix([{}])".format(
            			"\n".ljust(9).join([
               				" ".join([
                    				str(round(num, 2)).ljust(self.longest_value) for num in vector
                			]).rjust(10, " ") for vector in self.rows
            			])
        	)

	def __smoothen__(self):
		long_num_len = 0
		for index in range(self.row_size):
			self.rows[index].extend(self.column_size)
			
			for component in self.rows[index]:
				component_len = len(str(round(component,2)))
				if component_len>long_num_len:
					long_num_len=component_len
				
		self.longest_value=long_num_len

	def matmul(self, other):
		if isinstance(other, Matrix):
			if self.shape[0]==other.shape[1]:
				
				result = []
				
				other_prep = Matrix([
					list(tuple) for tuple in zip(
						*[vector.components for vector in other.rows]
					)
				])
				
				for x in self.rows:
					result.append(
						[x*y for y in other_prep.rows]
					)
			else:
				raise ValueError(
					"The matrices do not have the right shapes to be multiplied"
				)
		elif isinstance(other, Vector):
			if other.dimensions==self.shape[0]:
				result = [
					[
						other.dotmul(Vector(b)) for b in zip(
							*self.rows
						)
					]
				]
				
		else:
			raise TypeError(
				"A matrix multiplication requires vectors or matrices."
			)

		return Matrix(result)
		
	def hadmul(self, other):
		if isinstance(other, Matrix):
			if self.shape == other.shape:
				return Matrix(
					[
						[x*y for x,y in zip(a, b)] for a, b in zip(
							self.rows, other.rows
						)				
					]
				)
			else:
				raise ValueError(
					"We cannot derive a Hadamard product"
					" of matrices of different dimensions"
				)
		else:
			raise TypeError(
				"Hadamard product requires only matrices"
			)

	def smul(self, other):
		if type(other) in (int, float):
			return (
				Matrix([vector.smul(other).components for vector in self.rows])
			)
		else:
			raise ValueError(
				"Scalar multiplication must involve a sccalar and a matrix"
			)

	def transposed(self):
		return (
			Matrix(
				[
					tuple for tuple in zip(
						*[vector.components for vector in self.rows]
					)
				]
			)
		)

	def add(self, other):
		if isinstance(other, Matrix):
			if self.shape == other.shape:
				return Matrix(
					[
						a.add(b).components for a, b in zip(
							self.rows, other.rows
						)				
					]
				)
			else:
				raise ValueError(
					"Matrices of different dimensions cannot be added"
				)

		else:
			raise TypeError(
				"Matrix addition can only occur between matrices"
			)

	def subtract(self, other):
		if isinstance(other, Matrix):
			if self.shape == other.shape:
				return Matrix(
					[
						a.subtract(b).components for a, b in zip(
							self.rows, other.rows
						)				
					]
				)
			else:
				raise ValueError(
					"Matrices of different dimensions cannot be subtracted"
				)

		else:
			raise TypeError(
				"Matrix subtraction can only occur between matrices"
			)


def idmatrix(n):
	rows = []
	for i in range(n):
		rows.append(
			[0 if x!=i else 1 for x in range(n)]
		)
	
	return Matrix(rows)
