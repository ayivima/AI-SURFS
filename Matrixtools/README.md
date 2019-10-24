
Matrix Arithmetic
-----------------

Leveraging the flexibility and speed of standard python data structures to make matrix 
arithmetic handy for the beginner or pro python programmer.


```
D:\60AI\matrixtools\matrixkit>python
Python 3.7.1 (v3.7.1:260ec2c36a, Oct 20 2018, 14:05:16) [MSC v.1915 32 bit (Intel)] on win32
Type "help", "copyright", "credits" or "license" for more information.
>>>
>>> from matrixtools import Matrix
>>>
>>> x = Matrix([[1,2],[2,1]])
>>>
>>> x.flattened()
Matrix([1.0 2.0 2.0 1.0])
>>>
>>> x
Matrix([   1.0 2.0
           2.0 1.0])
>>>
>>> y = Matrix([[2,1],[1,2]])
>>>
>>> x.matmul(y)
Matrix([   4.0 5.0
           5.0 4.0])
>>>
>>> x.hadmul(y)
Matrix([   2.0 2.0
           2.0 2.0])
>>>
>>> x.add(y)
Matrix([   3.0 3.0
           3.0 3.0])
>>>
>>> x + y
Matrix([   3.0 3.0
           3.0 3.0])
>>>
>>> x**2
Matrix([   1.0 4.0
           4.0 1.0])
>>>
>>> from matrixtools import idmatrix
>>>
>>> idmatrix(3)
Matrix([1.0 0.0 0.0
        0.0 1.0 0.0
        0.0 0.0 1.0])
>>>
>>> idmatrix(7)
Matrix([1.0 0.0 0.0 0.0 0.0 0.0 0.0
        0.0 1.0 0.0 0.0 0.0 0.0 0.0
        0.0 0.0 1.0 0.0 0.0 0.0 0.0
        0.0 0.0 0.0 1.0 0.0 0.0 0.0
        0.0 0.0 0.0 0.0 1.0 0.0 0.0
        0.0 0.0 0.0 0.0 0.0 1.0 0.0
        0.0 0.0 0.0 0.0 0.0 0.0 1.0])
>>>
```
