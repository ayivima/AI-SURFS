
PLAYING WITH MATRICES
=====================

[ Victor Mawusi Ayi ]

Disclaimer! This is a spontaneous play around with tensors in
python, which may need some optimisation. This is just for
demonstration, and a tip of a project underway.

Just thought if we played around with code a bit, we would get to the place where we could quickly mobilise a small neural network when we do not have the resources to run powerful libraries...And we could do it from the scratch.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Lets see a rough sketch of how we could quickly mobilise a matrix class, and possibly optimise it later, if applicable.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

MATRIX CLASS
============

Let's write a small matrix class
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    class Matrix():
    
        def __init__(self, seq_of_seq):
            
            # seq_of_seq stands for sequence of sequence
            # my fancy way of calling N-dimensional arrays
            self.value = seq_of_seq
            
            # for a N-dimensional arrays, 
            # number of rows is simply its length
            self.num_of_rows = len(seq_of_seq)
            
            if self.num_of_rows < 2: 
                raise TypeError(
                    "A matrix cannot have just one row!"
                )
            
            # for an array of arrays, which translates to a table or matrix
            # number of columns should be equal to the 
            # length of its longest sub-arrays,
            # with the assumption that the shorter arrays could be extended
            # with zeros into the dimensional space of the longest array.
            self.num_of_cols = len(
                sorted(seq_of_seq, key = lambda seq: len(seq))[-1]
            )
            
            max_length = lambda seq: max([len(str(i)) for i in seq])
            long_num_len = 1
            
            for t in seq_of_seq:
                new_max = max_length(t)
                if new_max > long_num_len:
                    long_num_len = new_max
            
            self.long_num_len = long_num_len + 2
            
            if self.num_of_cols < 2: 
                raise TypeError(
                    "A matrix cannot have just one column!"
                )
    
            self.shape = (self.num_of_rows, self.num_of_cols)
    
        def __repr__(self):
            return "matrix [{}]".format(
                "\n".ljust(9).join([
                    " ".join([
                        str(num).ljust(self.long_num_len) for num in self.extend(array)
                    ]).rjust(10, " ") for array in self.value
                ]).rstrip()
            )
    
        def describe(self):
            return "{} X {} Matrix".format(*self.shape)
        
        def extend(self, array):
    
            extension_list = [0] * (self.num_of_cols - len(array))
            return array + extension_list
        

------------------------------------------------------------------------------------------
==========================================================================================

PLAYGROUND STARTS HERE...
=========================

Example 1
~~~~~~~~~

::

    >>> a = Matrix([[2,3], [4,5], [5, 6, 7]])
    >>> a
    matrix [2   3   0  
            4   5   0  
            5   6   7]

::

    >>> a.shape
    (3, 3)



Example 2
~~~~~~~~~

.. code:: ipython3

    >>> b = Matrix([[0.2,0.4567,0.34], [0.657, 8.9, 7], [90.8762, 89736.09, 562.89]])
    >>> b
        matrix [0.2        0.4567     0.34      
                0.657      8.9        7         
                90.8762    89736.09   562.89]


::

    >>> b.describe()
    '3 X 3 Matrix'



Example 3
---------

::

    >>> c = Matrix([[0.2,0.4567], [0.657, 8.9, 7], [90.8762, 89736.09, 562.89, 9983.654]])
    >>> c
    matrix [0.2        0.4567     0          0         
            0.657      8.9        7          0         
            90.8762    89736.09   562.89     9983.654]
            
::

    >>> c.shape
    (3, 4)


