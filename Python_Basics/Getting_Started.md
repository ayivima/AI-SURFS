

PYTHON AT A GLANCE
==================

**Victor Mawusi Ayi**


A BRIEF INTRODUCTION
--------------------

Python is trending everywhere in Tech these days. It is used everywhere software is possible...And...it's the defacto for 
traditional Machine Learning(ML), Deep Learning(DL) and Artificial Intelligence(AI). Well, to prevent debate, it is the most used language for ML, DL and AI :). 
Evidently, it is useful to understand the core concepts of python before delving into ML, DL, and AI. This attempts a quick surf on the waves 
of the most beloved AI language.


FIRST CONCEPTS
--------------

Every language has data types. Types allow us to interact with our data - they determine how we store them, modify them, and how we can 
use combine different kinds of data to achieve a purpose.

For example, 
`2` is a number, and `"two"` is a word. We cannot add `2` and `"two"` in real life but computers do not know that. By having data types, 
languages can tell computers that this is not done as well, and we specify that **Numbers** cannot be added to **Strings**. 

With that overview, let's explore the basic data types in python.

Integer
-------
The integer type, represents integers, or simply, numbers without decimal places. Eg. `-1`, `2`, `-25`
Let's have some practicals...Note that we use `print` to display a value in python. Let's assign `2` to a variable called 
first_integer:

```
>>> first_integer = 2
>>> print(first_integer)
2
```
Let's see how python considers the type of `2`. We use `type()` to check a value's data type.
```
>>> type(2)
<class 'int'>
```
Truly, it is an integer! Note that, Python calls its integers `int`.

Float
-----
The float type, in simple terms, represents numbers with decimal places.... 
Eg. `1.0998475`, `-0.2`, `10E-3`

Were you wondering what 10E-3 is? Let's assign it to a variable called `first_float` and print it out.
```
>>> first_float = 10E-3
>>> print(first_float)
0.01
>>>
```
Let's see how python calls decimal numbers.
```
>>> type(2.5)
<class 'float'>
>>> type(10E-3)
<class 'float'>
```
Voila! `10E-3` and `2.5` are of `float` type.

But before we continue, the `float` basic type deals with decimal numbers slightly differently from how we use them in real life. 
For this reason, python comes with another type called `decimal` which deals with decimal numbers in a more friendlier way. 
But, to use it we have to import it, and we have to explicitly tell python that we want our numbers to be decimals and not floats.
For, the purpose of sticking to what this tutorial is about, a quick review on python data types, 
I will provide the link for extra reading on the `decimal` type: https://docs.python.org/3/library/decimal.html

String
------
The simplest analogy for string is text. Eg. `"two"`, `"three"`, `"""four"""`, `'456'`

Some practical:
```
>>> string1 = "two"
>>> print(string1)
'two'
```
`456` and `"456"` are not the same
```
>>> 456 == "456"
False
```
We use `==` to check if two values are equal.

What about `"""This"""`? Triple quoted strings are specially used for specfying text that uses more than one line.

Practically:
```
a = """
This is a multiline string.
Observe how it takes more than one line.
And we can keep going
"""

print(a)
```
Let's run this, and we get our text printed out nicely.
```
This is a multiline string.
Observe how it takes more than one line.
And we can keep going
```
Let's try the same feat with double quotes:
```
b = "
This is an interesting string.
I am wondering if a multiline string ever uses double quotes.
Let's see!
"

print(b)
```
Let's see what happens if we try to run:
```
  File "test.py", line 1
    b = "
        ^
SyntaxError: EOL while scanning string literal
```
This is not the place for double strings :-). For the most basic use case, reserve triple quotes for multiline strings.
Triple quotes are also used for what is called **docstrings**. These, are used to provide information about functions, classes etc. 
Will not dive deep into that for this tutorial.

But wait. In case you wanted to use double quotes to write multiline strings you could use the special character for NEWLINE written as 
"\n".

Practically, like this: Note how a `\n` is place after every line.
```
b = "This is an interesting string.\nI am wondering if a multiline string ever uses double quotes.\nLet's see! It should work this way.\n"

print(b)
```
And here we go:
```
This is an interesting string.
I am wondering if a multiline string ever uses double quotes.
Let's see! It should work this way.
```

Boolean
-------
What would we do without `True` or `False`. Boolean types represent `True` and `False`.


We are moving to some interesting data types called Data Structures. Integers, Floats, String, Boolean, all allow us to store one 
value at a time. What if we wanted to store several numbers at a time, or several words, or even a mix of words and numbers? Data structures come 
to the rescue. 

Tuples and Lists
----------------
- **Tuple** allows us to store a sequence of values at once. 
We initialize a tuple by enclosing given values in brackets `()`. 
Eg. `(1,2,3,4,5)`, `("bat", "splendid", "at")`, `("is", 5, "it")`

- **List** also allows us to store a sequence of values at once. Eg. `[1,2,3,4,5]`, `["bat", "splendid", "at"]`.
We initialize a list by enclosing given values in square brackets `[]`.

Are there differences between tuples and lists?

- You can add new values to lists but not tuples
- You can delete values from lists but not tuples
- You can also change values in lists but not tuples.

When we can change values in a data structure, after it has been stored in memory, we say it is MUTABLE. 
When we can't, we say it is IMMUTABLE.

We can get a value from a tuple by using the `index`, the index is the serial position of a value within a set of values counting the 
first position as zero.

For a tuple, `(1,2,3,4)`, the index of `1` is 0, the index of `3` is 2.

Practically,
```
>>> a_tuple = (1, 2, 3, 4)
>>> a_tuple[0]
1
>>> a_tuple[2]
3
```

Same goes for list:
```
>>> a_list = [1, 2, 3, 4]
>>> a_list[0]
1
>>> a_list[2]
3
```


Dictionary
----------
Allows us to store values in pairs. It is useful because we can tell what a value is representing. 

For example, 
If we had the ages of Anne, James, Greg as 10, 12, 14, and we wanted to store them, 
we could use a list or tuple as follows:
```
>>> ages = [10, 12, 14]
```
or
```
ages = (10, 12, 14)
```
But, if we forgot the order we stored the ages, we wouldn't know whose age was which one. 
To solve this, we could use a dictionary to store them like this:
```
>>> ages = {"Anne":10, "James":12, "Greg":14}
```
And we can always find out the age of any of them by doing:
```
>>> ages["Anne"]
10
>>> ages["James"]
12
```
In a more meaningful way, we could do this:
```
>>> ages.get("Greg")
14
```

Set
---
A set behaves like a list. Except, it cannot be indexed. Additionally, it does not store duplicate values. 
It is initialized by enclosing values in curly brackets, `{}`.

CONVERTING BETWEEN DATA TYPES
-----------------------------

There is a time when you get data in a format that is not useful to what you want to do.

For example, 
You get data like this, `["1", "2", "3", "4"]` when you need this, [1, 2, 3, 4].

Let's see how we can do some conversions:

General Form
------------
We use `int()`, `float()`, `bool()`, `str()`, `tuple()`, `list()`, `set()`, `dict()` to convert values into 
integer, float, boolean, string, tuple, list, set, and dictionary respectively.



String to Numbers
-----------------
We can convert a string to a number if it represents that type of number:

We cannot convert a number that looks like a float, to an integer.
```
>>> str1 = "426.78"
>>>
>>> int(str1)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
ValueError: invalid literal for int() with base 10: '426.78'
```

```
>>> str1 = "426.78"
>>>
>>> float(str1)
426.78
```

Number to String
----------------

We can convert any number to string:

```
>>> str(3)
'3'
>>> str(4.562)
'4.562'
```

Convert to Boolean
------------------

Almost anything can be converted to boolean. Anything, except `0`, `''`, `""`, `""""""`, empty data structures like `[], (), {}`, and `None`, converts to `True`.

```
>>> bool("boy")
True
>>> bool(4)
True
>>> bool(-1)
True
>>> bool(0)
False
>>> bool(None)
False
>>> bool([])
False
```

String to List, Tuple or Set
----------------------------

```
>>> a = "123456789"
>>> 
>>> list(a)
['1', '2', '3', '4', '5', '6', '7', '8', '9']
>>>
>>> tuple(a)
('1', '2', '3', '4', '5', '6', '7', '8', '9')
>>>
>>> set(a)
{'9', '1', '5', '7', '8', '6', '2', '3', '4'}
```

From a sequence of sequences to dictionary
------------------------------------------
Note, each sub sequence must have length of 2 for this to work.
```
>>> r = [["a", 1], ["b", 2]]
>>> dict(r)
{'a': 1, 'b': 2}
>>>
>>> t = (("a", 1), ("b", 2))
>>> dict(t)
{'a': 1, 'b': 2}
>>>
```

Explore more:
-------------

- https://docs.python.org/3/tutorial/introduction.html
- https://docs.python.org/3/reference/datamodel.html#objects-values-and-types
- https://www.w3schools.com/python/python_variables.asp
- https://www.w3schools.com/python/python_numbers.asp
- https://www.w3schools.com/python/python_casting.asp





