
# Python Tricks

It is a compilation of few common tricks that we can use in python which are comparitively easier with other languges.
Feel free to contribute to these. 

## Strings

### Reverse a string


```python
s = "Reverse"
#To reverse a string in python, use the following.
s = s[::-1] # this will reverse the string
print(s)
```

    esreveR
    

### Separating a sentence into words

we can use the str.split() function to split long sentence into list of substring.
https://docs.python.org/3/library/stdtypes.html#str.split


```python
s = "one two three"
print(s.split())
```

    ['one', 'two', 'three']
    


```python
s = "1,2,3"
#splitting by comma
s.split(",")
```




    ['1', '2', '3']




```python
#above on can be extended with any substring
s="1/ /2/ /3/ /4/ /5"
s.split('/ /')
```




    ['1', '2', '3', '4', '5']


