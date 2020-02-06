##### Victor Mawusi Ayi

## What is flattening?

When you flatten an image, you simply convert it to a vector or, in some cases, a matrix with only one row.

![](/imgs/flatten1.png)

As an example, Consider a tensor, `X = [[1,2,3],[4,5,6],[7,8,9]]`. When we flatten X, we end up with `[1,2,3,4,5,6,7,8,9]` or `[[1,2,3,4,5,6,7,8,9]]`. This is the simplest way to look at it.

## How can we do it in code?

Most of the libraries have these methods available:

+ flatten()
+ view()
+ reshape()

We will use numpy to demonstrate the uses of these methods above. First of all, we get our simple red and green rectangular image and display it. Then, subsequently, we flatten it.

```

>>> import matplotlib.pyplot as plt
>>> import numpy as np
>>> 
>>> # Image tensor
>>> image = np.array(
...     [[[255, 0, 0], [255, 0, 0]],
...      [[0, 255, 0], [0, 255, 0]]]
... )
>>>
>>>
>>> # Display image
>>> plt.imshow(image)
```
OUTPUT: 

![](/imgs/flatten_samp_image.png)

We can now flatten this image and see what it becomes.
```
>>> image
array([[[255,   0,   0],
        [255,   0,   0]],

       [[  0, 255,   0],
        [  0, 255,   0]]])
>>>
>>> image.flatten()
array([255,   0,   0, 255,   0,   0,   0, 255,   0,   0, 255,   0])
>>> 
```
We can do the same as above using `reshape()`. Because we are looking at a vector or a simple array, we must pass the total number of values(items) in the tensor as a parameter.
We have 12 values in our rectangular image above(Count all the numbers in the image tensor). Therefore, we can alternatively flatten by doing `reshape(12)`.

```
>>> image
array([[[255,   0,   0],
        [255,   0,   0]],

       [[  0, 255,   0],
        [  0, 255,   0]]])
>>>
>>> image.reshape(12)
array([255,   0,   0, 255,   0,   0,   0, 255,   0,   0, 255,   0])
>>> 
```

Sometimes, we will not know how many values(items) there are in an image tensor. In reality, we will almost always not know, and even for large images we cannot afford to count all items.
But, not knowing should not get in the way of what we want to do, because we can ask a library to automatically count the number of items for us.
The way we ask a library to figure the number of items for us is to specify `-1` as the parameter to `reshape` or `view`(if present in the given library).

```
>>> image
array([[[255,   0,   0],
        [255,   0,   0]],

       [[  0, 255,   0],
        [  0, 255,   0]]])
>>>
>>> image.reshape(-1)
array([255,   0,   0, 255,   0,   0,   0, 255,   0,   0, 255,   0])
>>> 
```
And, Bingo!!! We have the same outcome.


Sometimes, we do not want a plain vector(or simple array). We may prefer a matrix with a single row carrying all the items. That means we want as many columns to accomodate all the items in the image tensor.
In this case, we specify `1` (row) as the first parameter, and `-1`(column) as the second parameter. As we know, for a shape of a tensor, the first value represents number of rows and the second value represents number of columns. Additionally, as explained above, 
we specify `-1` for the columns so that the library will figure it out automatically. Let's try that in code:

```
>>> image
array([[[255,   0,   0],
        [255,   0,   0]],

       [[  0, 255,   0],
        [  0, 255,   0]]])
>>>
>>> image.reshape(1,-1)
array([[255,   0,   0, 255,   0,   0,   0, 255,   0,   0, 255,   0]])
```






