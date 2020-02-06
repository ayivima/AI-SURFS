##### Victor Mawusi Ayi

What is transpose?
------------------

Most of the operations in AI depend on Matrix math, and several concepts including `Transpose` pertain to matrices. 
From matrix operations, we use transpose to flip a matrix. Then, its rows become columns, and columns become rows.

![](/imgs/transpose.png)

This can be demonstrated in code:

```
>>> import numpy as np
>>>
>>> matrix = np.array([[1,1,1],[2,2,2],[3,3,3]])
>>>
>>> matrix
array([[1, 1, 1],
       [2, 2, 2],
       [3, 3, 3]])
>>>
>>> matrix.transpose()
array([[1, 2, 3],
       [1, 2, 3],
       [1, 2, 3]])
```

How does transpose become useful in image analysis?
---------------------------------------------------

We have different ways of storing image information for computers. However, generally, they are stored as big nested matrices which store information on the height, width and [channels](/docs/channels.md) of the given images.
For example, we can create a simple rectangle having red, blue, and green colors as shown below:

```
import numpy as np
import matplotlib.pyplot as plt

image = np.array(
    [[[255,0,0],[255,0,0]],
     [[0,255,0],[0,255,0]],
     [[0,0,255],[0,0,255]]]
)

plt.imshow(image)
```
OUTPUT:

![](/imgs/samp_image_transpose.png)

When we want a peek at the information of the image we just created, using `.shape`, we realize that its storage format is in this order: 
 + height(number of pixel rows), 
 + width(number of pixel columns), and then 
 + channels(number of color channels) 
 
```
>>> print("Image shape: ", image.shape)
Image shape:  (3, 2, 3)
```
This format is abbreviated as (H X W X C). 

We can look at it graphically as below:

![](/imgs/samp_image_transpose2b.png)

Some libraries store images using the format `channels`, `height` and `width` (C X H X W). 
Therefore, when we are using several libraries together, we can have a problem passing images around, unless we change the format in which image information is stored, to suit a library we want to use at a given time.
This is what we achieve with transpose. We can use it to flip/rotate the image matrices. As an example, if the format was `H X W X C`, we can get `C X H X W` and vice versa.
For computers everything becomes a number or math at a time. So, we can, and, use the indices `0`, `1`, `2` to represent the `height`, `width`, `channels`. Then, when we want to change the format to `channels`, `height`, `width` we specify the order `2`, `0`, `1`.

![](/imgs/image_transpose.png)

This is demonstrated in code below.

```
>>>
>>> image.shape
(3, 2, 3)
>>>
>>> image
array([[[255,   0,   0],
        [255,   0,   0]],

       [[  0, 255,   0],
        [  0, 255,   0]],

       [[  0,   0, 255],
        [  0,   0, 255]]])
>>>
>>> transposed_image = image.transpose(2,0,1)
>>> transposed_image
array([[[255, 255],
        [  0,   0],
        [  0,   0]],

       [[  0,   0],
        [255, 255],
        [  0,   0]],

       [[  0,   0],
        [  0,   0],
        [255, 255]]])
>>>
>>> transposed_image.shape
(3, 3, 2)
>>>
```
```
