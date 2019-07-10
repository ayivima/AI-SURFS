SINGLE LAYER SHADES OF GREEN CLASSIFIER IMPLEMENTED IN STANDARD/CORE PYTHON
===========================================================================


A simple neural network built with core python, that classifies shades of green from non-shades of green.
This was built as a demonstration and exploration of the basic functionality of neural networks.

This demonstrates the inherent power of python and the amzing workings or artificial neural networks.

*Victor mawusi Ayi*

Sample Images
-------------
<img src="https://raw.githubusercontent.com/ayivima/AI-SURFS/master/Green_shade_classifier/shot_of_images.png"/>

Code
----

```
from math import exp
from PIL import Image
import numpy as np


def nn(features, weights):
    
    #weights = [-1.7,1.8,-1.7]
    bias = 1
    
    # sigmoid activation function
    def activation(x):
        return (
            1/(1 + exp(1-x))
        )
    
    # get inner product of features and weights
    feature_weight_dot = sum(
        [x*y for x,y in zip(features, weights)]
    )
    
    # add dot product of features and weight to bias
    linear_result = feature_weight_dot + bias
    
    
    return(bool(round(activation(linear_result),3)))
    

def load_images(list_of_image_paths, weights):
    # 
    results = []
    
    for path in list_of_image_paths:
        image = Image.open(path)
        arrays_from_image = np.array(image)
                
        results.append((path, nn(arrays_from_image[25][0], weights)))
    
    print("\n\n----", weights, "\n-----")    
    return results



img_paths = ["img{}.png".format(i) for i in range(1,13)]
    
# Load images and classify them using different weights    
for x,y in load_images(img_paths, [-1.6,1.8,-1.6]):
    print(x.upper(), y)

for x,y in load_images(img_paths, [-1.7,1.8,-1.7]):
    print(x.upper(), y)

for x,y in load_images(img_paths, [-1.8,1.8,-1.8]):
    print(x.upper(), y)    

```

How it works
------------

The load_images() function loads the image and gets the RGB values for its pixels. This system works on only homogenously coloured images. While this condition is satisfied, the load_images() function picks a pixel's RGB, and serves it as input to the neural network, with a specified array of corresponding weights. The bias is always 1. 

The summation of the bias and the inner product of the RGB features and weights, are passed through a sigmoid activation function and a boolean equivalent of the output is returned. `True` is returned if the image is a shade of green, and `False` is returned if it is not.


Outcomes and Performance
------------------------

1. Using weights [-1.6,1.8,-1.6]

```
[-1.6, 1.8, -1.6]
==================
[x] IMG1.PNG True
[x] IMG2.PNG True
[x] IMG3.PNG True
[x] IMG4.PNG False
[x] IMG5.PNG False
[ ] IMG6.PNG True
[x] IMG7.PNG False
[x] IMG8.PNG False
[x] IMG9.PNG True
[x] IMG10.PNG False
[x] IMG11.PNG True
[x] IMG12.PNG True
```

<img src="https://raw.githubusercontent.com/ayivima/AI-SURFS/master/Green_shade_classifier/shot_of_images.png"/>


2. Using weights [-1.7,1.8,-1.7]

```
[-1.7, 1.8, -1.7]
==================
[x] IMG1.PNG True
[x] IMG2.PNG True
[x] IMG3.PNG True
[x] IMG4.PNG False
[x] IMG5.PNG False
[ ] IMG6.PNG True
[x] IMG7.PNG False
[x] IMG8.PNG False
[x] IMG9.PNG True
[ ] IMG10.PNG False
[ ] IMG11.PNG False
[ ] IMG12.PNG False
```

The weights `[-1.6, 1.8, -1.6]` produced the best classification.


Conclusion
----------

- But for the trouble with IMG6.PNG, the neural network did great classifying shades of green from the sample images.
- Changing the weights changed the classification and this demonstrates the refinement that is automatically achieved using pytorch's autograd and backpropagation.
- Neural networks are amazing!


