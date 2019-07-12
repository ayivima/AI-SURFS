
RED, GREEN, BLUE CLASSIFIER IMPLEMENTED IN STANDARD/CORE PYTHON
===========================================================================
*Victor mawusi Ayi*

A simple neural network built with core python, that classifies shades of red, green and blue. 
An advancement on the 'shades of green' classifier, this neural network has an one input layer, 
one hidden layer with 3 nodes, and the output layer with a softmax activation function.
This is a continuation of the demonstration and exploration of the basic functionality of neural networks 
using core python. Consequently, this will strengthen the appreciation of pytorch. 

Architecture
------------
<img src="https://raw.githubusercontent.com/ayivima/AI-SURFS/master/Red_Green_Blue_Classifier/archtecture_nn_2.png"/>

Each selected pixel goes through each of the nodes of the hidden layer, each node selective for a particualr color. Then, the outputs from the hidden nodes are passed through through softmax function at the output node, and the class is assigned based on the maximum probability, and on a first-in selection principle in the case of a tie. This is just the beginning that will get refined as I add more hidden layers.

Sample Images
-------------
<img src="https://raw.githubusercontent.com/ayivima/AI-SURFS/master/Green_shade_classifier/shot_of_images.png"/>

Code
----

```
from math import exp
from PIL import Image
import numpy as np
import random


def node(features, weights):
    
    bias = 1
    
    # sigmoid activation function
    def activation(x):
        return (
            1/(1 + exp(-x))
        )
    
    # get inner product of features and weights
    feature_weight_dot = sum(
        [x*y for x,y in zip(features, weights)]
    )
    
    # add dot product of features and weight to bias
    linear_result = feature_weight_dot + bias
    
    
    return round(activation(linear_result),3)


def softmax(X):

    exps_of_x = [exp(i) for i in X]
    sum_of_exps = sum(exps_of_x)

    return [x_i/sum_of_exps for x_i in exps_of_x]


def classifier(list_of_image_paths):
    classes = [
        "RED",
        "GREEN",
        "BLUE",
    ]

    weights = [
        [0.8,-0.6,-0.6],
        [-0.6,0.8,-0.6],
        [-0.6,-0.6,0.8]
    ]
    results = []

    for path in list_of_image_paths:
        image = Image.open(path)
        arrays_from_image = np.array(image)

        # select random pixel. This is for demonstration purposes.
		select_point = random.randint(0,49)

        input = arrays_from_image[select_point][0]

        # hidden layers		
        hiddens = [node(input, weight) for weight in weights]

        # output layers
		output = softmax(hiddens)

        # get the maximum probability
        max_probability = max(output)

        # get the first index of the maximum probability
        shade_index = output.index(max_probability)

        # get the class of the layer
        color_class = classes[shade_index]

        # add the class of colour to results
        results.append((path, classes[shade_index]))

    return results


if __name__ = "__main__":
	img_paths = ["img{}.png".format(i) for i in range(1,17)]

    # Load and classify images   
    for x,y in classifier(img_paths):
        print(x.upper(), y)

```


Outcomes and Performance
------------------------

<img src="https://raw.githubusercontent.com/ayivima/AI-SURFS/master/Red_Green_Blue_Classifier/shot_of_outcomes2.png"/>

1. Using the weights [0.8,-0.6,-0.6], [-0.6,0.8,-0.6], [-0.6,-0.6,0.8] for the RED, GREEN, BLUE nodes of the hidden layer.

```
[x] IMG1.PNG GREEN
[x] IMG2.PNG GREEN
[x] IMG3.PNG GREEN
[x] IMG4.PNG BLUE
[x] IMG5.PNG BLUE
[x] IMG6.PNG RED
[x] IMG7.PNG BLUE
[x] IMG8.PNG RED

```

<img src="https://raw.githubusercontent.com/ayivima/AI-SURFS/master/Red_Green_Blue_Classifier/shot_of_images2.png"/>


```
[x] IMG10.PNG GREEN
[x] IMG11.PNG RED
[x] IMG12.PNG GREEN
[x] IMG13.PNG RED
[x] IMG14.PNG BLUE
[x] IMG15.PNG GREEN
[x] IMG16.PNG RED

```

Conclusion
----------

- It does well on Red, Green and blue colors. They are all it knows for now
- Python and Neural networks are amazing!


