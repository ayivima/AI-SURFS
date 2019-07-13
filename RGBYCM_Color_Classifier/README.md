

RED, GREEN, BLUE, YELLOW, CYAN, MAGENTA CLASSIFIER IMPLEMENTED IN STANDARD/CORE PYTHON
======================================================================================
*Victor mawusi Ayi*

An advancement on the red-green-blue classifier, with ah additional hidden layer with 6 nodes. 

Architecture
------------
<img src="https://raw.githubusercontent.com/ayivima/AI-SURFS/master/RGBYCM_Color_Classifier/architecture_nn_3.png"/>

Each selected pixel goes through each of the nodes of the first hidden layer, each node selective for either of red, green, and blue primary colors. Then, the outputs from the first hidden layer proceed to the second hidden layer, with 6 nodes selective for either of red, green, blue, yellow, cyan, and magenta. Finally, the outputs from the hidden nodes are passed through the softmax function at the output node, and the class is assigned based on the maximum probability, and on a first-in selection principle in the case of a tie. This is just a exploration of the possibilities with building neural networks in core python.

Sample Images
-------------
<img src="https://github.com/ayivima/AI-SURFS/blob/master/RGBYCM_Color_Classifier/shot_of_images3.png"/>

Code
----

```
"""

A simple neural network built with core python, 
that classifies shades of red, green, blue, yellow, magenta, cyan.

Built as a demonstration and exploration of the basic 
functionality of neural networks.

Victor Mawusi Ayi

"""
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
        "YELLOW",
        "CYAN",
        "MAGENTA"
    ]

    weights1 = [
        [0.8,-0.6,-0.6],
        [-0.6,0.8,-0.6],
        [-0.6,-0.6,0.8]
    ]

    weights2 = [
        [0.8,-0.6,-0.6],
        [-0.6,0.8,-0.6],
        [-0.6,-0.6,0.8],
        [0.8,0.8,-0.6],
        [-0.6,0.8,0.8],
        [0.8,-0.6,0.8]
    ]

    results = []

    for path in list_of_image_paths:
        image = Image.open(path)
        arrays_from_image = np.array(image)

        # select random pixel. This is for demonstration purposes.
        select_point = random.randint(0,49)

        input = arrays_from_image[select_point][0]

        # hidden layers		
        hiddens1 = [node(input, weight) for weight in weights1]
        hiddens2 = [node(hiddens1, weight) for weight in weights2]

        # output layer
        output = softmax(hiddens2)

        # get the maximum probability
        max_probability = max(output)

        # get the first index of the maximum probability
        shade_index = output.index(max_probability)

        # get the class of the layer
        color_class = classes[shade_index]

        # add the class of colour to results
        results.append((path, classes[shade_index]))

    return results


if __name__ == "__main__":
    img_paths = ["img{}.png".format(i) for i in range(1,17)]

    # Load and classify images
    for x,y in classifier(img_paths):
        print(x.upper(), y)

```


Outcomes and Performance
------------------------

<img src="https://raw.githubusercontent.com/ayivima/AI-SURFS/master/RGBYCM_Color_Classifier/nn_py3_shot.png"/>

<img src="https://github.com/ayivima/AI-SURFS/blob/master/RGBYCM_Color_Classifier/shot_of_images3.png"/>


