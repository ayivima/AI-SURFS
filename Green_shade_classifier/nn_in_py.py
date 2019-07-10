"""
A simple neural network built with core python,
that classifies shades of green from non-shades of green.

Built as a demonstration and exploration of the basic
functionality of neural networks.

Victor mawusi Ayi
"""

import random
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
    
for x,y in load_images(img_paths, [-1.6,1.8,-1.6]):
    print(x.upper(), y)
