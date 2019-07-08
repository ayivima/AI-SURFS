
from PIL import Image
import numpy as np


def get_image_array(image_path):
	
	image = Image.open(image_path)
	array_from_image = np.array(image)
	
	return array_from_image

	
def coalesce_into_column(multidir_image_array):
    single_array = []
    a,b,c = multidir_image_array.shape
	
	# we are hitting an interesting time complexity here
	# we could possibly look out for a snappier way
	# not found one yet
    for i in range(a):
        for j in range(b):
            for k in range(c):
                single_array.append(multidir_image_array[i][j][k])
    return single_array, len(single_array)
	

def get_corr(array_of_image_arrays, slice_):
	# slice_ alllows us to slice images into same length somehow :)
	
	return np.corrcoef(
		[array[:slice_] for array in array_of_image_arrays]
	)

	
def corr_of_multiple_images(list_of_paths):
	array_of_image_arrays = []
	minimum_length = float("+inf")
	
	# hitting another interest chaining
	# 
	for image_path in list_of_paths:
		
		array, length = coalesce_into_column(
			get_image_array(image_path)
		)
		
		# minimum_length will be useful for us to compare images
		# using the rough size of the smallest image
		minimum_length = (
			minimum_length if length>minimum_length else length
		)
		
		array_of_image_arrays.append(array)
	
	# In case the images are not the same, the 
	# the minimum length helps us to slice all images to 
	# same size. Would be better if the shapes are same.
	return get_corr(array_of_image_arrays, minimum_length)


# You just need to submit a list of the image names to 
# corr_of_multiple_images() function if the images are many

# change the image names available in your folder
# this was for my demo

print(corr_of_multiple_images(["samp1.jpeg","samp2.jpeg","samp3.jpeg","samp4.jpeg"]))	

 
	