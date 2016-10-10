import numpy as np
from sklearn.datasets import load_sample_image

array = np.zeros((2, 3, 4))
w, h, d = tuple(array.shape)
#X = np.array([[[1., 0.], [2., 1.], [0., 0.]]])
X = np.array([[1, 2], [1, 4], [1, 0], [4, 2], [4, 4], [4, 0]])
china = load_sample_image("china.jpg")
china = np.array(china, dtype=np.float64)/255


"""print china.size
print "Hello"
print china[:].size"""
print X.shape