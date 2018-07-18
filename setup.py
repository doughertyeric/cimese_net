from setuptools import setup

setup(name='cimese_net',
      version='0.10',
      description='Basic implementation of a Siamese network structure for classifying sets of images as matching or not matching',
      long_description='Using a set of high quality films and lower quality recordings of those films, individual frames were extracted and a training set of image triplets was developed. Using the VGG16 convolutional neural net (as implemented in Keras), each frame was encoded as a vector of length 4096. Using this as the input to a new set of top layers, a classifier was created that would output the probability that the two images were a match or not. In addition, a means of aligning the clip with the full film is available to optimize the neural net performance. The output is a single infringement probability averaged over the length of the potentially infringing clip. ',
      url='http://github.com/doughertyeric/cimese_net',
      author='Eric Dougherty',
      author_email='dougherty.eric@gmail.com',
      packages=['cimese_net'],
      install_requires=[
	  'keras',
	  'numpy',
	  'opencv-python',
	  'pickle-mixin',
	  'scikit-learn==0.19.2'
      ],
      zip_safe=False)
