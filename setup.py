from setuptools import setup

setup(name='cimese_net',
      version='0.35',
      description='Basic implementation of a Siamese network structure for detecting copyright infringement',
      long_description='A neural network is implemented here to detect copyrigt infringement violations in newly uploaded video files. The model takes the general form of a Siamese network, with two images filtered through the same convolutional neural network before a classification is made concerning the probability that the two images are a match. In order to train the model, frames were extracted from a set of high-resolution video files of movies and a corresponding set of lower-quality recordings of those movies. Randomly selected frames from the high-quality files were paired with the corresponding frame from the recorded version (a match) as well as a frame from another movie (not a match); these are referred to as triplets. Each of the images is run through the initial set of convolutional layers, which take the structure and weightings from the pre-trained VGG16 neural network, and a vector of length 4096 is returned. This serves as the input to a new set of top layers meant to classify the paired images as a match or not. In addition to the feature extraction and classification model, a means of aligning the recorded clip with the full movie is implemented to optimize the neural net performance. The top level function outputs a single probability of infringement that emerges as the average of the predicted match probabilities along the length of the potentially infringing clip.',
      url='http://github.com/doughertyeric/cimese_net',
      author='Eric Dougherty',
      author_email='dougherty.eric@gmail.com',
      packages=['cimese_net'],
      package_dir={'cimese_net': 'cimese_net'},
      package_data={'cimese_net': ['data/*.data', 'data/*.h5']},
      install_requires=[
	  'keras',
	  'numpy',
	  'opencv-python',
	  'pickle-mixin',
	  'scikit-learn<0.20',
	  'tensorflow'
      ],
      zip_safe=False)
