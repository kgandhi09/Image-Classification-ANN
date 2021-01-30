# Image Classification using Artificial Neural Network(ANN)
![sample-digit-images](https://user-images.githubusercontent.com/36654439/106346159-093a8500-6283-11eb-842c-a6690615e7af.png)

In this project, I built Artificial Neural Network for categorizing images. I wrote a program that takes images like the hand-written numbers above and output what numbers the images represent.

### Dataset
The dataset used for this project is taken from the MNIST database of handwritten digits. Each MNIST image is 28x28 grey-scale image. Data is provided as 28x28 matrices containing numbers ranging from 0 to 255. Labels images are also provided with integer values ranging from 0 to 9, corresponsidng to actual value in the image. I used 6500 images and 6500 corresponding labels in this project.

### Preprocessing Data
Image data was provided as 28x28 matrices of integer pixel values as Numpy .npy files. However, the input to the network was a flat vector of length 28x28 = 784. All the 28x28 matrices were flattened to be a vector of length 784. Also, the output/prediction had to be a hot vector instead of an integer, thus, all the corresponding labels were converted to hot vectors. 

### Training, Testing and Validation Sets
After preprocessing, the data was randomly split into Training, Validation and Test Sets. In order to create the three sets of data, I used stratified sampling, so that each set contains the same relatively frequency of the ten classes. In this project, training set contained ~60% of the data, the validation set contained ~15% of the data, and the test set contained ~25% of the data.

### Straitified Sampling
For the stratified sampling procedure, I took data and separated it into 10 classes, one for each digit. From each class, took 60% at random and put into the Training Set, took 15% at random and put into the Validation Set, took the remaining 25% and put into the Test Set.

### Building a Model
I used Keras to build the model, where models are instantiations of class Sequential. I mainly experimented with activation units such as ReLu, SeLu and Tanh. I also experimented with number of layers and number of neurons in each layer.

#### For more details on the performance of the model and how the model was experimented, please have a look at: https://drive.google.com/file/d/1_3FlaQugAKLtB0JQGFnIbHG_PMItBKGc/view?usp=sharing 
