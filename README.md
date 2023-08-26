# Strawberry-Leaf-Classification-Using-Image-Processing-and-Deep-Learning

The code demonstrates a project that uses a Convolutional Neural Network (CNN) to predict the status of a leaf as either "Healthy" or "Calcium Deficient" based on input images.

It uses the TensorFlow and Keras libraries for building and training the CNN model.

# Data Loading and Preprocessing:

1.The code starts by defining the dimensions of the input images (128x128 pixels).

2.The load_data function reads and preprocesses the leaf images from the given data path. It converts images to arrays, normalizes the pixel values to the range [0, 1], and one-hot encodes the labels.

# Model Architecture:

1.The CNN model architecture is defined using the create_model function.

2.The model consists of several Convolutional and MaxPooling layers to learn hierarchical features from the images.

3.It ends with a couple of fully connected layers with dropout for classification.

4.The model is compiled with the Adam optimizer and categorical cross-entropy loss.

# Data Split and Training:

1.The data is split into training and testing sets using a 80-20 split ratio.

2.The model.fit function trains the CNN on the training data, specifying batch size, epochs, and a validation split.

3.The training history is stored for later visualization.

# Training History Plotting:

1.The plot_training_history function takes the training history and plots the training/validation loss and accuracy curves over epochs.

# Model Evaluation:

1.The trained model is evaluated on the test set to measure its performance in terms of loss and accuracy.

# Leaf Status Prediction:

1.The predict_leaf_status function takes a trained model and an image path as input.

2.It loads the image, preprocesses it, and predicts the leaf status using the model.

3.The prediction is based on the predicted probabilities of the classes.

# Image Display and Prediction:

1.The code provides paths to two leaf images ('/content/test.jpg' and '/content/test5.jpg').

2.It loads and displays each image using Matplotlib.

3.The predict_leaf_status function is called for each image to predict the leaf status.

4.The predicted status is printed for each image.

# Summary :

1.Project focuses on predicting leaf status using a CNN model.

2.Data loading includes converting images to arrays and one-hot encoding labels.

3.Model architecture comprises Convolutional and MaxPooling layers.

4.Training involves splitting data, using model.fit, and storing training history.

5.Training history is plotted to visualize loss and accuracy trends.

6.Model evaluation is performed on the test set.

7.Leaf status prediction is done through preprocessing and using the trained model.

8.Provided images are displayed, and predicted statuses are printed.

# Note:
Please replace '/content/test.jpg' and '/content/test5.jpg' with actual paths to your test images for accurate results.
