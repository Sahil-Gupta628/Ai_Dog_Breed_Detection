# Ai_Dog_Breed_Detection

## Dog Breed Detection App
- This project implements a Dog Breed Detection application using Deep Learning techniques to classify dog breeds from images. The model is built using TensorFlow and Keras. The app is developed with Streamlit, providing an easy-to-use interface for users to upload dog images and receive breed predictions.

## Features
- Upload an image of a dog
- Predict the breed of the dog using a deep learning model
- Displays the predicted breed name
- Simple and intuitive interface using Streamlit
### Requirements
- To run this project locally, you'll need the following dependencies:

  - Python 3.x
  - TensorFlow
  - Streamlit
  - NumPy
  - Pillow
- You can install these dependencies using pip by running the following command:
  -- pip install -r requirements.txt
- requirements.txt
  - streamlit
  - tensorflow
  - numpy
  - Pillow
### Model Description
- The core of the application is a Convolutional Neural Network (CNN) that has been trained to classify images of dogs into 70 different breeds. The model was trained using a large dataset of labeled dog images.

### Model Details
- Model Type: Convolutional Neural Network (CNN)
- Library: TensorFlow/Keras
- Input Size: 224x224 pixels RGB images
- Output: Predicted breed of the dog from 70 classes
The trained model (dogclassification.h5) is loaded into the app to perform inference.

### App Interface
- Upload Image: Click the "Upload a dog image" button to select and upload a dog image from your local machine.
- Classify Button: After uploading the image, click the "Classify" button to predict the breed of the dog.
- Result: The predicted breed will be displayed below the image.
Example Usage
1. Upload an Image
Click the "Upload a dog image" button to upload your image.

2. Classification
Click the "Classify" button to get the prediction of the breed.

3. Output
The app will show the predicted breed below the image.

### Acknowledgements
- The model was trained on a dataset of labeled dog images.
Special thanks to Streamlit and TensorFlow for providing the frameworks to build and deploy this app.
