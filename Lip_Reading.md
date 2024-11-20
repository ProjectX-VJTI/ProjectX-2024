<h1 align="center"> Lip Reading 💬</h1>


<p align="center">
  <img src="https://github.com/user-attachments/assets/9052f2a1-c027-401e-a79f-e3835189365b" alt="read-my-lips" width="350" />
</p>


<div align="center">
  <strong>The goal is to convert visual information from lip movements into text. This involves recognizing and interpreting the movements of a speaker's lips to accurately transcribe spoken words.</strong>
</div>

<br />

## 📑 Table of Contents

- [About the Project](#-about-the-project)
- [Results](#-results)
- [Tech Stack](#️-tech-stack)
- [File Structure](#-file-structure)
- [Dataset](#-dataset-miracl-vc1)
- [Model Architecture](#-model-architecture)
- [Installation and Setup](#-installation-and-setup)
- [Future Scope](#-future-scope)
- [Acknowledgements](#-acknowledgement)
- [Contributors](#-contributors)

## 📘 About the Project

This project focuses on developing a sophisticated lip-reading system that interprets spoken words from sequences of images. Using Haar Cascade classifiers for face extraction and dlib’s facial landmark detection for lip extraction, we effectively preprocess the data. A train-test split ensures robust model evaluation. The core of the project is a hybrid model combining 3D CNNs, which capture spatial features, and LSTMs, which understand temporal dynamics. Extensive hyperparameter tuning enhances the model’s accuracy. The system has been tested on online videos for accuracy and reliability and includes a live detection feature to showcase real-time capabilities.

## 📊 Results

### Confusion Matrix
<table>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/713915fc-8c9d-4066-a285-da73781f3a52" alt="Image 1" width="200"></td>
    <td><img src="https://github.com/user-attachments/assets/1588f2ef-b0d3-4989-9e7a-5011e588de35" alt="Image 2" width="200"></td>
  </tr>
  <tr>
    <td align="center">For Words</td>
    <td align="center">For Phrases</td>
  </tr>
</table>

### Accuracy

<img src="https://github.com/user-attachments/assets/084ae834-7398-48e2-b3db-6bd9f5e279cb" alt="read-my-lips" width="250" />

### Online Testing

<table>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/c543b5a3-0eff-45a2-aba2-78fa44d6427f" alt="choose_american" width="200"></td>
    <td><img src="https://github.com/user-attachments/assets/fc72d19f-ca21-428b-8d49-1d1402e7e63b" alt="arrow" width="50">
    <td><img src="https://github.com/user-attachments/assets/0393c0f0-9890-48a4-a308-d6d85eca9cdd" alt="second_image" width="200"></td>
  </tr>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/0d2bb28d-5bf6-43d6-a3b6-2990621c493b" alt="choose_american" width="200"></td>
    <td><img src="https://github.com/user-attachments/assets/fc72d19f-ca21-428b-8d49-1d1402e7e63b" alt="arrow" width="50">
    <td><img src="https://github.com/user-attachments/assets/8249aa7f-336f-4b94-9324-65609dfff06d" alt="second_image" width="200"></td>
  </tr>
</table>


## ⚙️ Tech Stack

| **Category**                | **Technologies**                                                                                       |
|-----------------------------|----------------------------------------------------------------------------------------------------|
| **Programming Languages**   | [![Python](https://img.shields.io/badge/python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)              |
| **Frameworks**              | [![TensorFlow](https://img.shields.io/badge/tensorflow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/) [![Keras](https://img.shields.io/badge/keras-D00000?style=for-the-badge&logo=keras&logoColor=white)](https://keras.io/) |
| **Libraries**               | [![OpenCV](https://img.shields.io/badge/opencv-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/) [![NumPy](https://img.shields.io/badge/numpy-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)             |
| **Deep Learning Models**    | [![LSTM](https://img.shields.io/badge/LSTM-563D7C?style=for-the-badge&logo=lstm&logoColor=white)](https://en.wikipedia.org/wiki/Long_short-term_memory) [![CNN](https://img.shields.io/badge/CNN-0A192E?style=for-the-badge&logo=cnn&logoColor=white)](https://www.geeksforgeeks.org/convolutional-neural-network-cnn-in-machine-learning/) |
| **Dataset**                 | [![MIRACL-VC1](https://img.shields.io/badge/MIRACL--VC1-4D2A4E?style=for-the-badge&logo=dataset&logoColor=white)](https://paperswithcode.com/dataset/miracl-vc1)                                                                            |
| **Tools**                   | [![Git](https://img.shields.io/badge/git-F05032?style=for-the-badge&logo=git&logoColor=white)](https://git-scm.com/) [![Google Colab](https://img.shields.io/badge/google%20colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com/)                            |
| **Visualization & Analysis**| [![Matplotlib](https://img.shields.io/badge/matplotlib-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://matplotlib.org/) [![Seaborn](https://img.shields.io/badge/seaborn-013243?style=for-the-badge&logo=python&logoColor=white)](https://seaborn.pydata.org/)                 |


## 📁 File Structure

    ├── Dataset Preprocessing
       ├── xml files
          ├── haarcascade_frontalface_default.xml
          ├── haarcascade_mcs_mouth.xml
          ├── shape_predictor_68_face_landmarks.dat
       ├── 01_Face_Extraction.ipynb
       ├── 02_Lip_Extraction.ipynb
       ├── 03_Train_Test_Split.ipynb
    ├── Hyperparameter Tuning
       ├── Grid Search.ipynb
       ├── Random Search.ipynb
    ├── Mini Projects
       ├── Cat_Dog_Classifier_CNN.ipynb
       ├── Human_Action_Recognition_LSTM.ipynb
       ├── Next_Word_Predictor_LSTM.ipynb
       ├── Video_Anomaly_Detection_CNN_LSTM.ipynb
    ├── Model Architecture
       ├── Saved Model
          ├── 3D_CNN_Bi-LSTM.h5
       ├── 3D_CNN.ipynb
       ├── 3D_CNN_Bi-LSTM.ipynb
       ├── 3D_CNN_From_Scratch.ipynb
       ├── 3D_CNN_LSTM.ipynb
       ├── Adam.ipynb
       ├── CategoricalCrossentropy.ipynb
       ├── Data_Augmentation.ipynb
       ├── Dropout.ipynb
       ├── EarlyStopping.ipynb
       ├── L1_Regularization.ipynb
       ├── L2_Regularization_1.ipynb
       ├── L2_Regularization_2.ipynb
       ├── RMSprop.ipynb
    ├── Model Evaluation
       ├── Accuracy.ipynb
       ├── Live_Detection.ipynb
       ├── Onlne_Testing.ipynb
       ├── Precision.ipynb
       ├── Recall.ipynb
    ├── Notes
       ├── LSTM
       ├── OpenCV
       ├── Om Mukherjee
       ├── Sourish Phate       
    ├── README.md

## 💾 Dataset: MIRACL-VC1

The **MIRACL-VC1** dataset is structured to facilitate research in visual speech recognition, particularly lip reading. Here's a breakdown of its structure and contents:

#### Data Composition:
- **Video Clips**: The dataset contains short video clips of multiple speakers reciting specific phrases. Each clip captures the upper body, focusing mainly on the face and mouth area.
- **Speakers**: It features several speakers from diverse backgrounds, which helps models generalize across different individuals and speaking styles.
- **Languages**: The dataset is typically in English, though speakers may vary in accents and pronunciations.
- **Phrases**: Each video clip corresponds to one of a predefined set of phrases, which are recited by the speakers. The phrases are usually short and may cover simple daily expressions or numbers.

#### Dataset Contains The Following Words and Phrases:

![image](https://github.com/user-attachments/assets/89a99eee-09b5-4a17-9f25-8e413ecb8ebf)

[Download the MIRACL-VC1 dataset on Kaggle](https://www.kaggle.com/datasets/apoorvwatsky/miraclvc1)


## 🤖 Model Architecture

![276662464-b1a8a17b-da29-4424-9e5c-b3f51dd07a27](https://github.com/user-attachments/assets/08cbf766-8553-43ac-a3b7-5c987bce50b8)

1. **3D Convolutional Neural Network (3D CNN)**:
   Several convolutional layers are used, each followed by activation functions and pooling layers to reduce dimensionality while preserving essential features.

2. **Reshape Layer**:
   The tensor dimensions are adjusted to flatten the spatial data into a format that the LSTM can process.

3. **Long Short-Term Memory (LSTM)**:
   One or more LSTM layers are employed to process the sequential data, enabling the model to retain information over time and improve prediction accuracy.

4. **Flatten Layer**:
   This flattens the data without altering its values, preparing it for the next stage.

5. **Dropout Layer**:
    A dropout rate is set (e.g., 0.5) to control the fraction of neurons dropped this prevents overfitting.

6. **Dense Layers**:
    One or more dense layers with activation functions (e.g., softmax for multi-class classification) are used to output the prediction probabilities.
   

By combining these components, the model effectively learns to interpret lip movements, translating them into accurate predictions of spoken words.

## 🛠️ Installation and Setup

Follow these steps to set up the project environment and install necessary dependencies.

### Prerequisites
Ensure you have the following software installed:
- [Python](https://www.python.org/downloads/)
- [pip](https://pip.pypa.io/en/stable/installation/)
- [Git](https://git-scm.com/)

### Clone the Repository
Clone the project repository from GitHub:
```
git clone https://github.com/sourishphate/Project-X-Lip-Reading.git
cd Project-X-Lip-Reading
```

### Install Dependencies
Install the required Python packages:
```
pip install -r requirements.txt
```
### Run the Application
Start the live detection application using the following command:
```
python '.\Model Evaluation\Live_detection.py
```
### Troubleshooting
If you encounter issues or want to suggest any improvements [raise an issue](https://github.com/sourishphate/Project-X-Lip-Reading/issues) on GitHub.


## 🌟 Future Scope

- **Multilingual Model**: Extend the current lip-reading model to support multiple languages, making it adaptable for a global audience and capable of handling diverse linguistic inputs.

- **User Interface Development**: Design a user-friendly interface that allows real-time interaction with the lip-reading model, improving accessibility and practical usability.

- **Sentence-Level Lip Reading**: Upgrade the model to read and interpret entire sentences, moving beyond word-level predictions to understand more complex speech patterns.

- **Large-Scale Model with Bigger Datasets**: Transition to a large-scale model by training with much larger datasets, which will boost the model’s ability to generalize across various lip movements, leading to greater accuracy.

## 📜 Acknowledgement

We would like to express our gratitude to all the tools and courses which helped in successful completion of this project.

**Research Papers**
- [https://cs229.stanford.edu/proj2019aut/data/assignment_308832_raw/26646023.pdf](https://cs229.stanford.edu/proj2019aut/data/assignment_308832_raw/26646023.pdf)
- [https://colah.github.io/posts/2015-08-Understanding-LSTMs/](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [https://towardsdatascience.com/automated-lip-reading-simplified-c01789469dd8](https://towardsdatascience.com/automated-lip-reading-simplified-c01789469dd8)

**Courses**
- [Andrew Ng's Deep Learning Specialization](https://www.coursera.org/specializations/deep-learning)
- [OpenCV Free Bootcamp](https://opencv.org/university/free-opencv-course/)

A special thanks to our project mentor [Veeransh Shah](https://github.com/Veeransh14) and to the entire [Project X](https://github.com/ProjectX-VJTI) community for unwavering support and guidance throughout this journey.

## 👥 Contributors

- [Sourish Phate](https://github.com/sourishphate)
- [Om Mukherjee](https://github.com/meekhumor)






