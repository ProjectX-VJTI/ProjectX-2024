# GPT-Reimagined: KANs vs MLPs

This repository [GPT Reimagined: KANs vs MLPs](https://github.com/kavya-r30/GPT-Reimagined/) contains an implementation of a __Generative Pre-trained Transformer (GPT)__ model. The focus is to compare the performance and effectiveness of traditional __multilayer perceptron (MLP)__ layers and __Kolmogorov-Arnold Networks (KANs)__ in the architecture. 

__KANs__ are mathematical constructs based on the Kolmogorov-Arnold representation theorem, which suggests that any *multivariable continuous function* can be represented by a *composition of univariate functions and summation*. This unique approach can enable alternative network structures, potentially improving efficiency, expressiveness, or convergence rates for certain tasks.

## Table of Contents
- [Project Overview](#project-overview)
- [Installation](#installation)
- [Usage](#usage)
- [Experiment Details](#experiment-details)
- [Results](#results)
- [Directory Structure](#file-directory)
- [Contributors](#contributors)
- [Acknowledgement & Resources](#acknowledgement-and-Resources)

# Project Overview

## Description
In this project, we aim to explore the effectiveness of __Kolmogorov-Arnold Networks (KANs)__ as an alternative to traditional __Multi Layer Perceptrons (MLPs)__ for implementing Generative Pretrained Transformers (GPTs). GPTs are a class of machine learning models known for their ability to generate __natural language text__ and perform various natural language processing tasks. Traditionally, GPTs have been implemented using MLP architectures. However, KANs, a relatively new development, have shown promise in outperforming MLPs in certain tasks.

This project contributes to the ongoing research in machine learning architectures by providing empirical evidence on the efficacy of Kolmogorov-Arnold Networks as an alternative to traditional MLPs for implementing state-of-the-art language models like GPTs. The findings of this study can inform future developments in neural network architectures and guide the design of more efficient and effective models for natural language processing tasks.

## Model: KAN-GPT architecture
<div align="center">
    <img src="./assets/kan-gpt.png" alt="KAN-GPT architecture" />
</div>

## Tech Stack

| **Category**                | **Technologies**                                                                                       |
|-----------------------------|----------------------------------------------------------------------------------------------------|
| **Programming Languages**   | [![Python](https://img.shields.io/badge/python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)              |
| **Frameworks**              | [![PyTorch](https://img.shields.io/badge/pytorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/) |
| **Libraries**               | [![scipy](https://img.shields.io/badge/scipy-8CAAE6?style=for-the-badge&logo=scipy&logoColor=white)](https://scipy.org/) [![pandas](https://img.shields.io/badge/pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/) [![numpy](https://img.shields.io/badge/numpy-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/) [![tqdm](https://img.shields.io/badge/tqdm-4A4A4A?style=for-the-badge&logo=python&logoColor=white)](https://tqdm.github.io/) [![tiktoken](https://img.shields.io/badge/tiktoken-009688?style=for-the-badge&logo=python&logoColor=white)](https://github.com/openai/tiktoken) |
| **Datasets**                | [![TinyShakespeare](https://img.shields.io/badge/TinyShakespeare-4D2A4E?style=for-the-badge&logo=dataset&logoColor=white)](https://www.kaggle.com/datasets/harvardnlp/tiny-shakespeare) [![WikiText-2](https://img.shields.io/badge/WikiText--2-4D2A4E?style=for-the-badge&logo=dataset&logoColor=white)](https://huggingface.co/datasets/wikitext) |
| **Tools**                   | [![Git](https://img.shields.io/badge/git-F05032?style=for-the-badge&logo=git&logoColor=white)](https://git-scm.com/) [![Google Colab](https://img.shields.io/badge/google%20colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com/) [![Kaggle](https://img.shields.io/badge/kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/) |
| **Visualization & Analysis**| [![Matplotlib](https://img.shields.io/badge/matplotlib-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://matplotlib.org/) [![TensorBoard](https://img.shields.io/badge/tensorboard-FF6F00?style=for-the-badge&logo=tensorboard&logoColor=white)](https://www.tensorflow.org/tensorboard) |

## Objectives

- Implement GPT using the traditional MLP approach.
- Implement GPT using Kolmogorov-Arnold Networks (KANs).
- Compare the performance of GPT implemented with MLPs and KANs across various metrics, including but not limited to:
  - Language generation quality
  - Training speed
  - Model size
  - Resource utilization
- Provide a proof of principle for the performances of MLP-based GPTs versus KAN-based GPTs.

## Other mini-projects
- Neural Network based on MNIST dataset
  - Using MLP
  - Using KAN
- Fashion classifier using CNN
- NameGPT
- Masked Language Model using encoder
- Language translation (English-French) model using transformers

# Installation

1. Clone the Repository: 
    ```bash
    git clone https://github.com/your-username/GPT-Reimagined.git
    cd GPT-Reimagined
    ```
2. Install all the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download the Tiny Shakespeare or WikiText-2 Dataset: This is handled in dataset_shakespeare.py, which automatically downloads and tokenizes the dataset if it's not present.

# Usage

1. Training the Model: Run the main script to train the models.
    ```bash
    python main.py 
    ```
    - Training details, including training and validation loss, are logged and saved for analysis in TensorBoard.

2. Generating Text: After training, you can generate text using the trained model:
    ```bash
    python generate.py
    ```
    - his will generate text based on a provided input prompt.
    - Customize generate.py with desired configurations such as max_new_tokens to control the length of generated text.

3. Troubleshooting:
If you encounter issues or want to suggest any improvements [raise an issue](https://github.com/kavya-r30/GPT-Reimagined/issues) on GitHub.

# Experiment Details

The goal is to evaluate the comparative performance between KANs and MLPs when used in transformer models. Key experimental configurations:

- **Block Size:** 64 (number of tokens processed in a single pass)
- **Batch Size:** 64
- **Learning Rate:** 2e-5
- **Training Epochs:** 6 ~ ( 9435 * 6 steps)
- **Loss Function:** Cross-entropy for next-token prediction
- **Evaluation Metric:** Validation loss and perplexity

## Logging and Model Saving

- Training progress is logged to TensorBoard.
- Model checkpoints are saved in the `models/` directory.

# Results

- **Text Generation Quality:** Generated text samples from both models reveal the qualitative differences in coherence and fluency in similar number of epochs and hyper-parameter.

| **Metrics KAN-GPT**              | **Tiny-Shakespeare** | **WikiText-2** |
|-------------------------|-------------|-------------|
| Tokens  | ~337k tokens  | ~3M tokens |
| Training Time (per epoch)   | 65 min     | 335 min       |
| Perplexity | 40.44       | 35.16       |
| Loss | 3.70       | 3.56       |
| Parameters (millions)       |  25.47M      ||


## Generated Results (KANs)
![kan-gpt-generated-text](./assets/kan-generated-text.png)

## Conclusion

- Concluded that KANs, despite potential, are hindered by **high computational requirements**, favoring MLPs in the
long run for Natural language text generation.

# File Directory`

<pre><code>
GPT-Reimagined/
├── data/                          # Dataset (tiny Shakespeare data used here)
│   ├── tinyshakespeare/
│   │   ├── input.txt              # Encoded input data
│   │   ├── train.bin              # Encoded training data
│   │   ├── val.bin                # Encoded validation data
├── models/                        # Directory for saving trained models
├── logs/                          # Training logs for TensorBoard
├── archive_logs/                  # Archive of zipped logs
├── main.py                        # Main script to initiate training
├── dataset_shakespeare.py         # Data processing and loading script
├── model_kan.py                   # Kolmogorov-Arnold Network (KAN) model
├── model_mlp.py                   # MLP-based GPT model
├── train.py                       # Training loop for the models
├── config.py                      # Configuration for hyperparameters and paths
├── generate.py                    # Script for generating text with the trained model
├── utils.py                       # Utility functions
├── requirements.txt               # Required dependencies
└── README.md                      # This README file
</code></pre>

# Contributors
- [Kavya Rambhia](https://github.com/kavya-r30)
- [Abhay Upadhyay](https://github.com/urabhay10)

# Acknowledgement and Resources
- Andrej Karpathy https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ 
- CMU Deep Learning lectures https://www.youtube.com/playlist?list=PLp-0K3kfddPzMmSaoGy5SqQsqhUeMC1ay
- Research Paper-Attention Is All You Need https://arxiv.org/abs/1706.03762
- Research Paper-Kolmogorov Arnold Networks https://arxiv.org/pdf/2404.19756
- Special thanks to our mentors [Param Thakkar](https://github.com/ParamThakkar123) and [Mayank Palan](https://github.com/MayankPalan2004/) and to the entire [Project X](https://github.com/ProjectX-VJTI) community for unwavering support and guidance throughout this journey.