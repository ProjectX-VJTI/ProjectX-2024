# SmartMailGuard
# Table of Contents

- [About the Project](#about-the-project)
	- [Aim](#aim)
	- [Description](#description)
	- [Tech Stack](#tech-stack)
- [File Structure](#file-structure)
- [Requirements](#requirements)
- [Contributors](#contributors)
- [Mentors](#mentors)
- [Resources and Acknowledgements](#resources-and-acknowledgements)

# About the Project
## Aim

The objective of this project is to develop an intelligent email classification system using machine learning and deep learning models.

## Description

![](https://lh7-rt.googleusercontent.com/slidesz/AGV_vUdxr1hA556P3LyHXsHQAl0342btIbEHKfXUm5Ubr-SoGkIzortajR-8X9fCmn4aj1TznMczbVZ_XSVMs2elTKZQPzK67FTMjjST7XrsP10E9jGOtIRhrnO4At1C5Zgr443GIqhYhcuhYD0aqGTjmqE6-K12QZFT=s2048?key=9e3EFz_MWWXAqOMHKU1kTg)

SmartMailGuard is a system designed to categorize emails using Naïve Bayes, LSTM, and other Transformer architectures.

Using these different models and algorithms we can compare and grade their effectiveness on datasets of varying sizes and on the type of classification: Binary (Spam/Not-Spam) or Multiclass.

## Tech Stack

1. [Python](https://www.python.org/)
2. [NumPy](https://numpy.org/)
3. [PyTorch](https://pytorch.org/)
4. [TensorFlow](https://www.tensorflow.org/)
5. [Pandas](https://pandas.pydata.org/)
6. [HuggingFace](https://huggingface.co/)

## Models and Accuracies

83k Dataset Link(For Binary Classification): [Kaggle](https://www.kaggle.com/datasets/amalverma27/email-classification-dataset)<br>
3k Dataset Link(For Multiclass Classification): [Kaggle](https://www.kaggle.com/datasets/kevinzb56/final-combined)<br>
Dataset for AutoLabeler: [Kaggle](https://www.kaggle.com/datasets/amalverma27/multiclass-mail)

### 1. Naïve Bayes

#### 1.1. Without N-gram Optimization

- **Train**: ![](https://lh7-rt.googleusercontent.com/slidesz/AGV_vUdJZsgFPekxGzclL-KIvdHIZosz1ktyQjjNMKNg777uFzh_wA0avgpLH-CLKlF-qx15qZp7Tsdt9ApxKaWxELtXUlI7WVCH3hLNMpuinMAOAL-g7IAvilD3-fBNl2sHhVxHkleDXTO-IbD2hO-6lNO0yvPa7Ho=s2048?key=9e3EFz_MWWXAqOMHKU1kTg)
- **Test**: ![](https://lh7-rt.googleusercontent.com/slidesz/AGV_vUewwS3U92Kx_kuNhMcr-E0ks6-H5ZgNnX_E9c5k-0N8JMQF2PmYi0gKo9_RFqUhdIGg9HT2uwYT-J9AM-WUSEVqxo8eE4kAP7sXKkZHWyI_0bXC7F3TVtreOhzQy7lNEK4t-b2zWunkwwePC1vNAwH2C9W6GxZU=s2048?key=9e3EFz_MWWXAqOMHKU1kTg)
#### 1.2. With N-gram Optimization

- **Train**: ![](https://lh7-rt.googleusercontent.com/slidesz/AGV_vUcZj2F_xHOylMKHed0kUZ3_mXcc8j84lLTofpKX7TWn3P55ClWK6vcJbw81Fe2tvDd87CPCQG2ViJBtIN9U0gSBzfzxF73ebgvksCJ4mJUAzqjxHM3PIc41yqV9AxhRCDxAVQJ9_9-D765hFz4uj0BXV6vBNSWb=s2048?key=9e3EFz_MWWXAqOMHKU1kTg)
- **Test**: ![](https://lh7-rt.googleusercontent.com/slidesz/AGV_vUfzEcodN5Dxl2MmoyDF1J0viqglHNCfcA6Ki7-eX4354iOvOYxCYheNc_cIu9ANzciuTrMW176Indad_exFuov4tCTUhasX16mDMfGubziNiR1-AL4d1sSNg0sEN2OiTEI2LYy78ngOOeFIkNOZtjuivEk-9-lz=s2048?key=9e3EFz_MWWXAqOMHKU1kTg)

##### Toy Examples:

![](https://lh7-rt.googleusercontent.com/slidesz/AGV_vUecNG_z___gdxEkhi3zjhoyVNu-RPk4BHhgGpS8NDTWkp0CR8ymTMd_HMJyIZ-uW1IObh95t_j1NtGA_ofQLewH7MdC8yUaJ5PKDCFqVEM-iJcBZUsJ62qSV84i1PzXIkkGHLE76EOPc30Xp0sh3PYgdttNJtIj=s2048?key=9e3EFz_MWWXAqOMHKU1kTg)
### 2. Recurrent Neural Network (RNN)

- **Train**: ![](https://lh7-rt.googleusercontent.com/slidesz/AGV_vUd1VROZhMZflLsmdul4tne829ZKXm_E0g92Ohgyyd-Mqmcxs02xr0tRTiefY5HGmsdMEI0CilxGpRJJgEbFyTqUW7UpfOfi2LgufUB983lgTYmb9Eu1Xq6_uD_uU5Jh9-lxYGaKXYKCa37NO3msMrmKKLeQlp8=s2048?key=9e3EFz_MWWXAqOMHKU1kTg)
- **Test**: ![](https://lh7-rt.googleusercontent.com/slidesz/AGV_vUfqLQyPFaFXPBNyUF42aazZ7GKG27JYX0XhDwhpSGkcl0qZB_lts8qtMnOsrop83KfjLs37EVeZBdDrdrhAHSP5nESaD2DtsVB13aupW5F5Re6WhboV3Es_mTM6I__grZESD1R4ySdVHSIYWfPZdkGWgfRhmti0=s2048?key=9e3EFz_MWWXAqOMHKU1kTg)

### 3. Long Short-Term Memory (LSTM)

- **Train**: ![](https://lh7-rt.googleusercontent.com/slidesz/AGV_vUeE-rKCJa0pprSSpuhfDreGS4xr1h5CBcfQUVmktL2Uuvc0uNPCUO2hXrspmqwtgFeetg7SGhyaSLK47GajNHCLIfgJkntffxEpPf7rnALo2O4tyE-Y7Pvup5Zndjl21UEnntvJ1vaRbxU7qpEnqNMNV9UnOM4=s2048?key=9e3EFz_MWWXAqOMHKU1kTg)
- **Test**: ![](https://lh7-rt.googleusercontent.com/slidesz/AGV_vUczVo2_-tK7D1vZxNk0-GU2LOgzaemV28FrGemAcHuIignBHP8Y6C6WxcUHla4wwez5CAwyy2nOLfWXWfxU192GiIYjbzrEm9yip-XqP27ZbVn47eYeJnn2Hnr3psS7Chm5oCIAekMmo70q0-RErRMQNmB_xHA=s2048?key=9e3EFz_MWWXAqOMHKU1kTg)

### 4. Multinomial Naïve Bayes

![](https://lh7-rt.googleusercontent.com/slidesz/AGV_vUdqOiwp4MVN967XGa3xecauK3u2iibMCjTQZ0r2kpjxNeZ1QjTTt_1hQe7uBA9TY2Z-FKsn1jGtBavAAjx1gzobLhIRa5BQEjd4MZDq8Fib0eJ-B9NVe83Vy6KPmcFO-yqOsdrzvv-FxekXPGQd6w1SJKixshok=s2048?key=9e3EFz_MWWXAqOMHKU1kTg)

### 5. Bidirectional Encoder Representations from Transformers (BERT)

#### 5.1. From Scratch

- **Train**: ![](https://lh7-rt.googleusercontent.com/slidesz/AGV_vUdUQswILD76uT4y2dwIqNj0xlujaT0nzWwfyz0HkWo0pJRwJk7ddmLCWYJVGiyPbgS_pCcKEhj0jQSg58VqehwE5Ey9vKGYAb3qLOcGCMURtpMERDoJS3-w_vnFqlmL_ZYLnnfu9W822fuZR2shoVcBaQMRSdo=s2048?key=9e3EFz_MWWXAqOMHKU1kTg)
- **Test**: ![](https://lh7-rt.googleusercontent.com/slidesz/AGV_vUcugxD2-coTTGziASOaz8EkKhJ3iFKZaOyu2PNt7e1NnF9MjiW8PYoJwQzJVR-gfhrCR0WQGXLPwFk-g4AqkEHEWMdpGZFJNGIqll5dFGFnZf1XFyGxXfpRLMuppcy121BMZNXaTkrMjDehQRj57I73PK7_09C3=s2048?key=9e3EFz_MWWXAqOMHKU1kTg)

#### 5.2. Implementation from a Pre-Trained Model

- **Train**: ![](https://lh7-rt.googleusercontent.com/slidesz/AGV_vUcMLFelDa1U0hftQRtV1WaFlgx4Cy5KGYraQ5yrR33J8CMCFzWZYjvzPXB1-I4oK7dtFStfv3iD2VJNm4ysOA8OI4N1-cP41JjY9MExx1cTIMM9VzlVzObHUTZ3_6EWhIEIfteecsShFHxm9pYVf6wU7k65h60=s2048?key=9e3EFz_MWWXAqOMHKU1kTg)
- **Test**: ![](https://lh7-rt.googleusercontent.com/slidesz/AGV_vUeZrEkNRXIpYV5BlE9CeQtBhWmX8S-VFUt2crLP8-80X0cYpJ-0M9oaVSrkyxIS7JzGLwzPOrgjdBmSGkzrJlETOCv83dNS5Mj5InPN5zZq23J-c_A4bblTQJM6FUb18y3VbFWJt_pLX-tuDRF1whmx88XIZ4M=s2048?key=9e3EFz_MWWXAqOMHKU1kTg)
##### Toy Examples

![](https://lh7-rt.googleusercontent.com/slidesz/AGV_vUe-K3fsFCyeBKbfnZI38UpNj0Ugy9P_aIn6fn---W-iKwXjnCbAJfIDMKSe7yD5FYxQgKiGzsMlc0e3GUhDaqzOaSmPzUS5VhvfuF6VKxWoKnjo__jTCPzAPN2X3CMmSVFWtgqHRGemJLvWLT69eyb7M5CRjdr7=s2048?key=9e3EFz_MWWXAqOMHKU1kTg)

#### 6. Support Vector Machine (SVM)

- **Train**: ![](https://lh7-rt.googleusercontent.com/slidesz/AGV_vUdQUZns37mw6oUetj_s_ARS4Pe1e5oiRLmJEgfzeZV9sdy32M8lmSefu1yMgPDywprWLEh--1CqR-2b645jaResMfKHA2CWzuMpWfTaoAf2LqMfiDqJyVCAY9epQN71B6qgMHNLGQqssFg_ALJMll51M1p07Cvj=s2048?key=9e3EFz_MWWXAqOMHKU1kTg)
- **Test**: ![](https://lh7-rt.googleusercontent.com/slidesz/AGV_vUcWBdrrR3LG3MLH_C7ldfJOEYrDyNB6326PEqDwnZVKSLoN6M3Beq05Ear_qaDVRY7XxbGLjtLL6GeN5hd26KA-DeVlY2bww9HwMnvD-YTSAbVA8O5eHHduXul2Fbv9EtJr3jLWq2HvzpnobZGnJRPNlJcuFIz7=s2048?key=9e3EFz_MWWXAqOMHKU1kTg)

#### 7. Decision Tree

![](https://lh7-rt.googleusercontent.com/slidesz/AGV_vUfs_BtKHlb2uoomNFeQAmzrmj1NDgK6yedHUZ4GFe4DqhGpz5EsJMa9kfxkBhR1sICfrJC3iPtrzPcD3yL0JbX7Ck0TVYORqHhd15EgyYMu2707RK3Cw6wd2XNvf1r9MTfdBvvHBN-M2CtTHGXL5UJBcQR2VTY3=s2048?key=9e3EFz_MWWXAqOMHKU1kTg)

#### 8. Random Forest Classififer

- **Train**: ![](https://lh7-rt.googleusercontent.com/slidesz/AGV_vUcKqZdOa9rJ7PFN2an3P-51iY-dF5DSJT30OD5M8UwCjpJv77g0AVJSOxkkVLQORY_xqOnSM5fK4CR2dDndDnCUTT9TDt3fxF2aTMrQTk9bDsbOCbmbH44ZXwr8LVGBDzZOH-CDYTmEzLOEFIpHrPXELIPkrR0=s2048?key=9e3EFz_MWWXAqOMHKU1kTg)
- **Test**: ![](https://lh7-rt.googleusercontent.com/slidesz/AGV_vUcH0KGa4L7s5YaQ1O3wmm0Lpzb8grOhBGTOZYLmnbGntjv5kL0eUamllL8vc1AtQ-8Vf86zELY4cijzfL6tGu3Ub_iPMp5pYgn7U-P73-tkW2XKKl_1eQKQRFQFadthvdZbj0IeTLvqAO829VXyURxnSEJtIehr=s2048?key=9e3EFz_MWWXAqOMHKU1kTg)
# File Structure
```
├── Binary Classification
│   ├── Naive_Bayes_Final.ipynb
│   ├── Naive_Bayes_enron_dataset.ipynb
│   ├── Naive_Bayes_sklearn.ipynb
│   ├── lstmemailclassification.ipynb
│   └── RNN_spam_not_spam.ipynb
├── Coursera Notes
│   ├── Course1
│   ├── Course2
│   ├── Course5
├── Multi Intent Classification
│   ├── Decision Tree
|   │   ├── decision-tree-grid-search.ipynb
|   │   ├── decision-tree.ipynb
│   ├── Random Forest Classifier
|   │   ├── RandomForestClassifier-grid_search.ipynb
|   │   ├── RandomForestClassifier.ipynb
│   ├── Support Vector Machine
|   │   ├── SVM_grid_search.ipynb
|   │   ├── SVM_multiclass_classifier.ipynb
|   ├── AutoLabeler.ipynb
│   ├── Multiclass.ipynb
│   ├── multiclass-bert-Finaldataset.ipynb
│   ├── multiclass-bert-Finaldataset-from-scratch.ipynb
│   └── multinomial_combined.ipynb
├── SmartMailGuard Report
│   ├── SmartMailGuard Report.pdf
└── README.md 
```

# Requirements

- Install [Python 3.1](https://www.python.org/downloads/).
- Install [Pip](https://pip.pypa.io/en/stable/installation/) and verify its installation using the following terminal command:

```bash
pip --version
```

- **Optional**: Install [Jupyter](https://jupyter.org/install) using the following command:

```bash
pip install jupyter lab
```

Alternatively, [Google Colaboratory](https://colab.research.google.com/) and [Kaggle](https://www.kaggle.com/) can also be used to run the notebooks (with some RAM limitations).

- Run the following command to install all the dependencies:

```
pip install pandas pytorch scikit-learn tensorflow transformers
```

- Clone the repository:

```bash
git clone https://github.com/aitwehrrg/SmartMailGuard.git
```

- Run any of the models (`.ipynb`) as Jupyter notebooks.
# Contributors

1. [Amal Verma](https://github.com/Amal-Verma)
2. [Kevin Shah](https://github.com/kevinzb56)
3. [Rupak R. Gupta](https://github.com/aitwehrrg)

# Mentors

1. [Druhi Phutane](https://github.com/druhi021204)
2. [Raya Chakravarty](https://github.com/Raya679)

# Acknowledgements and Resources

1. CoC and Project X for providing this opportunity.
2. Course on _[Deep Learning Specialization](https://www.coursera.org/specializations/deep-learning)_ by _[DeepLearning.AI](https://www.deeplearning.ai/)_
3. _[Long Short-Term Memory](https://deeplearning.cs.cmu.edu/F23/document/readings/LSTM.pdf)_
4. _[BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)_
5. _[Attention is all you need](https://arxiv.org/abs/1706.03762)_
6. [Kaggle datasets](https://www.kaggle.com/datasets)
7. [HuggingFace Transformer Models](https://huggingface.co/models)
