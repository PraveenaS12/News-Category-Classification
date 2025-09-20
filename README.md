# News Category Classification using NLP and Machine Learning

![News Classification Banner](https://user-images.githubusercontent.com/79269520/150174169-1ddc4349-3e3a-4395-8167-154a43b952a2.png)

## 📖 Overview

This project focuses on **multi-class text classification**, a core task in Natural Language Processing (NLP). The goal is to build and evaluate a machine learning model that can automatically categorize news articles into one of four distinct topics: **World, Sports, Business, or Sci/Tech**.

The model is trained on a large dataset of news headlines and descriptions. By leveraging text preprocessing techniques and TF-IDF vectorization, it learns to identify the linguistic patterns associated with each category. The final trained model can accurately predict the category of new, unseen news articles, making it a valuable tool for content tagging and organization.

---

## 📂 Repository Contents

*   **`News_category_classification.ipynb`**: The main Jupyter Notebook that contains the entire workflow for this project. This includes data loading, exploratory data analysis, text preprocessing, feature extraction, model training, and a detailed performance evaluation.

---

## 💾 Dataset

This project utilizes the **"AG News Classification Dataset"**, a widely-used benchmark dataset for text classification. It consists of over 120,000 training samples and 7,600 testing samples.

*   **Link:** [AG News Classification Dataset on Kaggle](https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset)

**Instructions:**
To run the `News_category_classification.ipynb` notebook, you must download the dataset from the link above. After unzipping, place the `train.csv` and `test.csv` files in the same root directory as the notebook.

---

## ✨ Project Workflow & Features

The project follows a systematic machine learning pipeline:

1.  **Data Loading & Exploration:** The `train.csv` and `test.csv` files are loaded into Pandas DataFrames. Initial analysis is performed to understand the distribution of news categories.
2.  **Text Preprocessing:** A custom function cleans the text data by:
    *   Converting text to lowercase.
    *   Removing stopwords (e.g., "the", "is", "in").
    *   Removing punctuation and special characters.
    *   Applying lemmatization to reduce words to their root form (e.g., "running" becomes "run").
3.  **Feature Extraction:** The cleaned text is converted into a numerical format using **TF-IDF (Term Frequency-Inverse Document Frequency)**. This technique creates a matrix where each row represents an article and each column represents a word, with the values indicating the word's importance.
4.  **Model Training:** Two different classifiers are trained and compared:
    *   **Multinomial Naive Bayes:** A probabilistic classifier well-suited for text data.
    *   **Logistic Regression:** A robust and highly effective linear model for classification.
5.  **Performance Evaluation:** The models are evaluated on the unseen test data using key metrics, including **Accuracy Score**, a **Classification Report** (with precision, recall, and F1-score), and a **Confusion Matrix**.

---

## 🛠️ Tech Stack & Libraries

*   **Language:** Python
*   **Core Libraries:**
    *   **Pandas & NumPy:** For data manipulation and numerical operations.
    *   **NLTK:** For the entire text preprocessing pipeline.
    *   **Scikit-learn:** For data splitting, TF-IDF vectorization, model training, and evaluation.
    *   **Matplotlib & Seaborn:** For creating visualizations like the confusion matrix.

---

## ⚙️ How to Run This Project

To replicate this project on your local machine, please follow these steps:

1.  **Clone the repository:**
    ```
    git clone https://github.com/your-username/your-repo-name.git
    ```
2.  **Navigate to the project directory:**
    ```
    cd your-repo-name
    ```
3.  **Install the required libraries:**
    *(It is recommended to create a virtual environment first)*
    ```
    pip install pandas numpy nltk scikit-learn matplotlib seaborn
    ```
4.  **Download NLTK resources:**
    Run the following commands in a Python interpreter:
    ```
    import nltk
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('wordnet')
    ```
5.  **Download the dataset** from the Kaggle link provided above and place `train.csv` and `test.csv` in the project folder.

6.  **Launch Jupyter Notebook** and open the `News_category_classification.ipynb` file to run the code cells.
    ```
    jupyter notebook News_category_classification.ipynb
    ```

---

## 📊 Results & Evaluation

The Logistic Regression model demonstrated superior performance in this multi-class classification task.

*   **Final Model Accuracy:** **88.92%**



```
