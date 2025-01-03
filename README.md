
# Email Spam Classification Project

## Overview
This project aims to classify email messages as either spam or ham using natural language processing (NLP) and machine learning techniques. It includes data cleaning, feature extraction, exploratory data analysis, and the implementation of multiple machine learning models.

---

## Features
- **Data Cleaning**: Removing unwanted columns, handling missing values, and deduplication.
- **Text Preprocessing**: Tokenization, stopword removal, stemming, and transformation using TF-IDF.
- **Visualization**: Correlation heatmaps and WordClouds for spam word analysis.
- **Modeling**: Implementation of Naive Bayes, Logistic Regression, SVM, Decision Trees, and ensemble techniques.
- **Evaluation**: Comparison of model performances using metrics such as accuracy and precision.

---

## Dataset
The dataset used (`spam.csv`) contains:
- **text**: The email message content.
- **target**: Classification label (0 for ham, 1 for spam).

---

## Steps

### 1. Data Cleaning
- Removed unnecessary columns and rows with missing values.
- Mapped labels to numerical values (`ham` to 0 and `spam` to 1).
- Removed duplicates.

### 2. Exploratory Data Analysis (EDA)
- Plotted the distribution of spam and ham emails.
- Created correlation heatmaps for text-derived features.

### 3. Text Preprocessing
- Lowercased text and removed non-alphanumeric characters.
- Removed stopwords and applied stemming to reduce dimensionality.

### 4. Feature Extraction
- Transformed text into numerical vectors using TF-IDF.

### 5. Model Training and Comparison
- Trained multiple classifiers including Naive Bayes, SVM, Logistic Regression, and ensemble methods.
- Evaluated models based on accuracy and precision scores.

### 6. Advanced Techniques
- Implemented Voting and Stacking Classifiers for better performance.

### 7. Model Saving
- Saved the trained model and vectorizer using `pickle` for reuse.

---

## Installation
### Prerequisites
Ensure the following Python libraries are installed:
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `nltk`
- `scikit-learn`
- `pickle`
- `wordcloud`

### Steps
1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/EmailSpamIdentifier.git
    ```
2. Navigate to the project directory:
    ```bash
    cd EmailSpamIdentifier
    ```
3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
4. Run the script:
    ```bash
    python app.py
    ```

---

## Results
- **Best Model**: Naive Bayes achieved a high accuracy and precision score.
- **Ensemble Techniques**: Voting and Stacking classifiers further improved performance.

---

## Future Work
- Expand dataset for better generalization.
- Experiment with deep learning approaches like RNNs.
- Deploy the model as a web application.

---

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request.

---

## License
This project is licensed under the MIT License.

---

## Contact
For any questions or issues, please contact:
- **Email**: vaghelameet765@gmail.com
- **GitHub**: https://github.com/Meetvaghela-code
