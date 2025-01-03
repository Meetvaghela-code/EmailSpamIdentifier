import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score
import string
import pickle
from wordcloud import WordCloud

# Load dataset
data = pd.read_csv(r'Email_SpamIdentifiers/spam.csv', encoding='cp1252')

# Data Cleaning
data.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace=True)
data.rename(columns={'v1': 'target', 'v2': 'text'}, inplace=True)
data.dropna(subset=['text'], inplace=True)  # Remove rows with NaN text

data['target'] = data['target'].map({'ham': 0, 'spam': 1})
data.drop_duplicates(inplace=True)

# Exploratory Data Analysis (EDA)
plt.figure(figsize=(5, 5))
plt.pie(data['target'].value_counts(), labels=['Ham', 'Spam'], autopct='%0.2f', colors=['blue', 'red'])
plt.title("Distribution of Ham vs Spam")
plt.show()

# Feature Engineering
nltk.download('punkt')
nltk.download('stopwords')

data['num_char'] = data['text'].apply(len)
data['num_words'] = data['text'].apply(lambda x: len(nltk.word_tokenize(x)))
data['num_sentences'] = data['text'].apply(lambda x: len(nltk.sent_tokenize(x)))

# Correlation Heatmap
sns.heatmap(data[['num_char', 'num_words', 'num_sentences', 'target']].corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# Text Preprocessing
ps = PorterStemmer()

def transform_text(text):
    text = str(text).lower()  # Convert to lowercase
    text = nltk.word_tokenize(text)  # Tokenize

    # Remove non-alphanumeric characters
    text = [word for word in text if word.isalnum()]

    # Remove stopwords and punctuation
    text = [word for word in text if word not in stopwords.words('english') and word not in string.punctuation]

    # Stemming
    text = [ps.stem(word) for word in text]

    return " ".join(text)

data['transformed_text'] = data['text'].apply(transform_text)

# WordCloud Visualization
wc = WordCloud(width=500, height=500, min_font_size=10, background_color='white')
spam_wc = wc.generate(data[data['target'] == 1]['transformed_text'].str.cat(sep=" "))
plt.figure(figsize=(8, 8))
plt.imshow(spam_wc)
plt.title("Spam WordCloud")
plt.axis("off")
plt.show()

# Convert Text to Vectors
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(data['transformed_text']).toarray()
y = data['target'].values

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

# Model Training and Evaluation
mnb = MultinomialNB()
mnb.fit(X_train, y_train)

y_pred = mnb.predict(X_test)

print("MultinomialNB Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Precision Score:", precision_score(y_test, y_pred))

# Additional Models for Comparison
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier, BaggingClassifier,
                              ExtraTreesClassifier, GradientBoostingClassifier)
from xgboost import XGBClassifier

svc = SVC(kernel='sigmoid', gamma=1.0, probability=True)  # Enable probability=True for soft voting
knc = KNeighborsClassifier()
dtc = DecisionTreeClassifier(max_depth=5)
lrc = LogisticRegression(solver='liblinear', penalty='l1')
rfc = RandomForestClassifier(n_estimators=50, random_state=2)
abc = AdaBoostClassifier(n_estimators=50, random_state=2)
bc = BaggingClassifier(n_estimators=50, random_state=2)
etc = ExtraTreesClassifier(n_estimators=50, random_state=2)
gbdt = GradientBoostingClassifier(n_estimators=50, random_state=2)
xgb = XGBClassifier(n_estimators=50, random_state=2)

clfs = {
    'SVC': svc,
    'KN': knc,
    'DT': dtc,
    'LR': lrc,
    'RF': rfc,
    'AdaBoost': abc,
    'BgC': bc,
    'ETC': etc,
    'GBDT': gbdt,
    'xgb': xgb
}

def train_classifier(clf, X_train, y_train, X_test, y_test):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return accuracy_score(y_test, y_pred), precision_score(y_test, y_pred)

accuracy_scores = []
precision_scores = []

for name, clf in clfs.items():
    current_accuracy, current_precision = train_classifier(clf, X_train, y_train, X_test, y_test)
    print("For", name)
    print("Accuracy -", current_accuracy)
    print("Precision -", current_precision)
    accuracy_scores.append(current_accuracy)
    precision_scores.append(current_precision)

performance_df = pd.DataFrame({'Algorithm': clfs.keys(), 'Accuracy': accuracy_scores, 'Precision': precision_scores}).sort_values('Precision', ascending=False)
performance_df1 = pd.melt(performance_df, id_vars="Algorithm")

sns.catplot(x='Algorithm', y='value', hue='variable', data=performance_df1, kind='bar', height=5)
plt.ylim(0.5, 1.0)
plt.xticks(rotation='vertical')
plt.show()

# Voting Classifier
from sklearn.ensemble import VotingClassifier

voting = VotingClassifier(estimators=[('svm', svc), ('nb', mnb), ('et', etc)], voting='soft')
voting.fit(X_train, y_train)

y_pred = voting.predict(X_test)
print("Voting Classifier Accuracy:", accuracy_score(y_test, y_pred))
print("Voting Classifier Precision:", precision_score(y_test, y_pred))

# Stacking Classifier
from sklearn.ensemble import StackingClassifier

stacking = StackingClassifier(estimators=[('svm', svc), ('nb', mnb), ('et', etc)], final_estimator=RandomForestClassifier())
stacking.fit(X_train, y_train)

y_pred = stacking.predict(X_test)
print("Stacking Classifier Accuracy:", accuracy_score(y_test, y_pred))
print("Stacking Classifier Precision:", precision_score(y_test, y_pred))

# Save Models and Vectorizer
pickle.dump(tfidf, open('vectorizer.pkl', 'wb'))
pickle.dump(mnb, open('model.pkl', 'wb'))