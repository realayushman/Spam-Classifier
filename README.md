# Spam-Classifier
📩 Spam Messages Classifier using Naive Bayes Overview: This project is a Spam Message Classifier built using the Naive Bayes algorithm, a probabilistic machine learning model well-suited for text classification tasks. The system analyzes SMS messages and classifies them as spam or ham (not spam) based on their content.
📩 Spam Messages Classifier using Naive Bayes
Overview:
This project is a Spam Message Classifier built using the Naive Bayes algorithm, a probabilistic machine learning model well-suited for text classification tasks. The system analyzes SMS messages and classifies them as spam or ham (not spam) based on their content.

Key Features:

🧹 Text Preprocessing: Cleans the SMS data using tokenization, stopword removal, and stemming with NLTK.

📐 Feature Extraction: Converts text into numerical features using TF-IDF Vectorization or Bag-of-Words.

🧠 Naive Bayes Classifier: Trains a Multinomial Naive Bayes model on labeled data to detect spam with high accuracy.

📊 Evaluation: Uses metrics like accuracy, precision, recall, and F1-score to evaluate performance.

🌐 Streamlit Web App (Optional): Provides an easy-to-use UI for real-time message classification.

Tech Stack:

Python

NLTK for NLP

Scikit-learn for machine learning

Pandas and NumPy for data handling

Streamlit for deploying the app (optional)

Use Cases:

Blocking unwanted marketing messages

Building safer messaging platforms

Email or chat spam detection

