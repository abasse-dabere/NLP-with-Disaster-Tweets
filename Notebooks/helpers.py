from matplotlib import pyplot as plt

import re
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import  stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score, precision_score, recall_score



def plot_history(history):
    # plot the training and validation accuracy
    plt.figure(figsize=(15,6))

    plt.subplot(1,2,1)
    plt.plot(history.history['accuracy'], label='Training accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # plot the training and validation loss
    plt.subplot(1,2,2)
    plt.plot(history.history['loss'], label='Training loss')
    plt.plot(history.history['val_loss'], label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

# Preprocessing the text
def preprocess_text(text):
    text_copy = text
    text = text.lower() # Lowercasing
    url_pattern = r'https?://\S+|www\.\S+'
    text = re.sub(url_pattern, '', text) # Remove URLs
    html_pattern = r'<.*?>'
    text = re.sub(html_pattern, '', text) # Remove HTML tags
    punctuation_pattern = r'[^\w\s]'
    text = re.sub(punctuation_pattern, '', text) # Remove punctuation
    number_pattern = r'[0-9]'
    text = re.sub(number_pattern, '', text) # Remove numbers
    stop_words = set(stopwords.words('english'))
    word_tokens = text.split()
    text = ' '.join([word for word in word_tokens if word not in stop_words]) # Remove stopwords
    lemmatizer = WordNetLemmatizer()
    word_tokens = text.split()
    text = ' '.join([lemmatizer.lemmatize(word) for word in word_tokens]) # Lemmatization
    if text == '':
        return text_copy
    return text

def print_report(y_train, y_train_pred, y_test, y_test_pred):
    print('Training set:')
    print('Accuracy:', accuracy_score(y_train, y_train_pred))
    print('F1 score:', f1_score(y_train, y_train_pred))
    print('Precision:', precision_score(y_train, y_train_pred))
    print('Recall:', recall_score(y_train, y_train_pred))
    print('\n')
    print('Testing set:')
    print('Accuracy:', accuracy_score(y_test, y_test_pred))
    print('F1 score:', f1_score(y_test, y_test_pred))
    print('Precision:', precision_score(y_test, y_test_pred))
    print('Recall:', recall_score(y_test, y_test_pred))