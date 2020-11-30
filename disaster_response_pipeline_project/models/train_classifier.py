import sys
# import libraries
import pandas as pd
import pickle
import nltk
import numpy as np
import re

nltk.download(['punkt', 'wordnet'])
nltk.download('stopwords')
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from sqlalchemy import create_engine
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import precision_recall_fscore_support, classification_report
from sklearn.grid_search import GridSearchCV
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split


def load_data(database_filepath):
        
    """
    Load datasets
    
    Args:
        database_filepath -> Path to the database file
    Returns:
        X-> Feature which is messages
        y-> 36 category
        category_names
    """
    # load data from database
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('disaster_response',engine)
    X = df['message']
    y = df[df.columns[4:]]
    category_names = y.columns
    return X, y, category_names 


def tokenize(text):
            
    """
    Text processing
    
    Args:
        text
    Returns:
        cleaned processed text
    """
    # normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # tokenize text
    tokens = word_tokenize(text)
    
    # lemmatize and remove stop words
    lemmatizer = WordNetLemmatizer()
    clean_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stopwords.words("english")]
    
    return clean_tokens
    


def build_model():
            
    """
    Build a machine learning pipeline
    
    Returns:
           tuned model  
    """
    pipeline = Pipeline([('vect', CountVectorizer(tokenizer=tokenize)),
                     ('tfidf', TfidfTransformer()),
                     ('clf', MultiOutputClassifier(RandomForestClassifier()))])
    
    parameters = {'tfidf__use_idf': (True, False),
              'clf__estimator__min_samples_split':[2, 4, 6]
             }

    cv = GridSearchCV(pipeline, parameters)
    
    return cv
    


def evaluate_model(model, X_test, Y_test, category_names):
    """
    return a classification report for model evaluation
    """
    # train classifier
    cv.fit(X_train, y_train)
    # predict on test data
    predicted = cv.predict(X_test)
    
    # print classification report for each category
    for i, col in enumerate(category_names):
        print('Category: {}\n'.format(col))
        print('Accuracy: {}\n'.format((y_test[col] == predicted[:,i]).mean()))
        print(classification_report(y_test[col], predicted[:,i]))
    


def save_model(model, model_filepath):
    """
    save the model to a pickle file
    """
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
