import sys
import pandas as pd
from sqlalchemy import create_engine
import re
import nltk
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import pickle


def load_data(database_filepath):
    """
    Loads data from an SQLite database, extracting messages and categories for model input.

    Args:
        database_filepath (str): File path to the SQLite database.

    Returns:
        tuple: A tuple containing:
            - X (numpy.ndarray): Array of messages (features).
            - y (numpy.ndarray): Array of category labels (targets).
            - category_names (pd.Index): Index of category names corresponding to the labels.
    """
    
    engine = create_engine(database_filepath)
    df = pd.read_sql_table('DisasterResponseData', engine)
    X = df['message'].values
    y = df.drop(['id', 'message', 'original', 'genre'], axis=1).values
    category_names = df.drop(['id', 'message', 'original', 'genre'], axis=1).columns
    
    return X, y, category_names

def tokenize(text):
    """
    Processes and tokenizes text by normalizing, removing stop words, 
    and lemmatizing tokens.

    Args:
        text (str): The input text to be tokenized.

    Returns:
        list: A list of cleaned tokens.
    """

    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    stop_words = stopwords.words("english")
    clean_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    
    return clean_tokens


def build_model():
    """
    Builds a machine learning pipeline with text processing and classification, 
    and performs hyperparameter tuning using GridSearchCV.

    Returns:
        GridSearchCV: A GridSearchCV object with a pipeline that includes
        text feature extraction and a multi-output classifier.
    """

    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
                ])),
        ])),
        ('clf', MultiOutputClassifier(AdaBoostClassifier()))
    ])
    
    parameters = {
        'clf__estimator__learning_rate': [0.5, 1],
        'clf__estimator__n_estimators': [10, 20]
    }
    
    cv = GridSearchCV(pipeline, param_grid=parameters, cv=2, n_jobs=-1, verbose=3)
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluates the trained model by predicting on the test data and 
    printing a classification report.

    Args:
        model: The trained model to be evaluated.
        X_test (numpy.ndarray or pd.DataFrame): Test features.
        Y_test (numpy.ndarray or pd.DataFrame): True labels for the test set.
        category_names (list of str): List of category names for the target labels.

    Returns:
        None
    """

    y_pred = model.predict(X_test)
    print(classification_report(Y_test, y_pred, target_names=category_names))


def save_model(model, model_filepath):
    """
    Saves the trained model to a file using pickle.

    Args:
        model: The trained model to be saved.
        model_filepath (str): The file path where the model will be saved.

    Returns:
        None
    """

    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data('sqlite:///' + database_filepath)
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