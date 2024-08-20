import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Loads and merges messages and categories datasets from specified file paths.

    Args:
        messages_filepath (str): File path to the messages CSV file.
        categories_filepath (str): File path to the categories CSV file.

    Returns:
        pd.DataFrame: Merged DataFrame containing both messages and categories data.
    """
    
    messages_df = pd.read_csv(messages_filepath)
    categories_df = pd.read_csv(categories_filepath)
    merged_df = messages_df.merge(categories_df)
    
    return merged_df


def clean_data(df):
    """
    Cleans the merged DataFrame by splitting categories into separate columns, 
    converting values to numeric, removing duplicates, and handling missing values.

    Args:
        df (pd.DataFrame): DataFrame containing merged messages and categories data.

    Returns:
        pd.DataFrame: Cleaned DataFrame with categories expanded into individual columns.
    """
    
    categories = df['categories'].str.split(';', expand=True)
    row = list(categories.loc[0])
    category_names = [name[:-2] for name in row]
    categories.columns = category_names
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = [value[len(value) - 1] for value in list(categories[column])]
        # convert column from string to numeric
        categories[column] = [int(value) for value in list(categories[column])]
    df = df.drop(['categories'], axis=1)
    df = pd.concat([df, categories], axis=1)
    df = df[~df.duplicated(subset=None, keep='first')]
    df.dropna(subset=list(categories.columns), inplace=True)
    df['related'] = df['related'].replace(2, 1)
    
    return df


def save_data(df, database_filename):
    """
    Saves the cleaned DataFrame to an SQLite database.

    Args:
        df (pd.DataFrame): The DataFrame to be saved.
        database_filename (str): The filename for the SQLite database.

    Returns:
        None
    """
    
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('DisasterResponseData', engine, index=False, if_exists='replace')


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()