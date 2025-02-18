import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    
    """
    Load datasets
    
    Args:
        messages_filepath -> Path to the CSV file containing messages
        categories_filepath -> Path to the CSV file containing categories
    Returns:
        df -> a combined data frame containing messages and categories dataframes
    """
    # load datasets
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    # merge datasets
    df = messages.merge(categories, on='id')
    
    return df


def clean_data(df):
    
    """
     Clean dataset by Splitting categories into separate category columns, 
     converting category values to just numbers 0 or 1
     and removing duplicates.

    Args:
    Combined dataframe


    Returns:
    cleaned dataframe

    """
        
    # create a dataframe of the 36 individual category columns
    categories = df.categories.str.split(';', expand=True)
    # extract a list of new column names for categories
    row = categories.iloc[0]
    category_colnames = [name.split('-')[0] for name in row]
    # rename the columns of `categories`
    categories.columns = category_colnames
    
    # Convert category values to just numbers 0 or 1.
    for column in categories:
        categories[column] = categories[column].astype('str').str[-1]
        categories[column] = categories[column].astype('int')
        
    # Replace categories column in df with new category columns
    df.drop(columns='categories', inplace=True)
    df = pd.concat([df, categories], axis=1)
    # Ensure related column is 0 and 1 values only
    df=df[df['related']!=2]
    
    # Remove duplicates
    df.drop_duplicates(inplace=True)
    
    return df

def save_data(df, database_filename):
    
    """
     Save the clean dataset into an sqlite database

    Args:
    cleaned dataframe
    database_filename -> Path to SQLite destination database
    
    """
    engine = create_engine('sqlite:///' + database_filename) 
    df.to_sql('disaster_response', engine, index=False, if_exists='replace')


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
