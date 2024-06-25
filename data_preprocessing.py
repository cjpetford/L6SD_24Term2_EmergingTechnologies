import pandas as pd

def drop_irrelevant_columns(df):
    irrelevant_columns = ['Client Name', 'Client e-mail', 'Profession', 'Education', 'Country', 'Net Worth']
    return df.drop(columns=irrelevant_columns, errors='ignore')
