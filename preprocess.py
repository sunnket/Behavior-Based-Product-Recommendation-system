import pandas as pd

def load_data(path):
    df = pd.read_csv(path, parse_dates=['timestamp'])
    return df

def preprocess_sessions(df):
    df = df.sort_values(by=['user_id', 'session_id', 'timestamp'])
    session_dict = df.groupby(['session_id'])['product_id'].apply(list).reset_index()
    return session_dict
