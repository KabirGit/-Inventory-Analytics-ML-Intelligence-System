import sqlite3
import pandas as pd
from sklearn.model_selection import train_test_split

def load_vendor_invoice(db_path:str):
    conn=sqlite3.connect(db_path)
    query="select * from vendor_invoice"
    df=pd.read_sql_query(query,conn)
    conn.close()
    return df

def prepare_features(df:pd.DataFrame):
    x=df[['Dollars']] #target feature
    y=df['Freight']
    return x,y

def split_data(x,y,ts=0.2,rs=42):
    return train_test_split(x,y,test_size=ts,random_state=rs)

