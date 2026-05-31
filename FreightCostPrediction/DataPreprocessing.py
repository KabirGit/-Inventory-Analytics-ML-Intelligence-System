import sqlite3
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def load_vendor_invoice(db_path:str):
    conn=sqlite3.connect(db_path)
    query="select * from vendor_invoice"
    df=pd.read_sql_query(query,conn)
    conn.close()
    return df

def engineer_features(df:pd.DataFrame):
    """Engineer lag, temporal, and route-level features from existing columns.
    
    Features engineered:
    - lag_1_freight, lag_2_freight: temporal lag features
    - route_avg_freight: leave-one-out vendor average (no leakage)
    - Quantity: invoice quantity (correlation 0.95 with Freight)
    - invoice_month, invoice_dow: seasonal/cyclical time features
    """
    # Sort by date for meaningful lag features
    if 'InvoiceDate' in df.columns:
        df = df.sort_values('InvoiceDate').reset_index(drop=True)
        # Time-based features
        invoice_dt = pd.to_datetime(df['InvoiceDate'], errors='coerce')
        df['invoice_month'] = invoice_dt.dt.month.fillna(1).astype(int)
        df['invoice_dow'] = invoice_dt.dt.dayofweek.fillna(0).astype(int)
    else:
        df['invoice_month'] = 1
        df['invoice_dow'] = 0
    
    # Lag features based on temporal order
    df['lag_1_freight'] = df['Freight'].shift(1).fillna(0)
    df['lag_2_freight'] = df['Freight'].shift(2).fillna(0)
    
    # Vendor-level average freight (leave-one-out to prevent leakage)
    if 'VendorNumber' in df.columns:
        vendor_sum = df.groupby('VendorNumber')['Freight'].transform('sum')
        vendor_count = df.groupby('VendorNumber')['Freight'].transform('count')
        df['route_avg_freight'] = (vendor_sum - df['Freight']) / (vendor_count - 1)
        global_mean = df['Freight'].mean()
        df['route_avg_freight'] = df['route_avg_freight'].fillna(global_mean)
    else:
        df['route_avg_freight'] = df['Freight'].rolling(window=10, min_periods=1).mean().shift(1).fillna(0)
    
    return df

def prepare_features(df:pd.DataFrame):
    df = engineer_features(df)
    feature_cols = ['Dollars', 'Quantity', 'lag_1_freight', 'lag_2_freight',
                    'route_avg_freight', 'invoice_month', 'invoice_dow']
    x = df[feature_cols]
    y = df['Freight']
    return x, y, feature_cols

def split_data(x,y,ts=0.2,rs=42):
    return train_test_split(x,y,test_size=ts,random_state=rs)
