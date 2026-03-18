import sqlite3
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.model_selection import train_test_split
import os

def load_data():
    conn=sqlite3.connect("C:\\Users\\Kabir\\Desktop\\my SALES ANALYSIS\\data\\inventory.db")
    query=("""
    with purchase_agg as(
    select 
    p.PONumber,
    count(distinct p.Brand) as total_brands,
    sum(p.Quantity) as total_quantity,sum(p.Dollars) as total_dollars,
    avg(julianday(p.ReceivingDate)-julianday(p.PODate)) as average_receiving_delay 
    from purchases p group by p.PONumber
    )
    select
    v.PONumber,
    v.Quantity as invoice_quantity,
    v.Dollars as invoice_dollars,
    v.Freight,
    (julianday(v.InvoiceDate)-julianday(v.PODate)) as days_po_to_invoice,
    (julianday(v.PayDate)-julianday(v.InvoiceDate)) as payment_delay,
    pa.total_brands,
    pa.total_quantity,
    pa.total_dollars,
    pa.average_receiving_delay
    from vendor_invoice v
    left join purchase_agg as pa
    on v.PONumber=pa.PONumber
    """)
    df=pd.read_sql_query(query,conn)
    conn.close()
    return df

def create_invoice_risk_label(row):
    if (abs(row["invoice_dollars"]-row["total_dollars"])>5):
        return 1
    if row["average_receiving_delay"]>10:
        return 1
    return 0

def apply_labels(df):
    df["flag_invoice"]=df.apply(create_invoice_risk_label,axis=1)
    return df

def split_data(df,features,target):
    X=df[features]
    y=df[target]
    return train_test_split(X,y,test_size=0.2)

def scale_features(xtrain,xtest,scaler_path):
    scaler=StandardScaler()
    xtrain_scaled=scaler.fit_transform(xtrain)
    xtest_scaled=scaler.transform(xtest)
    os.makedirs('models', exist_ok=True)
    joblib.dump(scaler,'models/scaler.pkl')
    return xtrain_scaled,xtest_scaled
