import sqlite3
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, RobustScaler
from config import DATA_PATH, MODEL_DIR

def load_data():
    conn=sqlite3.connect(str(DATA_PATH))
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

def prepare_and_scale(df):
    """Prepare features and scale them. Returns X_scaled, feature_names, scaler.
    
    Features include:
    - Original 5: invoice_quantity, invoice_dollars, total_quantity, total_dollars, average_receiving_delay
    - Added: days_po_to_invoice, payment_delay (temporal patterns in invoice processing)
    
    Using RobustScaler for heavy right-skew (skewness > 4 for quantity/dollar features).
    """
    features = ['invoice_quantity', 'invoice_dollars', 'total_quantity', 'total_dollars',
                'average_receiving_delay', 'days_po_to_invoice', 'payment_delay']
    X = df[features].fillna(0)
    
    # RobustScaler: uses median and IQR, robust to outliers in skewed data
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Save scaler
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(scaler, MODEL_DIR / "scaler.pkl")
    
    return X_scaled, features, scaler
