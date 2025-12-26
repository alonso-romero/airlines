import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
import os
import sys

# Load the data
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

# main EDA function
def eda_pipeline():
    """
    Perform EDA analysis of the dataset
    """

    print("\n[EDA] Loading data...")
    df_path = DATA / "airlines.csv"
    df = pd.read_csv(df_path)
    print("[EDA] Data loaded successfully!")

    print("\n[EDA] Display the first five rows of the dataset")
    print("="*80)
    print(df.head())

    print("\n[EDA] Display the data types of the columns")
    print("="*80)
    print(df.dtypes)

    print("\n[EDA] Display the column names")
    print("="*80)
    print(df.columns)

    print("\n[EDA] Renaming columns...")
    df = df.rename(columns={
        'Statistics.# of Delays.Carrier': 'Delays.Carrier',
        'Statistics.# of Delays.Late Aircraft': 'Delays.Late',
        'Statistics.# of Delays.National Aviation System': 'Delays.NAS', 
        'Statistics.# of Delays.Security': 'Delays:Security', 
        'Statistics.# of Delays.Weather': 'Delays.Weather', 
        'Statistics.Carriers.Names': 'Carriers.Names', 
        'Statistics.Carriers.Total': 'Carriers.Total', 
        'Statistics.Flights.Cancelled': 'Flights.Cancelled', 
        'Statistics.Flights.Delayed': 'Flights.Delayed', 
        'Statistics.Flights.Diverted': 'Flights.Diverted', 
        'Statistics.Flights.On Time': 'Flights.On_Time', 
        'Statistics.Flights.Total': 'Flights.Total', 
        'Statistics.Minutes Delayed.Carrier': 'Min_Delay.Carrier', 
        'Statistics.Minutes Delayed.Late Aircraft': 'Min_Delay.Late', 
        'Statistics.Minutes Delayed.National Aviation System': 'Min_Delay.NAS', 
        'Statistics.Minutes Delayed.Security': 'Min_Delay.Security', 
        'Statistics.Minutes Delayed.Total': 'Min_Delay.Total', 
        'Statistics.Minutes Delayed.Weather': 'Min_Delay.Weather'
    })
    print("[EDA] Columns renamed!")

if __name__ == "__main__":
    eda_pipeline()