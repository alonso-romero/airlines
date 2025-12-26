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

    #print("\n[EDA] Display the column names")
    #print("="*80)
    #print(df.columns)

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

    print("\n[EDA] Displaying the updated column names")
    print("="*80)
    print(df.columns)

    print("\n[EDA] Dropping unnecessary columns...")
    df = df.drop(columns=[
        'Time.Label',
        'Time.Month Name'
    ])
    print("[EDA] Unnecessary columns dropped!")

    print("\n[EDA] Displaying unique year values")
    print("="*80)
    print(df['Time.Year'].unique())

    print("\n[EDA] Handling missing values...")
    missing_values = df.isnull().sum()
    if missing_values.sum() == 0:
        print("[EDA] No missing values found.")
    else:
        print("[EDA] Missing values found in the following columns:")
        print(missing_values[missing_values > 0])

    print("\n[EDA] Handling duplicate rows...")
    initial_shape = df.shape
    df = df.drop_duplicates()
    final_shape = df.shape
    if initial_shape[0] == final_shape[0]:
        print("[EDA] No duplicate rows found.")
    else:
        print(f"[EDA] Dropped {initial_shape[0] - final_shape[0]} duplicate rows.")

    print("\n[EDA] Outputting CSV files...")

    # Create a csv file with airport data to be used later

    # ============================== Airport Info CSV ==============================
    airport_df = df[['Airport.Code', 'Airport.Name']].drop_duplicates().reset_index(drop=True)
    coord_map = {
        'ATL': (33.6324, -84.4333),
        'BOS': (42.3656, -71.0096),
        'BWI': (39.1774, -76.6684),
        'CLT': (35.2170, -80.8431),
        'DCA': (38.8521, -77.0375),
        'DEN': (39.8647, -104.6737),
        'DFW': (32.8968, -97.0229),
        'DTW': (42.0078, -83.0458),
        'EWR': (40.6925, -74.1750),
        'FLL': (25.7907, -80.1373),
        'IAD': (38.9119, -77.4592),
        'IAH': (29.9844, -95.3621),
        'JFK': (40.6397, -73.7789),
        'LAS': (36.0800, -115.1522),
        'LAX': (33.9416, -118.2437),
        'LGA': (40.7772, -73.9442),
        'MCO': (28.4051, -81.3090),
        'MDW': (41.8781, -87.6349),
        'MIA': (25.7907, -80.2056),
        'MSP': (44.8821, -93.2420),
        'ORD': (41.9026, -87.6298),
        'PDX': (45.5865, -122.6780),
        'PHL': (39.8760, -75.1652),
        'PHX': (33.4484, -112.0740),
        'SAN': (32.7336, -117.1645),
        'SEA': (47.4490, -122.3320),
        'SFO': (37.6152, -122.4194),
        'SLC': (40.7899, -111.8609),
        'TPA': (27.9756, -82.5000),
    }

    hubs ={
        'American Hub': ['CLT', 'ORD', 'DFW', 'LAX', 'MIA', 'JFK', 'LGA', 'PHL', 'PHX', 'DCA'],
        'Delta Hub': ['ATL', 'BOS', 'DTW', 'LAX', 'MSP', 'JFK', 'LGA', 'SLC', 'SEA'],
        'United Hub': ['ORD', 'DEN', 'IAH', 'LAX', 'EWR', 'SFO', 'IAD'],
        'Southwest Hub': ['ATL', 'BWI', 'MDW', 'DEN', 'LAS', 'LAX', 'MCO', 'PHX', 'SAN', 'TPA'],
        'Alaska Hub': ['SEA', 'PDX', 'LAX', 'SFO', 'SAN'],
        'JetBlue Hub': ['BOS', 'JFK', 'FLL', 'MCO', 'LAX']
    }

    for col_name, codes in hubs.items():
        airport_df[col_name] = airport_df['Airport.Code'].isin(codes)
    
    airport_df['Latitude'] = airport_df['Airport.Code'].map(lambda x: coord_map.get(x, [None, None])[0])
    airport_df['Longitude'] = airport_df['Airport.Code'].map(lambda x: coord_map.get(x, [None, None])[1])

    airport_df.to_csv(DATA / "airport_info.csv", index=False)

    print("[EDA] CSV files outputted successfully!")

if __name__ == "__main__":
    eda_pipeline()