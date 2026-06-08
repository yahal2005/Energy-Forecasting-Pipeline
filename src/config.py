import os

# PATHS
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
RAW_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "energy_data_set.csv")
PROCESSED_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed")

DATETIME_COL = "date"
TARGET = "Appliances"

# FEATURE GROUPINGS
INDOOR_TEMP_FEATURES = [f"T{i}" for i in range(1, 10)]
INDOOR_HUMIDITY_FEATURES = [f"RH_{i}" for i in range(1, 10)]

WEATHER_FEATURES = [
    "T_out", "Press_mm_hg", "RH_out", 
    "Windspeed", "Visibility", "Tdewpoint"
]

SUSPECTED_NOISE = ["rv1", "rv2", "lights"]
