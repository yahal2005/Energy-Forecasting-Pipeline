import os

# PATHS
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "energy_data_set.csv")

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
