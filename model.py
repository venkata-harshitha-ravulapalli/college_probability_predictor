import pandas as pd
import math
import os
import re

# ------------------------------------
# NORMALIZATION FUNCTIONS
# ------------------------------------
def normalize_college(name):
    if pd.isna(name):
        return None
    x = str(name).upper()
    x = x.replace(".", "").replace(",", "")
    x = x.replace("ENGG", "ENGINEERING").replace("ENG", "ENGINEERING")
    x = re.sub(r"\s+", " ", x).strip()
    return x

def normalize_category(cat):
    if pd.isna(cat):
        return None
    c = str(cat).upper().strip().replace("_", "").replace("-", "")
    mapping = {
        "BCA": "BC-A",
        "BCB": "BC-B",
        "BCC": "BC-C",
        "BCD": "BC-D",
        "BCE": "BC-E",
        "EWS": "OC_EWS",
        "OCEWS": "OC_EWS",
        "OC": "OC",
        "SC": "SC",
        "ST": "ST",
    }
    return mapping.get(c, c)

# ------------------------------------
# LOAD CSVs SAFELY (NO MERGING)
# ------------------------------------
CSV_FILES = [
    ("sample_data/2019.csv", 2019),
    ("sample_data/2020.csv", 2020),
    ("sample_data/2022.csv", 2022),
    ("sample_data/2023.csv", 2023),
    ("sample_data/2024.csv", 2024),
]

dfs = []
for file, year in CSV_FILES:
    if os.path.exists(file) and os.path.getsize(file) > 0:
        try:
            df = pd.read_csv(file, on_bad_lines="skip")
            if not df.empty:
                # Add Year column if missing
                if "Year" not in df.columns:
                    df["Year"] = year
                dfs.append(df)
        except:
            pass

if not dfs:
    raise ValueError("No valid CSV files found")

data = pd.concat(dfs, ignore_index=True)

# ------------------------------------
# NORMALIZE DATA
# ------------------------------------
data["CollegeName"] = data["CollegeName"].apply(normalize_college)
data["Branch"] = data["Branch"].astype(str).str.upper().str.strip()
data["Gender"] = data["Gender"].astype(str).str.upper().str.strip()
data["Category"] = data["Category"].apply(normalize_category)

# ------------------------------------
# SIGMOID (ML-STYLE PROBABILITY)
# ------------------------------------
def sigmoid(diff):
    k = 0.002
    return 1 / (1 + math.exp(-k * diff))

# ------------------------------------
# PREDICTION FUNCTION
# ------------------------------------
def predict_probability(rank, college_code, branch, category, gender):
    category = normalize_category(category)
    branch = branch.upper().strip()
    gender = gender.upper().strip()

    subset = data[
        (data["CollegeCode"] == college_code) &
        (data["Branch"] == branch) &
        (data["Category"] == category) &
        (data["Gender"] == gender)
    ]

    if subset.empty:
        return None, None, None, None

    # Average cutoff
    avg_cutoff = int(subset["CutoffRank"].mean())

    # Latest cutoff (safe now)
    latest_cutoff = int(
        subset.sort_values("Year")["CutoffRank"].iloc[-1]
    )

    diff = avg_cutoff - rank
    prob = sigmoid(diff) * 100
    prob = round(min(prob, 97.0), 2)

    yearwise = subset[["Year", "CutoffRank"]].sort_values("Year")

    return prob, avg_cutoff, latest_cutoff, yearwise
