""" 
This script analyzes spike trains of cells stimulated with Carbachol (CCh). 
It performs the following tasks:
1. Detrends the time series data of ISI and amplitude to cut out the deterministic contribution to the process. 
2. Calculates the moment relation between Interspike Interval (ISI) and its Standard Deviation (SD).
3. Computes statistical coefficients (e.g., Pearson correlations, regression slopes).
4. Investigates correlations between computed coefficients to uncover patterns.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Upload data
df = pd.read_csv(pycasig_df, sep=";", na_values="NaN", decimal=",")
df.dropna(inplace=True)
df = df.rename(columns={"time": "time (s)", "ISI": "ISI (s)", "amplitude": "A (DF/F0)"})

# Subtract the first time value for each group of 'ST' and 'stimulus'
df["time (s)"] = df.groupby(["ST", "stimulus"])["time (s)"].transform(
    lambda x: x - x.iloc[0]
)

# Apply detrending
df_tf = detrend_column_grouped(df, "time (s)", "ISI (s)", "tISI")
df_tf = detrend_column_grouped(df_tf, "time (s)", "A (DF/F0)", "tA")
df_tf = format_detrended_columns(df_tf)

# Ensure that 'ST' and 'stimulus' columns are integers
df_tf["ST"] = df_tf["ST"].astype(int)
df_tf["stimulus"] = df_tf["stimulus"].astype(int)

selected_columns = df[
    ["ST", "stimulus", "ISI (s)", "ISItf (s)", "A (DF/F0)", "Atf (DF/F0)", "rise"]
]
selected_columns = selected_columns.round(4)
selected_columns.index.name = "Index"

# Convert 'ST' and 'stimulus' to strings to ensure they are saved without '.0'
selected_columns["ST"] = selected_columns["ST"].apply(lambda x: str(int(x)))
selected_columns["stimulus"] = selected_columns["stimulus"].apply(lambda x: str(int(x)))
# Define save path
file = f"{save_path}_data.txt"
save_dataframe_with_header_and_format(selected_columns, file)

# Group and process the groups
grouped = df.groupby(["ST", "stimulus"])

for (st_value, cch_value), group in grouped:
    group.reset_index(drop=True, inplace=True)

    # Ensure 'stimulus' is an integer and convert it to a string to prevent .0
    group["stimulus"] = group["stimulus"].astype(int).apply(str)

    # Set index name to 'Index'
    group.index.name = "Index"

    # Select specific columns to save
    columns_to_save = [
        "stimulus",
        "ISI (s)",
        "ISItf (s)",
        "A (DF/F0)",
        "Atf (DF/F0)",
        "rise",
    ]
    df_filtered = group[columns_to_save]

    # Define the file path for saving
    file_path = f"{save_path}_ST{st_value}_CCh{cch_value}.txt"

    # Save the DataFrame to a file with proper formatting
    save_dataframe_with_header_and_format(df_filtered, file_path)

    # Optionally, print a confirmation for each saved file
    print(f"File saved for ST={st_value}, stimulus={cch_value} to {file_path}")


# Save PCC and p-values
correlation_df = (
    df.groupby(["ST", "stimulus"], group_keys=False)
    .apply(calculate_pearson)
    .reset_index()
)
file_path = save_path + "_PCC.txt"
save_dataframe_with_header_and_format(correlation_df, file_path)

# Coefficients of Variation
CVA_df = calculate_cv_amplitude(df)
CVISI_df = calculate_cv_ISI(df)

df_merged = pd.merge(CVISI_df, CVA_df, on=["ST", "stimulus"])
df_merged = pd.merge(df_merged, correlation_df, on=["ST", "stimulus"])

df_final = pd.merge(
    df_merged, df[["ST", "stimulus", "tA"]], on=["ST", "stimulus"], how="left"
)
df_final = df_final.drop_duplicates(subset=["ST", "stimulus"])

# Calculate tArs and regression coefficients
df_final["tArs"] = (df_final["Tav (s)"] * df_final["tA"]) / df_final["Aav (DF/F0)"]
regression_results = get_regression_coefficients(df)

df_final = pd.merge(
    df_final,
    regression_results[["ST", "stimulus", "Reg.Coeff.(DF/sF0)"]],
    on=["ST", "stimulus"],
    how="left",
)
df_final["Reg.Coeff.rs"] = (
    df_final["Tav (s)"] * df_final["Reg.Coeff.(DF/sF0)"]
) / df_final["Aav (DF/F0)"]

# Merge spike counts
spike_counts = df.groupby(["ST", "stimulus"]).size().reset_index(name="spikes")
df_final = pd.merge(df_final, spike_counts, on=["ST", "stimulus"], how="right")

# Reorder columns
cols = df_final.columns.tolist()
cols.insert(cols.index("stimulus") + 1, cols.pop(cols.index("spikes")))
df_final = df_final[cols]

# Round numeric columns
df_final = round_significant_figures(df_final, significant_digits=5)

# Ensure that 'ST', 'stimulus', and 'spikes' columns are integers
df_final["ST"] = df_final["ST"].astype(int)
df_final["stimulus"] = df_final["stimulus"].astype(int)
df_final["spikes"] = df_final["spikes"].astype(int)

# Create the 'FileName' column based on the ST and stimulus values
df_final["FileName"] = df_final.apply(
    lambda row: f"{folder_name}/{file_name}_ST{int(row['ST'])}_CCh{int(row['stimulus'])}.txt",
    axis=1,
)

# Move the 'FileName' column to the first position
file_name_col = df_final.pop("FileName")  # Remove the column temporarily
df_final.insert(0, "FileName", file_name_col)  # Insert it at the first position

# Define the save path
save_path = rf"yourpath/{folder_name}/{file_name}" # change the path


#From here, just additional commands for a specific file format. 
# Save paths for both versions of the file
file_with_filename_path = save_path + "_with_filename.txt"
file_without_filename_path = save_path + "_without_filename.txt"

# Step 1: Save the file with the 'FileName' column
save_dataframe_with_header_and_format(df_final, file_with_filename_path)

# Step 2: Drop the 'FileName' column and ensure 'ST', 'stimulus', and 'spikes' remain as integers
df_without_filename = df_final.drop(columns=["FileName"]).copy()

# Convert the integer columns ('ST', 'stimulus', 'spikes') to string to ensure no '.0'
df_without_filename["ST"] = df_without_filename["ST"].apply(lambda x: str(int(x)))
df_without_filename["stimulus"] = df_without_filename["stimulus"].apply(
    lambda x: str(int(x))
)
df_without_filename["spikes"] = df_without_filename["spikes"].apply(
    lambda x: str(int(x))
)

# Save the second file without the 'FileName' column
save_dataframe_with_header_and_format(df_without_filename, file_without_filename_path)

print(f"File with 'FileName' saved to: {file_with_filename_path}")
print(f"File without 'FileName' saved to: {file_without_filename_path}")
