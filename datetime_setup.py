import datetime  # Importing datetime for working with dates and time

# Set up date and experiment parameters
date = datetime.date(2024, 8, 9)
experiment = "E2"

# Format the date components
year = str(date.year)
month = f"{date.month:02d}"
day = f"{date.day:02d}"

#change your path in the following lines: 

folder_name = rf"{year}{month}{day}"
analysed_data = rf"yourpath/CChCB_{year}_{month}_{day}_{experiment}_analysis.csv"
file_name = f"{year}{month}{day}{experiment}"
pycasig_df = rf"yourpath/{folder_name}/{file_name}.csv"
save_path = rf"yourpath/{folder_name}/{file_name}"
