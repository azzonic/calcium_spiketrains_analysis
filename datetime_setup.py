import datetime  # Importing datetime for working with dates and time

# Set up date and experiment parameters
date = datetime.date(2024, 8, 9)
experiment = "E2_3"

# Format the date components
year = str(date.year)
month = f"{date.month:02d}"
day = f"{date.day:02d}"

folder_name = rf"{year}{month}{day}"
analysed_data = rf"C:\Users\azzonic\Desktop\PhD_Falcke\experiments\Ca2imag_CalB_CCh_HEK\ISIscriptanalysed_data/CChCB_{year}_{month}_{day}_{experiment}_analysis.csv"
file_name = f"{year}{month}{day}{experiment}"
pycasig_df = rf"C:\Users\azzonic\Desktop\PhD_Falcke\experiments\Ca2imag_CalB_CCh_HEK\ISI\PyCaSig_analysis/{folder_name}/{file_name}.csv"
save_path = rf"C:\Users\azzonic\Desktop\PhD_Falcke\experiments\Ca2imag_CalB_CCh_HEK\ISI\PyCaSig_analysis/{folder_name}/{file_name}"
