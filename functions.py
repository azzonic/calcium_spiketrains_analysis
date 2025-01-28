import libraries  # Your libraries.py module
import datetime_setup  # Your datetime_setup.py module
from libraries import plt

print("plt available in functions:", plt)

plt.rcParams.update(
    {
        "axes.labelsize": "large",  # Font size of x and y labels
        "axes.titlesize": "large",  # Font size of titles
        "xtick.labelsize": "medium",  # Font size of x-axis ticks
        "ytick.labelsize": "medium",  # Font size of y-axis ticks
        "legend.fontsize": "medium",
    }
)  # Font size of legend


def detrend_column_grouped(df, x_col, y_col, slope_name):
    """
    Detrend a specific column in the dataframe using another column as the x-axis,
    grouped by 'ST' and 'stimulus'.

    Parameters:
    - df: the DataFrame
    - x_col: the column to use as the x-axis (e.g., 'time')
    - y_col: the column to detrend (e.g., 'ISI (s)' or 'A (DF/F0)')
    - slope_name: the name of the slope column to be added (e.g., 'tISI' or 'tA')

    Returns:
    - df_grouped: The detrended DataFrame with new columns and rounded values.
    """

    def detrend_column(df_group):
        # Calculate N (number of points)
        N = len(df_group)
        Sx = df_group[x_col].sum()  # Sum of x (time)
        Sy = df_group[y_col].sum()  # Sum of y (dependent variable)
        Sxx = (df_group[x_col] ** 2).sum()  # Sum of x^2
        Sxy = (df_group[x_col] * df_group[y_col]).sum()  # Sum of x*y

        # Calculate delta and slope B (detrending factor)
        delta = N * Sxx - Sx**2
        if delta == 0:  # Avoid division by zero
            B = 0
        else:
            B = (N * Sxy - Sx * Sy) / delta

        # Calculate the detrended values
        mean_x = Sx / N  # Mean of x
        df_group[f"tf_{y_col}"] = df_group[y_col] - B * (df_group[x_col] - mean_x)
        df_group[slope_name] = B

        return df_group

    # Apply the detrending function to groups by 'ST' and 'stimulus'
    df_grouped = df.groupby(["ST", "stimulus"]).apply(detrend_column)

    # Reset the index to clean up any grouping indices
    df_grouped.reset_index(drop=True, inplace=True)

    return df_grouped


def format_detrended_columns(df):
    """
    Format the detrended DataFrame by renaming and rounding specific columns.

    Parameters:
    - df: The detrended DataFrame.

    Returns:
    - df: The formatted DataFrame with rounded and renamed columns.
    """
    # Rename detrended columns for clarity
    df = df.rename(columns={"tf_A (DF/F0)": "Atf (DF/F0)", "tf_ISI (s)": "ISItf (s)"})

    # Round the necessary columns to 6 decimal places
    df["ISItf (s)"] = df["ISItf (s)"].round(6)
    df["Atf (DF/F0)"] = df["Atf (DF/F0)"].round(6)
    df["tA"] = df["tA"].round(6)
    df["tISI"] = df["tISI"].round(6)

    return df


def check_isi_difference_and_means(df):
    """
    This function computes the difference between 'ISI (s)' and 'ISItf (s)' columns
    without altering the DataFrame. It prints the differences for the first few rows
    and calculates and prints the mean for both columns.

    Parameters:
    - df: DataFrame containing 'ISI (s)' and 'ISItf (s)' columns.

    Returns:
    - None
    """

    # Calculate the difference
    isi_difference = df["ISI (s)"] - df["ISItf (s)"]

    # Display the first few rows with the differences
    print(df[["ISI (s)", "ISItf (s)"]].head())
    print("ISI Difference (first few rows):")
    print(isi_difference.head())

    # Compute the means
    mean_isi = df["ISI (s)"].mean()
    mean_isi_tf = df["ISItf (s)"].mean()

    # Print the means
    print(f"Mean of ISI: {mean_isi}")
    print(f"Mean of ISItf (s): {mean_isi_tf}")


def calculate_pearson(group):
    """
    Calculate the Pearson Correlation Coefficient (PCC) and p-value for the relationship
    between ISItf (s) and Atf (DF/F0) in the given group.

    Parameters:
    - group: A DataFrame containing the group to be analyzed.

    Returns:
    - A Series with the PCC and the p-value.
    """
    corr, p_value = pearsonr(group["ISItf (s)"], group["Atf (DF/F0)"])
    return pd.Series({"PCC": corr, "PValue": p_value})


def plot_comparison(df, y_col_original, y_col_detrended, ylabel, save_filename):
    """
    Plot comparison of original and detrended values over time for both amplitude (A) and ISI.

    Parameters:
    - df: DataFrame with data to plot.
    - y_col_original: Column name for the original y-axis values (e.g., 'A (DF/F0)', 'ISI (s)').
    - y_col_detrended: Column name for the detrended y-axis values (e.g., 'Atf (DF/F0)', 'ISItf (s)').
    - ylabel: Label for the y-axis.
    - save_filename: Filename to save the plot.
    """
    # Determine the number of subplots based on unique values in 'ST'
    num_plots = len(df["ST"].unique())  # Create subplots with one plot per row
    fig, axs = plt.subplots(
        num_plots, 1, figsize=(10, 5 * num_plots)
    )  # Adjust the height of the plot

    # Zoom in by adjusting the y-axis range around the detrended values
    y_min = (
        df[y_col_detrended].min() - 5
    )  # Adjust y-range to visualize detrended values
    y_max = df[y_col_detrended].max() + 5

    # Flatten axs if it's a 2D array
    if num_plots == 1:
        axs = [axs]

    # Define the custom color palette for the stimulus
    custom_palette = {3: "lightblue", 5: "gold", 10: "orange", 15: "red", 30: "purple"}

    # Loop through each spike train and create overlay plots
    for ax, (spike_train, sub_df) in zip(axs, df.groupby("ST")):
        # Plot original values
        sns.scatterplot(
            data=sub_df,
            x="time (s)",
            y=y_col_original,
            hue="stimulus",
            ax=ax,
            palette=custom_palette,
            s=80,
            marker="o",
            alpha=0.7,
        )

        # Plot detrended values
        sns.scatterplot(
            data=sub_df,
            x="time (s)",
            y=y_col_detrended,
            hue="stimulus",
            ax=ax,
            palette=custom_palette,
            s=80,
            marker="X",
            alpha=0.7,
        )

        # Add labels and titles
        ax.set_title(f"Spike Train #{spike_train}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(ylabel)
        ax.set_ylim(y_min, y_max)  # Apply zoomed-in y-axis limits

        # Handle legends for the hue variable and make sure to display both original and detrended in the legend
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(
            handles=handles[: len(custom_palette)],
            labels=labels[: len(custom_palette)],
            title="Stimulus",
        )

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Save and show the plot
    plt.savefig(save_filename)
    plt.show()


def calculate_cv_amplitude(df):
    """
    Calculate the mean, standard deviation, and coefficient of variation (CV) for amplitude (Atf (DF/F0)).

    Parameters:
    - df: DataFrame containing the data.

    Returns:
    - cv_df: A DataFrame with the calculated mean, std deviation, and CV for amplitude.
    """
    cv_data = []

    # Group by spike train and calculate statistics
    for spike_train, sub_df in df.groupby("ST"):
        # Calculate means, standard deviations, and coefficients of variation for amplitude
        stats = sub_df.groupby("stimulus")["Atf (DF/F0)"].agg(["mean", "std"])
        stats["CV"] = stats["std"] / stats["mean"]  # Calculate the CV

        # Define positions (assuming you have a list of positions corresponding to each concentration)
        positions = np.arange(len(stats.index))  # Example positions

        # Append CV data along with mean and its standard deviation to the list
        for conc, pos in zip(stats.index, positions):
            mean = stats.loc[conc, "mean"]
            std = stats.loc[conc, "std"]
            cv = stats.loc[conc, "CV"]

            # Store the data with proper column names for amplitude
            cv_data.append(
                {
                    "ST": spike_train,
                    "stimulus": conc,
                    "Aav (DF/F0)": mean,
                    "SDA (DF/F0)": std,
                    "CVA": cv,
                }
            )

    # Convert cv_data list to DataFrame
    cv_df = pd.DataFrame(cv_data)

    # Extract unique ID, ST, and stimulus from the original DataFrame
    ID_col = df[["ST", "stimulus"]].drop_duplicates()

    # Merge the unique ID DataFrame with the CV DataFrame
    cv_df = pd.merge(ID_col, cv_df, on=["ST", "stimulus"])

    return cv_df


def calculate_cv_ISI(df):
    """
    Calculate the mean, standard deviation, and coefficient of variation (CV) for ISI.

    Parameters:
    - df: DataFrame containing the data.

    Returns:
    - cv_df: A DataFrame with the calculated mean, std deviation, and CV for ISI.
    """
    cv_data = []

    # Group by spike train and calculate statistics
    for spike_train, sub_df in df.groupby("ST"):
        # Calculate means, standard deviations, and coefficients of variation for ISI
        stats = sub_df.groupby("stimulus")["ISItf (s)"].agg(["mean", "std"])
        stats["CV"] = stats["std"] / stats["mean"]  # Calculate the CV

        # Define positions (assuming you have a list of positions corresponding to each concentration)
        positions = np.arange(len(stats.index))  # Example positions

        # Append CV data along with mean and its standard deviation to the list
        for conc, pos in zip(stats.index, positions):
            mean = stats.loc[conc, "mean"]
            std = stats.loc[conc, "std"]
            cv = stats.loc[conc, "CV"]

            # Store the data with proper column names for amplitude
            cv_data.append(
                {
                    "ST": spike_train,
                    "stimulus": conc,
                    "Tav (s)": mean,
                    "SDISI (s)": std,
                    "CVISI": cv,
                }
            )

    # Convert cv_data list to DataFrame
    cv_df = pd.DataFrame(cv_data)

    # Extract unique ID, ST, and stimulus from the original DataFrame
    ID_col = df[["ST", "stimulus"]].drop_duplicates()

    # Merge the unique ID DataFrame with the CV DataFrame
    cv_df = pd.merge(ID_col, cv_df, on=["ST", "stimulus"])

    return cv_df


def plot_box_and_stats(df, cv_df, y_col, ylabel, save_filename):
    """
    Plot boxplots and CV stats for both amplitude (Atf (DF/F0)) and ISI (ISItf (s)).

    Parameters:
    - df: DataFrame with data to plot.
    - cv_df: DataFrame with CV (Coefficient of Variation) statistics.
    - y_col: Column name for the y-axis values (e.g., 'Atf (DF/F0)', 'ISItf (s)').
    - ylabel: Label for the y-axis (e.g., 'Amplitude', 'ISI').
    - save_filename: Filename to save the plot.
    """
    # Determine the number of subplots based on unique values in 'ST'
    num_plots = len(df["ST"].unique())  # Create subplots with one plot per row
    fig, axs = plt.subplots(
        num_plots, 1, figsize=(10, 5 * num_plots)
    )  # Adjust the height of the plot

    # Flatten axs if it's a single Axes object
    if num_plots == 1:
        axs = [axs]

    # Define the custom color palette for the stimulus
    custom_palette = {3: "lightblue", 5: "gold", 10: "orange", 15: "red", 30: "purple"}

    # Loop through each spike train and create boxplots
    for ax, (spike_train, sub_df) in zip(axs, df.groupby("ST")):
        # Plot boxplot for the specified y_col
        sns.boxplot(
            data=sub_df,
            x="stimulus",
            y=y_col,
            hue="stimulus",
            ax=ax,
            palette=custom_palette,
        )
        sns.set(style="ticks")
        sns.color_palette("Paired")

        # Filter cv_df for the current spike train
        plot_cv_df = cv_df[cv_df["ST"] == spike_train]

        # Define positions (assuming you have a list of positions corresponding to each concentration)
        positions = np.arange(len(plot_cv_df))

        # Scatter plot for means
        if "Amplitude" in ylabel:
            ax.scatter(positions, plot_cv_df["Aav (DF/F0)"], color="black")
        elif "ISI" in ylabel:
            ax.scatter(positions, plot_cv_df["Tav (s)"], color="black")

        # Add text annotations for mean, standard deviation, and CV
        for conc, pos in zip(plot_cv_df["stimulus"], positions):
            if "Amplitude" in ylabel:
                mean = plot_cv_df.loc[
                    plot_cv_df["stimulus"] == conc, "Aav (DF/F0)"
                ].values[0]
                std = plot_cv_df.loc[
                    plot_cv_df["stimulus"] == conc, "SDA (DF/F0)"
                ].values[0]
                cv = plot_cv_df.loc[plot_cv_df["stimulus"] == conc, "CVA"].values[0]
            elif "ISI" in ylabel:
                mean = plot_cv_df.loc[plot_cv_df["stimulus"] == conc, "Tav (s)"].values[
                    0
                ]
                std = plot_cv_df.loc[
                    plot_cv_df["stimulus"] == conc, "SDISI (s)"
                ].values[0]
                cv = plot_cv_df.loc[plot_cv_df["stimulus"] == conc, "CVISI"].values[0]

            ax.text(
                pos,
                mean + 0.02,
                f"Mean: {mean:.4f}\nSD: {std:.4f}\nCV: {cv:.4f}",
                ha="center",
                va="bottom",
                fontsize=10,
                color="black",
            )

        # Add labels and titles
        ax.set_title(f"Spike Train #{spike_train}")
        ax.set_xlabel("Stimulus")
        ax.set_ylabel(ylabel)
        ax.get_legend().remove()  # Remove the legend to avoid duplicate legends

    # Adjust layout
    plt.tight_layout()

    # Save and show the plot
    plt.savefig(save_filename)
    plt.show()


def calculate_regression_coefficients(sub_df):
    """
    Calculate regression coefficients (Intercept and Slope) for each group.

    Parameters:
    - sub_df: DataFrame with 'ISItf (s)' as X and 'Atf (DF/F0)' as Y.

    Returns:
    - model.params (Series): The intercept and slope of the regression line.
    """
    if len(sub_df) < 2:
        return None  # Not enough data points to perform regression
    X = sm.add_constant(sub_df["ISItf (s)"])  # Adds the intercept term
    y = sub_df["Atf (DF/F0)"]
    model = sm.OLS(y, X).fit()  # Ordinary Least Squares regression
    return model.params


def get_regression_coefficients(df):
    """
    Groups data by 'ST' and 'stimulus', calculates regression coefficients, and stores them in a DataFrame.

    Parameters:
    - df: DataFrame containing the 'ST', 'stimulus', 'ISItf (s)', and 'Atf (DF/F0)' columns.

    Returns:
    - reg_df: DataFrame with calculated regression coefficients.
    """
    coefficients_list = []  # List to store coefficients

    for (st, interval), group in df.groupby(["ST", "stimulus"]):
        coef = calculate_regression_coefficients(group)
        if coef is not None:
            coefficients_list.append(
                {
                    "ST": st,
                    "stimulus": interval,
                    "Intercept": coef.iloc[0],  # Intercept
                    "Reg.Coeff.(DF/sF0)": coef.iloc[
                        1
                    ],  # Slope (regression coefficient)
                }
            )
        else:
            coefficients_list.append(
                {
                    "ST": st,
                    "stimulus": interval,
                    "Intercept": None,
                    "Reg.Coeff.(DF/sF0)": None,
                }
            )

    # Convert the list of coefficients to a DataFrame
    reg_df = pd.DataFrame(coefficients_list)
    return reg_df


def round_significant_figures(df, significant_digits=6):
    """
    Rounds all numeric columns in a DataFrame to a specified maximum number of significant digits.
    Non-numeric columns and NaN values are left unchanged.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    significant_digits (int): The maximum number of significant digits to round to.

    Returns:
    pd.DataFrame: A DataFrame with numeric columns rounded to the specified significant digits.
    """

    def round_to_significant(x, sig_digits):
        if pd.isna(x):  # Skip NaN values
            return x
        if x == 0:
            return 0
        return round(x, sig_digits - int(np.floor(np.log10(abs(x)))) - 1)

    for col in df.select_dtypes(include=["float", "int"]).columns:
        df[col] = df[col].apply(lambda x: round_to_significant(x, significant_digits))

    return df


def save_dataframe_with_header_and_format(df, file_path):
    """
    Saves a DataFrame with double spaces between columns, properly aligned headers,
    and the index column included without a header. Adds numbers in front of the column headers
    without changing the column names in the DataFrame.

    Parameters:
    df (pd.DataFrame): The DataFrame to save.
    file_path (str): The path where the file will be saved.
    """

    # Reset the index without naming it
    df.reset_index(drop=True, inplace=True)
    df.index += 1  # Start index from 1

    # Define column widths based on the longest element in each column for padding and alignment
    column_widths = {
        col: max(df[col].astype(str).apply(len).max(), len(col)) + 2
        for col in df.columns
    }
    column_widths["Index"] = max(len(str(df.index.max())), 2) + 2  # Index column width

    # Enumerate the columns starting from 1 for 'ST' and add the numbering to the header without changing column names
    numbered_columns = [f"{i} {col}" for i, col in enumerate(df.columns, start=1)]

    # Write the header and data to a file
    with open(file_path, "w") as file:
        # Write the numbered header, aligning with double spaces and prefixing with #
        header = "".join(
            "".ljust(column_widths["Index"])
        )  # Empty header for the index column
        header += "  ".join(
            col.center(column_widths[df.columns[i]])
            for i, col in enumerate(numbered_columns)
        )
        file.write(f"# {header}\n")

        # Write the data aligned with double spaces between columns
        for index, row in df.iterrows():
            row_data = str(index).ljust(column_widths["Index"])  # Add the index column
            row_data += "  " + "  ".join(
                str(row[col]).ljust(column_widths[col]) for col in df.columns
            )
            file.write(row_data + "\n")

    # Print a message confirming the save
    print(f"File with header saved to: {file_path}")


def process_file(file_name, folder_name):
    """
    Reads the specified file, reorders columns to ensure 'FileName' is immediately after the index,
    formats the DataFrame by adding numbers in front of column names starting from 1,
    excluding the first column from numbering, and aligns data under headers.
    Saves the formatted DataFrame to a new file.

    Parameters:
    - file_name (str): Name of the file to process (without extension).
    - folder_name (str): Name of the folder containing the file.

    Returns:
    - pd.DataFrame: The formatted DataFrame.
    """
    # Construct the full file path
    base_path = r"C:/Users/azzonic/Desktop/PhD_Falcke/experiments/Ca2imag_CalB_CCh_HEK/ISI/PyCaSig_analysis"
    file_path = os.path.join(base_path, folder_name, file_name + "_with_filename.txt")

    # Read the data from the file
    df = pd.read_csv(file_path, sep="  ", engine="python")

    # Assuming 'FileName' is not the first column and needs to be moved right after the index
    # which is the first column in df
    # Identify all columns, then rearrange them so 'FileName' is the second column
    column_order = [df.columns[0], "FileName"] + [
        col for col in df.columns[1:] if col != "FileName"
    ]
    df = df[column_order]

    # Modify column names, skipping numbering for the first column, start numbering from 1 for the rest
    new_columns = [
        col if idx == 0 else f"{idx} {col}" for idx, col in enumerate(df.columns)
    ]
    df.columns = new_columns

    # Define a column width dictionary for padding
    column_widths = {
        col: max(df[col].astype(str).apply(len).max(), len(col)) + 2
        for col in df.columns
    }  # +2 for padding

    # Adjust pandas display options for better formatting
    pd.set_option("display.expand_frame_repr", False)  # Prevent line wrapping
    pd.set_option("display.colheader_justify", "center")  # Center column headers

    # Save the formatted DataFrame to a new file
    output_file = os.path.join(
        base_path, folder_name, "formatted_" + file_name + ".txt"
    )
    with open(output_file, "w") as f:
        # Manually write the header aligned
        f.write("".join(col.ljust(column_widths[col]) for col in df.columns) + "\n")

        # Write data aligned
        for index, row in df.iterrows():
            f.write(
                "".join(str(row[col]).ljust(column_widths[col]) for col in df.columns)
                + "\n"
            )

    print(f"Formatted file saved to: {output_file}")

    # Return the formatted DataFrame
    return df
