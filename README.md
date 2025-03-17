# Spiking Analysis
This repository contains analytical tools designed for processing and analyzing spikes in HEK cells.

## Spike Train Analysis
  Filename: `Spike_Train_Analysis.py`
  
  Description:
  This script is dedicated to the analysis of spikes from sequences in HEK cells stimulated with Carbachol (CCh). It is designed to perform a detailed statistical analysis of the spikes to understand the underlying   mechanisms and behaviours. 
  Make sure to run the files:
  
  - `libraries.py`
  - `datetime_setup.py`
  - `functions.py` 
  
  before running the main script. 
  In `datetime_setup.py`, you can call the file by specifying the date and experiment number (an example file name is 20170815E1). 
  
  Tasks:
  - Detrending Time Series Data: Removes the deterministic contributions from the Interspike Interval (ISI) and amplitude data, preparing it for unbiased statistical analysis.
  - Moment Relation Calculation: Computes the relationship between the Interspike Interval (ISI) and its Standard Deviation (SD) to explore variability in spike timing.
  - Statistical Coefficient Computation: Calculates various statistical coefficients, including Pearson correlations and regression slopes, to quantify relationships in the data.
  - Correlation Investigation: Analyzes the correlations between computed coefficients to uncover patterns and possible causal relationships in the spike train data.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
To run these scripts, you will need an environment capable of executing Python code and Jupyter notebooks. Ensure that you have the following installed:

Python 3.x
Jupyter Notebook or JupyterLab
Necessary Python packages: numpy, scipy, matplotlib, pandas, peakutils


