# Spike Trace Analysis
This repository contains analytical tools designed for processing and analyzing spike traces extracted from time-lapses of cells loaded with Calbryte 520AM dye and stimulated with Carbachol.

## Contents
There are two main scripts in this repository, each tailored for specific parts of the data analysis workflow:

### 1. Jupyter Notebook: for Baseline Correction and Normalization
  Filename: Baseline_Correction_and_Normalization.ipynb
  
  Description:
  The Jupyter Notebook is used for preprocessing spike trace data obtained from time-lapse imaging. The notebook utilizes the Peakutils package to perform baseline correction through polynomial fitting. Following baseline correction, the notebook processes the fluorescence intensities of cells to normalize them.
  
  Tasks:
  - Baseline correction using polynomial fitting.
  - Normalization (deltaF/F0) of fluorescence intensities.


### 2. Python Script: Spike Train Analysis
  Filename: Spike_Train_Analysis.py
  
  Description:
  This script is dedicated to the analysis of spike trains in cells stimulated with Carbachol (CCh). It is designed to perform a detailed statistical analysis of the spike trains to understand the underlying mechanisms and behaviors.
  
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

To use these scripts:
Clone this repository to your local machine.
Open the Jupyter Notebook Baseline_Correction_and_Normalization.ipynb in JupyterLab, Jupyter Notebook or Visual Studio to perform the baseline correction and normalization.
Run the Spike_Train_Analysis.py script in a Python environment to carry out the detailed spike train analysis.
