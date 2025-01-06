# Detaction - Resconstruction - Analyses Algorithms
## 1. Detection -This code Detect and analyze moving targets, identify important values from a large dataset to optimize storage, while saving only one value for noise to allow signal reconstruction if needed:

  **1. SNR (Signal-to-Noise Ratio) values of the data are calculated.**

  **2. Values are selected based on a predefined criterion that can be manually adjusted (a tuner for detection sensitivity).**

  **3. The criterion can also be automatically set as the mean value of the noise.**

  **4. The values are then visualized to enable visual management of the data.**

  **5. Important data indices and values are saved in an HDF5 file, reducing storage space by 90%.**

## 2. Reconstruction -This code reconstructs the detected values from the detection step:

 **1. HDF5 files are read.**

 **2. A matrix matching the original signal is created, with noise equal to the original noise value.**

 **3. The original saved data is appended to the matrix at its original positions, making the reconstructed signal identical to the original signal but 90% smaller in size.**

 **4. A power profile is generated and plotted for visual management.**

## 3- Analysis Algorithms:
 **1. Signal Processing: Emphasis on processing radar signals (FFT, smoothing, SNR calculation).**
 **2. Data Management: Efficient handling of multi-dimensional radar data using HDF5.**
 **3. Automation: Automates data processing and visualization for multiple radar files.**
