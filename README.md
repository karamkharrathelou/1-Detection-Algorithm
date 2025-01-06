#Detaction - Resconstruction - Analysis Algorithms
## Detection-Algorithm
This code helps identify important values from a large dataset to optimize storage, while saving only one value for noise to allow signal reconstruction if needed:
SNR (Signal-to-Noise Ratio) values of the data are calculated.
Values are selected based on a predefined criterion that can be manually adjusted (a tuner for detection sensitivity).
The criterion can also be automatically set as the mean value of the noise.
The values are then visualized to enable visual management of the data.
Important data indices and values are saved in an HDF5 file, reducing storage space by 90%.
