# Development Set-up
## Prerequisites
* Python 3.8+
* Jupyter Notebook
* Numpy
* Matplotlib
* Pandas
* Seaborn
* Tensorflow
* OpenCV
* Scikit-Learn
* Scikit-Image
* test files (file1, file2)
* Desktop/Laptop Computer
* Visual Studio Code
## Instructions
1. Install `Python 3.8` or later from [link](https://www.python.org/downloads/)
2. Install `Visual Studio Code` from [link](https://code.visualstudio.com/download) and run the program
3. Press `Ctrl+Shift+X` to open the extensions tab, and search and install the extension `Jupyter`
3. Open a terminal and create a python virtual environment:
```
cd code
py -m venv pmd
pmd\Scripts\activate.bat
```
4. Run `INSTALL_PACKAGES.cmd` to install the packages:
3. Run the following command to use the programs:
* Segmentation and Feature Extraction
```
py peanut_mold_detection\segment_extract.py <batch_type> <seg_type>
```
batch_type and seg_type are optional commands
batch_type allow you to label the samples as 
'nc' non-contaminated (default)
or 'c' contaminated
seg_type allows yout to choose a segmentation method:
'watershed' for marker-based watershed segmentation (default)
or 'colseg; for color segmentation

* Building Neural Network Model
```
myprogram file1 file2
```
Testing Existing Neural Network Model
```
myprogram file1 file2
```
4. A binary file can also be downloaded from this link which can be used by any device.
