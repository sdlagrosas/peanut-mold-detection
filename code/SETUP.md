# Development Set-up

## Prerequisites

* Python 3.8 or newer
* Numpy
* Matplotlib
* Pandas
* Seaborn
* Tensorflow
* OpenCV
* Scikit-Learn
* Scikit-Image
* Test image files inside:
  * `samples\Contaminated\Test_batch`
  * `samples\Non-Contaminated\Test_batch`
* Desktop/Laptop Computer
* Visual Studio Code with Jupyter Notebook Extension

## Instructions

1. Install `Python 3.8` or later from [link](https://www.python.org/downloads/)
2. Install `Visual Studio Code` from [link](https://code.visualstudio.com/download) and run the program
3. Press `Ctrl+Shift+X` to open the extensions tab, and search and install the extension `Jupyter`
4. Open a terminal and create a python virtual environment:

    ```
    cd code
    py -m venv pmd
    pmd\Scripts\activate.bat
    ```

5. Run `INSTALL_PACKAGES.cmd` to install the packages:
6. Run the following command to use the programs:

* Segmentation and Feature Extraction

    ```
    py peanut_mold_detection\segment_extract.py <batch_type> <seg_type>
    ```

  * `batch_type` and `seg_type` are optional commands only
  * `batch_type` allow you to label the samples as `'nc'` non-contaminated (default) or `'c'` contaminated
  * `seg_type` allows you to choose a segmentation method: `'watershed'` for marker-based watershed segmentation (default) or `'colseg'` for color segmentation

* Building Neural Network Model

    1. Open `create_ann_model.ipnyb` in Visual Studio Code
    2. Read the comments in each code block.
    3. Use `pmd` or the Python virtual environment you created earlier to run the code blocks.
    4. After running the code blocks, you have already produced a model saved in `peanut_mold_detection\models` that can be used on all platforms supported by `Tensorflow` or you can test the model on a new dataset in `validate_model.ipynb`

* Testing Existing Neural Network Model

    1. Open `validate_model.ipnyb` in Visual Studio Code
    2. Read the comments in each code block.
    3. Use `pmd` or the Python virtual environment you created earlier to run the code blocks.
