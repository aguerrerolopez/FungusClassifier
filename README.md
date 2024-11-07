# Fungal Classifier using MALDI-TOF Spectra

This project provides a simple implementation of a fungal classifier using MALDI-TOF spectra data. It is intended as an educational resource for students to learn the basics of data classification, feature extraction, and model evaluation in a biological context.

## Overview

The script, `fungus_classifier.py`, demonstrates the following steps:

1. **Loading the Dataset**: The dataset is loaded using the `MaldiDataset` class. The spectra data includes measurements for various fungal samples, and we split this data into training and test sets.
2. **Feature Extraction and Preprocessing**: The dataset is preprocessed and split, ensuring that unique samples do not overlap between the training and test sets.
3. **Training Classifiers**:
   - **Naive Mean Distance Classifier**: This classifier calculates the mean spectrum for each label in the training set and uses the Euclidean distance to classify test samples.
   - **K-Nearest Neighbors (KNN) Classifier**: A simple KNN classifier is trained to classify the fungi into different labels using neighborhood information.
4. **Model Evaluation**:
   - The accuracy of both the Naive and KNN classifiers is calculated.
   - Accuracy per label is plotted to understand how well each genus/species is being classified.
5. **Data Visualization**: Several visualizations are generated to understand the dataset and the classifier performance.

## Installation

To run this script, you will need to have Python 3 installed, as well as the following Python libraries:

- `numpy`
- `scipy`
- `scikit-learn`
- `matplotlib`
- `collections`

You can install these dependencies using pip:

```sh
pip install numpy scipy scikit-learn matplotlib
```

## Usage

1. **Prepare the Dataset**: Ensure that your MALDI-TOF spectra data is available in a directory and update the `dataset_path` variable with the path to your data.
2. **Run the Script**: Execute the script to train the classifiers and evaluate their performance.

```sh
python fungus_classifier.py
```

## Features

- **Naive Classifier**: Calculates the mean spectrum for each genus or genus+species label in the training data. For each test sample, the classifier assigns the label whose mean spectrum is closest (using Euclidean distance).
- **KNN Classifier**: Uses k-nearest neighbors to classify test samples based on the labels of the closest training samples.

## Visualizations

The script generates the following plots to provide insights into the data and the classifier performance:

1. **Distribution of Genus+Species Labels in Train and Test Data**: This bar plot shows how the samples are distributed across different labels in the training and test datasets.
2. **Accuracy per Label**: This plot shows the accuracy of classification for each genus/species label, helping to understand which labels are well classified and which are not.

## Example Workflow

1. **Loading and Splitting Data**: The dataset is loaded and split into training and test sets using the `load_and_split_data()` method. This ensures that each sample's unique identifier is present in either training or test, but not both.
2. **Training Naive Classifier**: The `naive_classifier()` method calculates the average spectrum per label. Then the `evaluate_naive_classifier()` method is used to evaluate the classifier and plot accuracy per label.
3. **Training and Evaluating KNN Classifier**: The `knn_classifier()` method trains a KNN model using the training spectra, and the `evaluate_knn_classifier()` method is used for evaluation.
4. **Plotting Results**: The script plots the distribution of genus/species labels and the accuracy per label for a clear understanding of the model's performance.

## Output

- **`Distribution_of_Genus_Species_Labels_in_Train_and_Test_Data.png`**: Shows the distribution of labels across train and test sets.
- **`Accuracy_per_Label_for_MeanDistance_Classifier_genus.png`**: Shows the accuracy per label for the Naive classifier (genus level).
- **`Accuracy_per_Genus_Species_Label_in_Test_Data.png`**: Shows the accuracy per genus/species label in the test data.

## Some ideas for your project

1. **Data Problems**:
   * **Is the data balanced per class?** If not, how can you handle class imbalance? (e.g., oversampling, undersampling, generating synthetic data, weighted loss functions).
   * **Data is still high-dimensional**: Can you reduce the dimensionality of the data using PCA, mRMR, LASSO, or other feature selection/extraction methods?
2. **Classifier Optimization**:
   * Can you improve the performance of the classifiers by optimizing the hyperparameters (e.g., GridSearchCV, RandomizedSearchCV)?
3. **Neural Network Models**:
   * Can you run NN-based models (e.g., MLP, 1DCNN to improve the classification performance? Keep it simple and explainable!
   * **Distance-based Classifier Improvement**: Can you make the distance-based classifier better by using a weighted distance metric (e.g., Mahalanobis distance, or other)?
4. **Visualization of Classifier Performance**:
   * How can you visualize the performance of the classifiers? (e.g., confusion matrix, ROC curve, precision-recall curve).
5. **Interpreting Classifier Results**:
   * How can you interpret the results of the classifiers and provide insights into the classification process? Which classes are easy/hard to classify? Which proteins (m/z values) are important for classification of each class? Use SHAP, LIME, or other interpretability methods.
6. **Deployment (completely optional)**:
   * How can you deploy the classifier to a web application or mobile app for real-time classification of fungal data? (e.g., Flask, Django, FastAPI, Streamlit, TensorFlow Lite, ONNX).
7. **Comparison with Simple Models**:
   * Always compare to the simplest model (distance-based) to understand the complexity of the problem and the performance of more advanced models.
8. **Real-life Considerations**:
   * This has to work in real life, so always think about the fastest model in inference time, and the most explainable model for the end-user

# Data Reader Functions

The `data_reader` module in this project provides various functions for reading and preprocessing spectral data. Below is an overview of the available functions:

### Reading Data

- **`from_bruker(acqu_file, fid_file)`**: Reads a spectrum from Bruker files, taking the "acqu" and "fid" files as inputs. This function uses metadata to properly calculate the mass/charge (m/z) values and extract intensity data, allowing for a comprehensive SpectrumObject.
- **`from_tsv(file, sep=" ")`**: Reads a spectrum from a tab-separated value file, extracting the m/z and intensity values from the first two columns.

### Preprocessing Functions

- **`Binner(start=2000, stop=20000, step=3, aggregation="sum")`**: Bins spectra into equal-width intervals, aggregating intensities using the specified method.
- **`Normalizer(sum=1)`**: Normalizes the intensity values to ensure the total intensity is equal to the specified sum (default is 1).
- **`Trimmer(min=2000, max=20000)`**: Trims m/z values outside the specified range, removing inaccurate measurements.
- **`VarStabilizer(method="sqrt")`**: Applies a transformation to stabilize variance, using methods like square root, log, log2, or log10.
- **`BaselineCorrecter(method="SNIP", ...)`**: Corrects the baseline using SNIP, ALS, or ArPLS methods, removing background noise from spectra.
- **`Smoother(halfwindow=10, polyorder=3)`**: Smooths the spectrum using a Savitzky-Golay filter to reduce noise.
- **`LocalMaximaPeakDetector(SNR=2, halfwindowsize=20)`**: Detects peaks by finding local maxima and using a signal-to-noise ratio threshold.
- **`PeakFilter(max_number=None, min_intensity=None)`**: Filters peaks by height or limits the number of peaks based on specified criteria.
- **`RandomPeakShifter(std=1.0)`**: Adds random Gaussian noise to the m/z values of peaks to simulate variability.
- **`UniformPeakShifter(range=1.5)`**: Adds uniform noise to the m/z values of peaks within the specified range.
- **`Binarizer(threshold)`**: Converts intensity values to binary (0 or 1) based on a specified threshold.
- **`SequentialPreprocessor(*args)`**: Chains multiple preprocessing steps into one callable pipeline for ease of use. For example, this allows applying variance stabilization, smoothing, baseline correction, normalization, binning, etc., in sequence.

### Typical Preprocessing Order Example

A typical order of preprocessing steps using `SequentialPreprocessor` might look like this:

```python
SequentialPreprocessor(
    VarStabilizer(method="sqrt"),
    Smoother(halfwindow=10),
    BaselineCorrecter(method="SNIP", snip_n_iter=20),
    Trimmer(),
    Binner(step=self.n_step),
    Normalizer(sum=1),
)
```
