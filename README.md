# FER2013 Emotion Recognition Project

This project implements an emotion recognition system using the FER2013 dataset. It utilizes image data to classify facial expressions into seven different emotions: Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral. The project involves data preprocessing, dimensionality reduction with PCA, model training using Support Vector Machines (SVM), and model evaluation.

## Table of Contents

1.  [Dataset](#dataset)
2.  [Libraries Used](#libraries-used)
3.  [Data Preprocessing](#data-preprocessing)
4.  [Dimensionality Reduction with PCA](#dimensionality-reduction-with-pca)
5.  [Model Training](#model-training)
6.  [Model Evaluation](#model-evaluation)
7.  [Usage](#usage)
8. [Results](#results)

## Dataset

The project uses the FER2013 dataset, which is a collection of facial images labeled with emotions. The dataset can be downloaded using the `kagglehub` library.

```python
python import kagglehub
path = kagglehub.dataset_download("deadskull7/fer2013")
```

## Libraries Used

*   **pandas** (v2.1.3): For data manipulation and analysis.
*   **numpy** (v1.26.3): For numerical operations, especially with arrays and matrices.
*   **matplotlib** (v3.8.2): For creating visualizations and plots.
*   **scikit-learn** (v1.3.2):
    *   `train_test_split`: For splitting data into training and testing sets.
    *   `cross_val_score`: for perform cross validation on the model.
    *   `PCA`: For dimensionality reduction.
    *   `SVC`: For implementing the Support Vector Machine classifier.
    *   `classification_report`: For generating a detailed performance report.
    *   `accuracy_score`: For calculating the accuracy of the model.
    * `ConfusionMatrixDisplay` :  for visualizing the confusion matrix.
    * `confusion_matrix` : for generating the confusion matrix.
* **kagglehub** (v0.0.7): To download the dataset from kagglehub

To install the above libraries, run the following code in a code cell in Colab:
```bash
pip install pandas==2.1.3 numpy==1.26.3 matplotlib==3.8.2 scikit-learn==1.3.2 kagglehub==0.0.7
```
## Data Preprocessing

1.  **Loading the Dataset:** The `fer2013.csv` file is loaded into a pandas DataFrame.
2.  **Extracting Pixel Data and Labels:** Pixel strings are converted into numerical arrays, and emotion labels are extracted.
3.  **Normalization:** Pixel values are normalized to a range between 0 and 1 by dividing by 255.
4. **Split the data:** split the data into two parts train and test with 0.2 as the test size.

## Dimensionality Reduction with PCA

*   **PCA Application:** Principal Component Analysis (PCA) is applied to the training and testing data to reduce the dimensionality of the pixel data.
*   **Number of Components:** The number of principal components is set to 100, capturing a high percentage of the explained variance. The `explained_variance` variable tell how much the PCA preserve data.

## Model Training

*   **Model:** A Support Vector Machine (SVM) classifier with an RBF kernel is used.
*   **Hyperparameters:** The `C` parameter is set to 1, and `class_weight` is set to `balanced`.
*   **Cross-Validation:** 5-fold cross-validation is performed to evaluate the model's performance on the training set.

## Model Evaluation

*   **Metrics:** The model's performance is evaluated using:
    *   **Accuracy:** Calculated for both training and testing sets.
    *   **Classification Report:** Provides precision, recall, F1-score, and support for each emotion class.
    *   **Confusion Matrix:** Visualizes the performance of the model, showing the true positive, true negative, false positive and false negative.

## Usage

1.  **Clone the Repository:** Get the repository files.
2.  **Install Dependencies:** Install the required libraries.
3.  **Run the Notebook:** Open and run the notebook in Google Colab. The notebook is structured sequentially, with each section building upon the previous one.

## Results

The model showed an accuracy of around 80% on the test data. The cross-validation score was around 47% .
