'''Introduction to Machine Learning for Fungal Data Classification
Author: Alejandro Guerrero-López
Date: 2024/11/07'''

import os
import random
from sklearn.model_selection import train_test_split
from data_reader import MaldiDataset
import numpy as np
from scipy.spatial.distance import euclidean
from sklearn.neighbors import KNeighborsClassifier
from collections import Counter
import matplotlib.pyplot as plt

# Set all seeds to make the results reproducible
random.seed(42)
np.random.seed(42)


# This script is a simple starting point to classify fungal data using MALDI-TOF spectra.
# It demonstrates loading the dataset, training a basic classifier, and evaluating its performance.

class SimpleFungusIdentifier:
    def __init__(self, dataset_path, test_size=0.2, random_state=42):
        # Initialize the classifier with dataset path, test size, and random state for reproducibility.
        self.dataset_path = dataset_path
        self.test_size = test_size
        self.random_state = random_state
        self.train_data = []
        self.test_data = []

    def load_and_split_data(self, n_step=3):
        # Load the dataset using MaldiDataset
        dataset = MaldiDataset(self.dataset_path, n_step=n_step)
        dataset.parse_dataset()  # Parse the dataset from the specified path
        data = dataset.get_data()  # Retrieve the parsed data

        # Split the dataset into training and test data, ensuring no overlap of unique samples between train and test
        # Unique genus_species_label
        genus_species_labels = list(set([sample['genus_species_label'] for sample in data]))

        # for each genus_species_label, ensure that 80% of the unique_ids are in train and 20% in test
        train_unique_ids = []
        test_unique_ids = []
        for genus_species_label in genus_species_labels:
            # Get all unique_ids for the current genus_species_label
            unique_ids_for_genus_species = list(set([sample['unique_id_label'] for sample in data if sample['genus_species_label'] == genus_species_label]))
            # Shuffle the unique_ids
            random.shuffle(unique_ids_for_genus_species)
            if len(unique_ids_for_genus_species) == 1:
                # If there is only one unique_id, add it to train
                train_unique_ids.extend(unique_ids_for_genus_species)
                continue
            # Split the unique_ids into train and test
            split_index = int(len(unique_ids_for_genus_species) * (1 - self.test_size))
            train_unique_ids.extend(unique_ids_for_genus_species[:split_index])
            test_unique_ids.extend(unique_ids_for_genus_species[split_index:])
        # Filter the data based on the train and test unique_ids
        self.train_data = [sample for sample in data if sample['unique_id_label'] in train_unique_ids]
        self.test_data = [sample for sample in data if sample['unique_id_label'] in test_unique_ids]

        # Assertions: no unique_id_label should be in both train and test data
        train_unique_ids = [sample['unique_id_label'] for sample in self.train_data]
        test_unique_ids = [sample['unique_id_label'] for sample in self.test_data]
        assert len(set(train_unique_ids).intersection(set(test_unique_ids))) == 0

        # Print total number of unique id labels in train and test data
        print(f"Number of unique_id_labels in train data: {len(set(train_unique_ids))}")
        print(f"Number of unique_id_labels in test data: {len(set(test_unique_ids))}")
        # Total of samples in train and test data
        print(f"Number of samples in train data: {len(self.train_data)}")
        print(f"Number of samples in test data: {len(self.test_data)}")
        # total number of classes to predict (genus+species)
        print(f"Number of classes to predict: {len(set([entry['genus_species_label'] for entry in self.train_data]))}")

    def naive_classifier(self, labels="genus"):
        # Create a naive classifier that calculates the mean spectrum for each label in the training data.
        label_to_mean_spectrum = {}
        for train_sample in self.train_data:
            # Use genus or genus+species label based on input parameter
            label = train_sample['genus_label'] if labels == "genus" else train_sample['genus_species_label']
            spectrum = train_sample['spectrum']
            if label not in label_to_mean_spectrum:
                label_to_mean_spectrum[label] = []
            label_to_mean_spectrum[label].append(spectrum)

        # Calculate the mean spectrum for each label
        for label in label_to_mean_spectrum:
            label_to_mean_spectrum[label] = np.mean(label_to_mean_spectrum[label], axis=0)

        # Store the mean spectrum for each label to use for predictions
        self.label_to_mean_spectrum = label_to_mean_spectrum

    def evaluate_naive_classifier(self, labels="genus"):
        # Evaluate the naive classifier on the test data
        spectra = np.array([entry['spectrum'] for entry in self.test_data])
        true_labels = [entry['genus_label'] if labels == "genus" else entry['genus_species_label'] for entry in self.test_data]

        predicted_labels = []
        # Predict the label for each spectrum in the test set
        for spectrum in spectra:
            min_distance = float('inf')
            min_label = None
            # Find the closest mean spectrum from the training data
            for label, mean_spectrum in self.label_to_mean_spectrum.items():
                distance = euclidean(spectrum, mean_spectrum)
                if distance < min_distance:
                    min_distance = distance
                    min_label = label
            predicted_labels.append(min_label)

        # Calculate accuracy of the naive classifier
        correct_predictions = sum([1 for true, pred in zip(true_labels, predicted_labels) if true == pred])
        accuracy = correct_predictions / len(true_labels)
        print(f"Naive Classifier Accuracy: {accuracy:.2f}")

        return accuracy, predicted_labels, true_labels

    def knn_classifier(self, n_neighbors=5, labels="genus"):
        # Train a K-Nearest Neighbors (KNN) classifier on the training data
        spectra = np.array([entry['spectrum'] for entry in self.train_data])
        train_labels = [entry['genus_label'] if labels == "genus" else entry['genus_species_label'] for entry in self.train_data]

        # Create and fit the KNN classifier
        knn = KNeighborsClassifier(n_neighbors=n_neighbors)
        knn.fit(spectra, train_labels)

        return knn

    def evaluate_knn_classifier(self, knn, labels="genus"):
        # Evaluate the KNN classifier on the test data
        spectra = np.array([entry['spectrum'] for entry in self.test_data])
        true_labels = [entry['genus_label'] if labels == "genus" else entry['genus_species_label'] for entry in self.test_data]

        # Predict the labels using the trained KNN classifier
        predicted_labels = knn.predict(spectra)
        correct_predictions = sum([1 for true, pred in zip(true_labels, predicted_labels) if true == pred])
        accuracy = correct_predictions / len(true_labels)
        print(f"KNN Classifier Accuracy: {accuracy:.2f}")

        return accuracy, predicted_labels, true_labels
    
    def plot_data_distribution(self):
        # Plot the distribution of genus_species labels in train and test data
        train_labels = [entry['genus_species_label'] for entry in self.train_data]
        test_labels = [entry['genus_species_label'] for entry in self.test_data]

        train_counter = Counter(train_labels)
        test_counter = Counter(test_labels)
        total_counter = train_counter + test_counter

        plt.figure(figsize=(10, 6))
        bar_width = 0.35
        bar_positions = np.arange(len(total_counter))
        train_bars = plt.bar(bar_positions, [train_counter[label] for label in total_counter], bar_width, label='Train')
        test_bars = plt.bar(bar_positions + bar_width, [test_counter[label] for label in total_counter], bar_width, label='Test')
        plt.xlabel('Genus+Species Label')
        plt.ylabel('Number of Samples')
        plt.title('Distribution of Genus+Species Labels in Train and Test Data')
        plt.xticks(bar_positions + bar_width / 2, total_counter.keys(), rotation=90)
        plt.legend()
        plt.tight_layout()
        plt.savefig("Distribution_of_Genus_Species_Labels_in_Train_and_Test_Data.png")
        # plt.show()

    def plot_accuracy_per_label(self, true_label, pred, model_name="Naive"):
        # Plot the accuracy per label
        accuracy_per_label = {}
        for true, pred in zip(true_label, pred):
            if true not in accuracy_per_label:
                accuracy_per_label[true] = {'correct': 0, 'total': 0}
            accuracy_per_label[true]['total'] += 1
            if true == pred:
                accuracy_per_label[true]['correct'] += 1

        labels = list(accuracy_per_label.keys())
        accuracies = [accuracy_per_label[label]['correct'] / accuracy_per_label[label]['total'] for label in labels]

        plt.figure(figsize=(10, 6))
        plt.bar(labels, accuracies)
        plt.xlabel('Genus+Species Label')
        plt.ylabel('Accuracy')
        plt.title(f'{model_name} Classifier Accuracy per Genus+Species Label')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(f"{model_name}_Classifier_Accuracy_per_Genus_Species_Label.png")
        # plt.show()

# Define the dataset path (update this path to where your dataset is located)
dataset_path = "data/fungus_db"

# Initialize the classifier with the dataset path
fungus_identifier = SimpleFungusIdentifier(dataset_path)

# Load and split the data into training and test sets. This n_step is a hyperparameter that can be tuned. In [https://www.nature.com/articles/s41591-021-01619-9], it was defined to 3, but it can should be cross-validated.
fungus_identifier.load_and_split_data(n_step=6)

# Plot data distribution
fungus_identifier.plot_data_distribution()

print("====================== GENUS LEVEL CLASSIFIERS ======================")
# Train the Naive Classifier
print("Training Naive Classifier...")
fungus_identifier.naive_classifier(labels="genus")

# Evaluate the Naive Classifier
print("Evaluating Naive Classifier...")
naive_accuracy = fungus_identifier.evaluate_naive_classifier(labels="genus")

# Train and evaluate a KNN Classifier
print("Training KNN Classifier...")
knn = fungus_identifier.knn_classifier(n_neighbors=5, labels="genus")

print("Evaluating KNN Classifier...")
knn_accuracy = fungus_identifier.evaluate_knn_classifier(knn, labels="genus")

print("====================== GENUS SPECIES LEVEL CLASSIFIERS ======================")
# Train the Naive Classifier
print("Training Naive Classifier...")
fungus_identifier.naive_classifier(labels="genus_species")

# Evaluate the Naive Classifier
print("Evaluating Naive Classifier...")
naive_accuracy, naive_pred, naive_true = fungus_identifier.evaluate_naive_classifier(labels="genus_species")

# Plot the accuracy per label for the Naive Classifier
fungus_identifier.plot_accuracy_per_label(naive_true, naive_pred, model_name="Naive")

# Train and evaluate a KNN Classifier
print("Training KNN Classifier...")
knn = fungus_identifier.knn_classifier(n_neighbors=5, labels="genus_species")

print("Evaluating KNN Classifier...")
knn_accuracy, knn_pred, knn_true = fungus_identifier.evaluate_knn_classifier(knn, labels="genus_species")

# Plot the accuracy per label for the KNN Classifier
fungus_identifier.plot_accuracy_per_label(knn_true, knn_pred, model_name="KNN")



# This script provides a starting point for students to understand the process of loading data,
# training simple classifiers (naive and KNN), and evaluating their performance on fungal data.

# Ideas to make this script more advanced:
# 1. Data problems:
#   1.1 Is the data balanced per class? If not, how can you handle class imbalance? (e.g., oversampling, undersampling, generating synthetic data, weighted loss functions)
#   1.2 Data is still high-dimensional, can you reduce the dimensionality of the data using PCA, mRMR, LASSO, or other feature selection/extraction methods?
# 2. Can you improve the performance of the classifiers by optimizing the hyperparameters (e.g., GridSearchCV, RandomizedSearchCV)?
# 3. Can you run nn-based models (e.g., MLP, CNN, RNN) to improve the classification performance? Keep it simple and explainable!
#   3.1. Can you make better the distance-based classifier by using a weighted distance metric (e.g., Mahalanobis distance, or other)?
# 4. How can you visualize the performance of the classifiers (e.g., confusion matrix, ROC curve, precision-recall curve)?
# 5. How can you interpret the results of the classifiers and provide insights into the classification process? Which classes are easy/hard to classify? Which proteins (m/z values) are important for classification of each class? Use SHAP, LIME, or other interpretability methods.
# 6. How can you deploy the classifier to a web application or mobile app for real-time classification of fungal data? (e.g., Flask, Django, FastAPI, Streamlit, TensorFlow Lite, ONNX)

# Compare always to the simplest model (distance-based) to understand the complexity of the problem and the performance of more advanced models.
# This have to work in real life, so always think about the fastest model in inference time, and the most explainable model for the end-user.


# REMEMBER: On real life laboratories, we are interested in genus+species level classification, as it is the most useful for clinicians and researchers.
