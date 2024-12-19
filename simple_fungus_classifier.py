'''Introduction to Machine Learning for Fungal Data Classification
Author: Alejandro Guerrero-López
Date: 2024/11/07'''

import os
import random
from sklearn.model_selection import train_test_split
from data_reader import MaldiDataset
import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean, cityblock, cosine
from sklearn.neighbors import KNeighborsClassifier
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.model_selection import GridSearchCV
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Input, Dense, Lambda
# from tensorflow.keras.losses import mse
# from tensorflow.keras import backend as K

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

        # Prepare spectra and labels
        train_spectra = np.array([sample['spectrum'] for sample in self.train_data])
        test_spectra = np.array([sample['spectrum'] for sample in self.test_data])
        train_labels = [sample['genus_species_label'] for sample in self.train_data]
        test_labels = [sample['genus_species_label'] for sample in self.test_data]

        # # -------------------------------------------------------------------------------------------------------------------------
        
        # PCA
        # PCA for dimensionality reduction
        n_components = 70  # Tunable parameter
        pca = PCA(n_components=n_components)
        train_spectra_pca = pca.fit_transform(train_spectra)
        test_spectra_pca = pca.transform(test_spectra)

        print(f"Explained Variance Ratio (first {n_components} components): {sum(pca.explained_variance_ratio_):.2f}")

        # Identify minority classes in training data
        label_counts = Counter(train_labels)
        threshold = 0.1 * max(label_counts.values())  # Define threshold as 10% of the largest class
        minority_classes = [cls for cls, count in label_counts.items() if count < threshold]
        print(f"Minority Classes (threshold {threshold}): {minority_classes}")

        # Separate spectra and labels into majority and minority
        minority_spectra = [spectrum for spectrum, label in zip(train_spectra_pca, train_labels) if label in minority_classes]
        minority_labels = [label for label in train_labels if label in minority_classes]
        majority_spectra = [spectrum for spectrum, label in zip(train_spectra_pca, train_labels) if label not in minority_classes]
        majority_labels = [label for label in train_labels if label not in minority_classes]

        # Apply SMOTE only to the minority classes
        smote = SMOTE(random_state=42)
        minority_spectra_smote, minority_labels_smote = smote.fit_resample(minority_spectra, minority_labels)

        # Combine the oversampled minority classes with the original majority classes
        train_spectra_smote = np.vstack((minority_spectra_smote, majority_spectra))
        train_labels_smote = np.hstack((minority_labels_smote, majority_labels))

        # Debug: Print class distribution before and after SMOTE
        print(f"Class distribution before SMOTE: {label_counts}")
        print(f"Class distribution after SMOTE: {Counter(train_labels_smote)}")

        # Reformat SMOTE-transformed training data
        self.train_data_smote = [
            {'spectrum': spectrum, 'genus_species_label': label, 'genus_label': label.split()[0]}
            for spectrum, label in zip(train_spectra_smote, train_labels_smote)
        ]

        # Reformat PCA-transformed test data
        self.test_data_pca = [
            {'spectrum': spectrum, 'genus_species_label': label, 'genus_label': label.split()[0]}
            for spectrum, label in zip(test_spectra_pca, [entry['genus_species_label'] for entry in self.test_data])
        ]


        # # --------------------------------------------------------------------------------------------------------------------------

        # # VAEs + PCA + SMOTE

        # # After splitting train/test data
        # threshold = 10  # Minimum number of samples required
        # underrepresented_classes = [label for label, count in Counter(train_labels).items() if count < threshold]

        # print(f"Underrepresented classes (less than {threshold} samples): {underrepresented_classes}")

        # input_dim = len(train_spectra[0])  # Dimension of spectra
        # latent_dim = 5  # Small latent dimension

        # # Train VAE for underrepresented classes and generate synthetic data
        # for target_class in underrepresented_classes:
        #     print(f"Generating data for class: {target_class}")
        #     class_data = np.array([sample['spectrum'] for sample in self.train_data if sample['genus_species_label'] == target_class])
            
        #     if len(class_data) == 0:
        #         continue

        #     # Build and train VAE
        #     vae = build_vae(input_dim, latent_dim)
        #     vae.fit(class_data, class_data, epochs=50, batch_size=16, verbose=0)

        #     # Generate synthetic spectra
        #     synthetic_spectra = self.generate_synthetic_data(vae, class_data, num_samples=10)
        #     for spectrum in synthetic_spectra:
        #         self.train_data.append({'spectrum': spectrum, 'genus_species_label': target_class, 'genus_label': target_class.split()[0]})

        # # Apply PCA
        # print("Applying PCA...")
        # train_spectra = np.array([sample['spectrum'] for sample in self.train_data])
        # test_spectra = np.array([sample['spectrum'] for sample in self.test_data])

        # # Apply PCA for dimensionality reduction
        # n_components = 70  # You can tune this parameter
        # pca = PCA(n_components=n_components)
        # train_spectra_pca = pca.fit_transform(train_spectra)
        # test_spectra_pca = pca.transform(test_spectra)

        # print(f"Explained Variance Ratio (first {n_components} components): {sum(pca.explained_variance_ratio_):.2f}")

        # # Apply SMOTE only to minority classes
        # smote = SMOTE(random_state=42)
        # train_spectra_smote, train_labels_smote = smote.fit_resample(train_spectra_pca, train_labels)

        # # Debug: Print class distribution before and after SMOTE
        # print(f"Class distribution before SMOTE: {Counter(train_labels)}")
        # print(f"Class distribution after SMOTE: {Counter(train_labels_smote)}")

        # # Reformat SMOTE-transformed training data
        # self.train_data_smote = [
        #     {'spectrum': spectrum, 'genus_species_label': label, 'genus_label': label.split()[0]}
        #     for spectrum, label in zip(train_spectra_smote, train_labels_smote)
        # ]

        # # Reformat PCA-transformed test data
        # self.test_data_pca = [
        #     {'spectrum': spectrum, 'genus_species_label': label, 'genus_label': label.split()[0]}
        #     for spectrum, label in zip(test_spectra_pca, [entry['genus_species_label'] for entry in self.test_data])
        # ]

        # # --------------------------------------------------------------------------------------------------------------------------

        # # LASSO
        # # Prepare spectra and labels
        # train_spectra = np.array([sample['spectrum'] for sample in self.train_data])
        # test_spectra = np.array([sample['spectrum'] for sample in self.test_data])
        # train_labels = [sample['genus_species_label'] for sample in self.train_data]
        # test_labels = [sample['genus_species_label'] for sample in self.test_data]

        # # Standardize the spectra (required for LASSO)
        # scaler = StandardScaler()
        # train_spectra_scaled = scaler.fit_transform(train_spectra)
        # test_spectra_scaled = scaler.transform(test_spectra)

        # # Apply LASSO for feature selection
        # alpha = 0.001  # Regularization strength (tune this value)
        # lasso = Lasso(alpha=alpha, max_iter=10000)  # Increase iterations

        # # Encode string labels to numerical values
        # label_encoder = LabelEncoder()
        # train_labels_encoded = label_encoder.fit_transform(train_labels)

        # # Fit LASSO
        # lasso.fit(train_spectra_scaled, train_labels_encoded)

        # # Select non-zero features
        # selected_features = np.where(lasso.coef_ != 0)[0]
        # print(f"Number of selected features: {len(selected_features)}")

        # # Reduce train and test spectra to selected features
        # train_spectra_lasso = train_spectra_scaled[:, selected_features]
        # test_spectra_lasso = test_spectra_scaled[:, selected_features]

        # # Replace original train and test data with LASSO-transformed spectra
        # # Reformat LASSO data to match original train/test data format
        # self.train_data_lasso = [
        #     {'spectrum': spectrum, 'genus_species_label': label, 'genus_label': label.split()[0]}
        #     for spectrum, label in zip(train_spectra_lasso, train_labels)
        # ]

        # self.test_data_lasso = [
        #     {'spectrum': spectrum, 'genus_species_label': label, 'genus_label': label.split()[0]}
        #     for spectrum, label in zip(test_spectra_lasso, test_labels)
        # ]

    
    # # Grid-search PCA + LASSO
    # def grid_search_pca_lasso(self, n_components_list, alpha_list):
    #     results = []

    #     for n_components in n_components_list:
    #         for alpha in alpha_list:
    #             print(f"Testing n_components={n_components}, alpha={alpha}...")

    #             # Standardize the spectra
    #             train_spectra = np.array([sample['spectrum'] for sample in self.train_data])
    #             test_spectra = np.array([sample['spectrum'] for sample in self.test_data])
    #             train_labels = [sample['genus_species_label'] for sample in self.train_data]
    #             test_labels = [sample['genus_species_label'] for sample in self.test_data]

    #             scaler = StandardScaler()
    #             train_spectra_scaled = scaler.fit_transform(train_spectra)
    #             test_spectra_scaled = scaler.transform(test_spectra)

    #             # Apply PCA
    #             pca = PCA(n_components=n_components)
    #             train_spectra_pca = pca.fit_transform(train_spectra_scaled)
    #             test_spectra_pca = pca.transform(test_spectra_scaled)

    #             # Apply LASSO
    #             lasso = Lasso(alpha=alpha, max_iter=10000)
    #             # Encode string labels to numerical values
    #             label_encoder = LabelEncoder()
    #             train_labels_encoded = label_encoder.fit_transform(train_labels)

    #             # Fit LASSO
    #             lasso.fit(train_spectra_pca, train_labels_encoded)
    #             selected_features = np.where(lasso.coef_ != 0)[0]

    #             # If no features are selected, skip this combination
    #             if len(selected_features) == 0:
    #                 print("No features selected. Skipping...")
    #                 continue

    #             train_spectra_lasso = train_spectra_pca[:, selected_features]
    #             test_spectra_lasso = test_spectra_pca[:, selected_features]

    #             # Prepare data in correct format
    #             self.train_data_lasso = [
    #                 {'spectrum': spectrum, 'genus_species_label': label, 'genus_label': label.split()[0]}
    #                 for spectrum, label in zip(train_spectra_lasso, train_labels)
    #             ]

    #             self.test_data_lasso = [
    #                 {'spectrum': spectrum, 'genus_species_label': label, 'genus_label': label.split()[0]}
    #                 for spectrum, label in zip(test_spectra_lasso, test_labels)
    #             ]

    #             # Train and evaluate classifiers
    #             print("Training classifiers...")
    #             self.naive_classifier(labels="genus_species")
    #             naive_accuracy, _, _ = self.evaluate_naive_classifier(labels="genus_species")

    #             knn = self.knn_classifier(n_neighbors=5, labels="genus_species")
    #             knn_accuracy, _, _ = self.evaluate_knn_classifier(knn, labels="genus_species")

    #             # Store results
    #             results.append({
    #                 'n_components': n_components,
    #                 'alpha': alpha,
    #                 'naive_accuracy': naive_accuracy,
    #                 'knn_accuracy': knn_accuracy,
    #                 'num_features_selected': len(selected_features)
    #             })

    #     # Convert results to DataFrame for better visualization
    #     results_df = pd.DataFrame(results)
    #     print(results_df)
    #     return results_df


    # def build_vae(input_dim, latent_dim=2):
    #     # Encoder
    #     inputs = Input(shape=(input_dim,))
    #     h = Dense(64, activation='relu')(inputs)
    #     z_mean = Dense(latent_dim)(h)
    #     z_log_var = Dense(latent_dim)(h)

    #     # Sampling
    #     def sampling(args):
    #         z_mean, z_log_var = args
    #         epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim))
    #         return z_mean + K.exp(z_log_var / 2) * epsilon

    #     z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

    #     # Decoder
    #     decoder_h = Dense(64, activation='relu')
    #     decoder_mean = Dense(input_dim, activation='linear')
    #     h_decoded = decoder_h(z)
    #     outputs = decoder_mean(h_decoded)

    #     # Define VAE
    #     vae = Model(inputs, outputs)
    #     reconstruction_loss = mse(inputs, outputs) * input_dim
    #     kl_loss = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    #     vae_loss = K.mean(reconstruction_loss + kl_loss)
    #     vae.add_loss(vae_loss)
    #     vae.compile(optimizer='adam')

    #     return vae

    # def generate_synthetic_data(self, vae, class_data, num_samples=10):
    #     # Generate synthetic data using the trained VAE
    #     latent_dim = vae.get_layer('lambda').output_shape[1]
    #     z_samples = np.random.normal(size=(num_samples, latent_dim))
    #     decoder = Model(vae.input, vae.output)
    #     synthetic_spectra = decoder.predict(z_samples)
    #     return synthetic_spectra

    def naive_classifier(self, labels="genus", use_weights=False):
        # Calculate class weights if needed
        if use_weights:
            class_counts = Counter([entry['genus_label'] if labels == "genus" else entry['genus_species_label'] for entry in self.train_data_smote])
            total_samples = sum(class_counts.values())
            class_weights = {label: total_samples / count for label, count in class_counts.items()}
        else:
            class_weights = None

        # Create a naive classifier that calculates the mean spectrum for each label in the training data.
        label_to_mean_spectrum = {}
        for train_sample in self.train_data_smote:
            # Use genus or genus+species label based on input parameter
            label = train_sample['genus_label'] if labels == "genus" else train_sample['genus_species_label']
            spectrum = train_sample['spectrum']
            if label not in label_to_mean_spectrum:
                label_to_mean_spectrum[label] = []
            label_to_mean_spectrum[label].append(spectrum)

        # Calculate the mean spectrum for each label
        # for label in label_to_mean_spectrum:
        #     label_to_mean_spectrum[label] = np.mean(label_to_mean_spectrum[label], axis=0)

        # Calculate mean spectrum (weighted if use_weights=True)
        for label, spectra in label_to_mean_spectrum.items():
            spectra_array = np.array(spectra)
            if use_weights:
                label_to_mean_spectrum[label] = np.average(spectra_array, axis=0, weights=np.ones(len(spectra_array)) * class_weights[label])
            else:
                label_to_mean_spectrum[label] = np.mean(spectra_array, axis=0)
                
        # Store the mean spectrum for each label to use for predictions
        self.label_to_mean_spectrum = label_to_mean_spectrum

    def evaluate_naive_classifier(self, labels="genus", metric="euclidean"):
        # Map metric names to functions
        metric_map = {
            "euclidean": euclidean,
            "manhattan": cityblock,
            "cosine": cosine
        }
        distance_function = metric_map.get(metric, euclidean)  # Default to euclidean if metric not recognized

        # Evaluate the naive classifier on the test data
        spectra = np.array([entry['spectrum'] for entry in self.test_data_pca])
        true_labels = [entry['genus_label'] if labels == "genus" else entry['genus_species_label'] for entry in self.test_data_pca]

        predicted_labels = []
        # Predict the label for each spectrum in the test set
        for spectrum in spectra:
            min_distance = float('inf')
            min_label = None
            # Find the closest mean spectrum from the training data
            for label, mean_spectrum in self.label_to_mean_spectrum.items():
                distance = distance_function(spectrum, mean_spectrum)
                if distance < min_distance:
                    min_distance = distance
                    min_label = label
            predicted_labels.append(min_label)

        # Calculate accuracy of the naive classifier
        correct_predictions = sum([1 for true, pred in zip(true_labels, predicted_labels) if true == pred])
        accuracy = correct_predictions / len(true_labels)
        print(f"Naive Classifier Accuracy ({metric} distance): {accuracy:.2f}")

        return accuracy, predicted_labels, true_labels

    def tune_naive_classifier(self, labels="genus"):
        metrics = ["euclidean", "manhattan", "cosine"]  # Distance metrics to test
        weight_options = [False, True]  # Weighted mean vs. unweighted mean
        best_config = None
        best_accuracy = 0

        # Iterate through all combinations of weights and metrics
        for use_weights in weight_options:
            for metric in metrics:
                print(f"Testing Naive Classifier with weights={use_weights}, metric={metric}...")
                # Train Naive Classifier
                self.naive_classifier(labels=labels, use_weights=use_weights)
                # Evaluate Naive Classifier
                accuracy, _, _ = self.evaluate_naive_classifier(labels=labels, metric=metric)
                print(f"Accuracy: {accuracy:.2f}")

                # Track the best configuration
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_config = {"use_weights": use_weights, "metric": metric}

        print(f"Best configuration: {best_config}, Accuracy: {best_accuracy:.2f}")
        return best_config



    def knn_classifier(self, n_neighbors=5, labels="genus", tune = False):
        # Train a K-Nearest Neighbors (KNN) classifier on the training data
        spectra = np.array([entry['spectrum'] for entry in self.train_data_smote])
        train_labels = [entry['genus_label'] if labels == "genus" else entry['genus_species_label'] for entry in self.train_data_smote]

        if tune:
            # Hyperparameter tuning with GridSearchCV
            param_grid = {'n_neighbors': [3, 5, 7, 9, 11]}
            knn = KNeighborsClassifier()
            print("Tuning KNN hyperparameters using GridSearchCV...")
            grid_search = GridSearchCV(knn, param_grid=param_grid, cv=5, scoring='accuracy', verbose=1, n_jobs=-1)
            grid_search.fit(spectra, train_labels)

            print(f"Best KNN parameters: {grid_search.best_params_}")
            print(f"Best cross-validated accuracy: {grid_search.best_score_:.2f}")
            return grid_search.best_estimator_

        # Create and fit the KNN classifier
        knn = KNeighborsClassifier(n_neighbors=n_neighbors)
        knn.fit(spectra, train_labels)

        return knn

    def evaluate_knn_classifier(self, knn, labels="genus"):
        # Evaluate the KNN classifier on the test data
        spectra = np.array([entry['spectrum'] for entry in self.test_data_pca])
        true_labels = [entry['genus_label'] if labels == "genus" else entry['genus_species_label'] for entry in self.test_data_pca]

        # Predict the labels using the trained KNN classifier
        predicted_labels = knn.predict(spectra)
        correct_predictions = sum([1 for true, pred in zip(true_labels, predicted_labels) if true == pred])
        accuracy = correct_predictions / len(true_labels)
        print(f"KNN Classifier Accuracy: {accuracy:.2f}")

        return accuracy, predicted_labels, true_labels
    
    def plot_data_distribution(self):
        # Plot the distribution of genus_species labels in train and test data
        train_labels = [entry['genus_species_label'] for entry in self.train_data_smote]
        test_labels = [entry['genus_species_label'] for entry in self.test_data]

        train_counter = Counter(train_labels)
        test_counter = Counter(test_labels)

        # Combine all labels to ensure consistent plotting
        all_labels = sorted(set(train_labels + test_labels))
        train_counts = [train_counter[label] for label in all_labels]
        test_counts = [test_counter[label] for label in all_labels]

        plt.figure(figsize=(10, 6))
        bar_width = 0.35
        bar_positions = np.arange(len(all_labels))
        plt.bar(bar_positions, train_counts, bar_width, label='Train')
        plt.bar(bar_positions + bar_width, test_counts, bar_width, label='Test')
        plt.xlabel('Genus+Species Label')
        plt.ylabel('Number of Samples')
        plt.title('Distribution of Genus+Species Labels in Train and Test Data')
        plt.xticks(bar_positions + bar_width / 2, all_labels, rotation=90)
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

# # Grid-search PCA+ LASSO
# # Define grid search parameters
# n_components_list = [10, 20, 30, 40, 50, 70, 100, 150]
# alpha_list = [0.1, 0.05, 0.01, 0.005, 0.001]

# # Run grid search for PCA + LASSO
# results_df = fungus_identifier.grid_search_pca_lasso(n_components_list, alpha_list)

# # Save or display results
# results_df.to_csv("pca_lasso_results.csv", index=False)
# print("Grid search results saved to pca_lasso_results.csv.")

# Plot data distribution
fungus_identifier.plot_data_distribution()

print("====================== GENUS LEVEL CLASSIFIERS ======================")
# Train the Naive Classifier
print("Training Naive Classifier...")
fungus_identifier.naive_classifier(labels="genus", use_weights=False)

# print("Tuning Naive Classifier...")
# best_naive_config = fungus_identifier.tune_naive_classifier(labels="genus")

# print(f"Best Naive Classifier Configuration: {best_naive_config}")

# Evaluate the Naive Classifier
print("Evaluating Naive Classifier...")
naive_accuracy = fungus_identifier.evaluate_naive_classifier(labels="genus", metric="euclidean")

# Train and evaluate a KNN Classifier
print("Training KNN Classifier...")
knn = fungus_identifier.knn_classifier(n_neighbors=3, labels="genus", tune=False)

print("Evaluating KNN Classifier...")
knn_accuracy = fungus_identifier.evaluate_knn_classifier(knn, labels="genus")

print("====================== GENUS SPECIES LEVEL CLASSIFIERS ======================")
# Train the Naive Classifier
print("Training Naive Classifier...")
fungus_identifier.naive_classifier(labels="genus_species", use_weights=False)

# print("Tuning Naive Classifier...")
# best_naive_config = fungus_identifier.tune_naive_classifier(labels="genus_species")

# print(f"Best Naive Classifier Configuration: {best_naive_config}")

# Evaluate the Naive Classifier
print("Evaluating Naive Classifier...")
naive_accuracy, naive_pred, naive_true = fungus_identifier.evaluate_naive_classifier(labels="genus_species", metric="cosine")

# Plot the accuracy per label for the Naive Classifier
fungus_identifier.plot_accuracy_per_label(naive_true, naive_pred, model_name="Naive")

# Train and evaluate a KNN Classifier
print("Training KNN Classifier...")
knn = fungus_identifier.knn_classifier(n_neighbors=3, labels="genus_species", tune=False)

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
