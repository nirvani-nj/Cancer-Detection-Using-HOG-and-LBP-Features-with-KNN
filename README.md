# Cancer-Detection-Using-HOG-and-LBP-Features-with-KNN

Project Overview
This project implements a cancer detection system using Histogram of Oriented Gradients (HOG) and Local Binary Pattern (LBP) features with K-Nearest Neighbors (KNN) classifiers. The goal is to classify histopathology images from the Histopathologic Cancer Detection dataset as cancerous (malignant) or non-cancerous (benign) using texture-based feature extraction and ensemble learning techniques.

Dataset
The project uses the Histopathologic Cancer Detection dataset from Kaggle, which contains 220,025 training images of lymph node tissue. Each 96x96 pixel image is labeled as containing metastatic tissue (1) or not (0). For this implementation, we used a random sample of 10,000 images to balance computational efficiency with model performance.

Key Features
Dual Feature Extraction

HOG (Histogram of Oriented Gradients) - Captures gradient structures in images

LBP (Local Binary Patterns) - Extracts texture patterns for classification

Two KNN Classifiers

One trained on HOG features

One trained on LBP features

Feature Fusion Techniques

Weighted Voting - Combines predictions based on model confidence

Stacking Classifier - Uses logistic regression as a meta-model

Performance Evaluation

Accuracy scores

Confusion matrices

ROC curves and AUC scores

Results Comparison
Model	Accuracy	AUC Score
HOG + KNN	53.6%	0.54
LBP + KNN	72.6%	0.73
Weighted Fusion	72.6%	0.73
Stacking	59.0%	0.59
Key Insights
LBP performed significantly better than HOG (72.6% vs 53.6%)

Fusion maintained LBP's performance but did not improve it

Stacking underperformed, likely due to high correlation between models

How It Works
Load and Preprocess Images from dataset

Grayscale conversion

Random sampling for faster experimentation

Feature Extraction

HOG: Computes gradient orientations

LBP: Encodes local texture patterns

Train KNN Models

Separate classifiers for HOG and LBP features

Fusion and Evaluation

Weighted voting based on model confidence

Stacking with logistic regression

Future Improvements
Hyperparameter tuning for KNN (optimal n_neighbors)

Alternative fusion methods (e.g., neural networks)

Deep learning comparison (CNNs vs traditional features)

Better meta-learner for stacking (e.g., SVM, Random Forest)

Conclusion
This project implemented HOG and LBP feature extraction techniques for cancer detection, trained KNN classifiers on each feature set, and explored prediction fusion through weighted averaging and stacking.

Key Findings:

->LBP Outperformed HOG – The LBP-based KNN achieved 72.6% accuracy, significantly better than HOG’s 53.6%, suggesting that LBP captures more discriminative texture patterns in this dataset.

->Fusion Matched LBP Alone – The weighted fusion model also achieved 72.6% accuracy, preserving LBP’s performance while incorporating HOG’s weaker signals.

->No Performance Gain from Fusion – HOG’s lower accuracy may have introduced noise instead of providing complementary information, explaining the lack of improvement with fusion.

->Stacking Underperformed – The stacking classifier (using logistic regression as the meta-learner) dropped to 59% accuracy, likely due to: High correlation between the base KNN models. Logistic regression possibly not being the most suitable choice as a meta-learner in this case.

