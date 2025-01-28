# Sentiment Analysis of Amazon Office Products Reviews

### ⭐⭐⭐⭐⭐ Positive Review

> **"Nice laser pointer. Bought it almost a year ago now and it still shines just as brightly as it did the first day. My cat loves it!"**

**Rating**: 5.0/5.0

### ⭐⭐ Negative Review

> **"This shredder is extremely loud. Good for about 4-5 papers at a time. You may want to look for something better, or let me know if you want to buy mine :("**

**Rating**: 2.0/5.0

---

- This project demonstrates a complete text classification ML pipeline using Amazon reviews from the Office Products category. The primary goal is to predict the sentiment (positive or negative) of a given product review. Below is a high-level overview of each step, from data acquisition to model comparison.

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Data Preparation](#data-preparation)
4. [Data Exploration & Visualization](#data-exploration--visualization)
5. [Data Preprocessing](#data-preprocessing)
6. [Modeling](#modeling)
7. [Results Comparison](#results-comparison)
8. [How to Run](#how-to-run)
9. [Acknowledgments](#acknowledgments)

## Overview

- **Data Source**: Amazon Reviews (Office Products category)
- **Techniques**: Data cleaning, TF-IDF feature extraction, and multiple machine learning classifiers (Perceptron, SVM, Logistic Regression, and Multinomial Naive Bayes)
- **Goal**: Predict whether a review is positive (rating > 3) or negative (rating < 3)

## Data Preparation

### Download Dataset

- The Amazon Office Products Reviews are downloaded and stored locally.
- The file is then read into a DataFrame, keeping only the `review_body` and `star_rating` columns.

### Clean Invalid Ratings

- Convert ratings to integers and remove invalid or missing entries.
- Results in a cleaned DataFrame with only valid reviews (ratings between 1 and 5).

## Data Exploration & Visualization

- Randomly sample and print a few reviews.
- Calculate and display:
  - Value counts of each rating
  - Percentages, mean, median, and standard deviation of ratings
- Plot the distribution of star ratings.

This step provides an understanding of the dataset’s rating distribution and typical review content.

## Data Preprocessing

### Create Positive/Negative Labels

- **Ratings > 3** → 1 (positive)
- **Ratings < 3** → 0 (negative)
- **Ratings = 3** → 2 (neutral, which is then removed for binary classification)

### Balance & Condense Dataset

- Sample 100,000 positive and 100,000 negative reviews to ensure class balance.

### Text Cleaning

1. Convert text to lowercase.
2. Expand contractions.
3. Remove HTML tags, URLs, non-alphabetical characters, and extra spaces.

### Stop Word Removal & Lemmatization

- Remove stopwords (e.g., “the”, “and”, “a”).
- Lemmatize words (e.g., “builds” → “build”).

The resulting cleaned text is stored in a new column.

### Train-Test Split

- Stratified 80/20 split to maintain balanced class proportions.

### TF-IDF Feature Extraction

- Convert cleaned text into a matrix of TF-IDF features.

## Modeling

Multiple machine learning classifiers are trained and compared:

1. **Perceptron**
   - A linear classifier that updates weights on misclassifications.
2. **Support Vector Machine (SVM)**
   - Uses a linear kernel to find the best separating hyperplane with maximum margin.
3. **Logistic Regression**
   - Fits a sigmoid (logistic) function to predict probabilities for each class.
4. **Multinomial Naive Bayes**
   - Probabilistic approach assuming features are conditionally independent.

### Shared Functions

- Evaluation functions calculate and print classification metrics.
- Visualization functions display a comparison chart of metrics for all models on both training and testing sets.

## Results Comparison

After training, each model’s performance is evaluated on both training and testing data. The following metrics are computed and compared:

### Accuracy

**Definition**: Accuracy measures the proportion of correctly predicted instances (both positive and negative) out of the total instances.  
**Formula**:  
![Accuracy](<https://latex.codecogs.com/png.latex?\text{Accuracy}=\frac{\text{True%20Positives%20(TP)}+\text{True%20Negatives%20(TN)}}{\text{Total%20Instances}}>)  
**Use Case**: Useful when the dataset is balanced (similar numbers of positive and negative instances).

### Precision

**Definition**: Precision measures the proportion of correctly predicted positive instances out of all instances predicted as positive (i.e., of all the postive predictions made by the model, what percentage of them are truly postive?).  
**Formula**:  
![Precision](<https://latex.codecogs.com/png.latex?\text{Precision}=\frac{\text{True%20Positives%20(TP)}}{\text{True%20Positives%20(TP)}+\text{False%20Positives%20(FP)}}>)  
**Use Case**: Important in scenarios where minimizing false positives is critical (e.g., spam detection).

### Recall

**Definition**: Recall (or Sensitivity) measures the proportion of correctly predicted positive instances out of all actual positive instances (i.e., out of all the truly postive instances that the model was tested on, what percentage of them did the model correctly identify as positive?).  
**Formula**:  
![Recall](<https://latex.codecogs.com/png.latex?\text{Recall}=\frac{\text{True%20Positives%20(TP)}}{\text{True%20Positives%20(TP)}+\text{False%20Negatives%20(FN)}}>)  
**Use Case**: Crucial in scenarios where minimizing false negatives is important (e.g., disease detection).

### F1-Score

**Definition**: F1-Score is the harmonic mean of Precision and Recall, providing a single metric that balances both.  
**Formula**:  
![Equation](https://latex.codecogs.com/png.latex?\text{F1-Score}=2\cdot\frac{\text{Precision}\cdot\text{Recall}}{\text{Precision}+\text{Recall}})  
**Use Case**: Useful when there is an imbalance between classes and you want a trade-off between Precision and Recall.

These allow quick comparison of how well each classifier performs in predicting sentiment on unseen reviews.
