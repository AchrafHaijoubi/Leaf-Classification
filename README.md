# Leaf Classification Project

This project implements various machine learning algorithms for leaf classification using a dataset of leaf features. It compares the performance of different classifiers including Decision Trees, Logistic Regression, Perceptron, Support Vector Machines (SVM), Neural Networks, and a baseline "Sans Caracteres" model.

## Features

- Multiple classification algorithms
- Data preprocessing (standardization, PCA)
- Model evaluation with metrics and visualizations
- Hyperparameter tuning using GridSearchCV

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd Leaf-Classification
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the main notebook `main.ipynb` to train and evaluate the models. The notebook includes:

- Data loading and preprocessing
- Training models on raw data
- Training models on standardized data
- Training models on PCA-reduced data
- Evaluation and comparison of model performance

To run the notebook:
```bash
jupyter notebook main.ipynb
```

## Project Structure

- `main.ipynb`: Main Jupyter notebook for running the classification pipeline
- `pretraitement.py`: Data preprocessing utilities (loading, standardization, PCA)
- `evaluation.py`: Model evaluation and visualization
- `arbre_de_decision.py`: Decision Tree classifier implementation
- `regression_logisitique.py`: Logistic Regression classifier implementation
- `perceptron_classification.py`: Perceptron classifier implementation
- `svm.py`: SVM classifier implementation
- `reseau_de_neuronnes.py`: Neural Network classifier implementation
- `sans_caracteres.py`: Baseline classifier (no features)
- `train.csv`: Training dataset
- `requirements.txt`: Python dependencies

## Algorithms

The project implements the following algorithms:

1. **Sans Caracteres**: Baseline model using no features
2. **Decision Tree**: With hyperparameter tuning for criterion and splitter
3. **Logistic Regression**: With regularization parameter tuning
4. **Perceptron**: Linear classifier with learning rate tuning
5. **SVM**: Support Vector Machine with kernel and C parameter tuning
6. **Neural Networks**: Multi-layer perceptron with hidden layer size tuning

## Evaluation

Models are evaluated using:
- Accuracy
- Precision
- Recall
- F1-Score
- ROC curves per class

Results are visualized with bar plots and ROC curves.

## Dataset

The project uses `train.csv` which contains leaf features and class labels. The data undergoes preprocessing including standardization and dimensionality reduction via PCA.

## Contributing

Feel free to contribute by adding new algorithms or improving existing implementations.

## License

This project is open-source. Please check the license file for details.