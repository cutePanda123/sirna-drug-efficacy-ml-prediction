# siRNA Drug Efficacy Prediction

This repository contains the code, datasets, and resources necessary to train a machine learning model for predicting the efficacy of siRNA-based drugs. The project focuses on developing and optimizing models, covering aspects such as data preparation, feature engineering, hyperparameter tuning, and more.

## Dependencies
- **Operating System**: macOS 14.6.1
- **Python**: 3.11.7
- **Packages**:
  - torch: 2.0.0
  - pandas: 2.1.3
  - lightgbm: 4.5.0
  - numpy: 1.24.3
  - scikit-learn: 1.5.1
  - tqdm: 4.66.1
  - rich: 13.7.0
  - multimolecule: 0.0.4
  - chardet: 5.2.0
  - datetime: 5.5

## How to Train
- Run train.ipynb file in Jupyter Notebook.

## How to Run
- Build image: docker build -t sirna-ml-prediction-image
- Run image: docker run sirna-ml-prediction-image:latest

## Scoring
- The model trained with the current solution achieved a score of 0.6017. Although our best score was 0.6072, we’ve exhausted all ways and haven't been able to reproduce that result. Therefore, we've decided to submit this solution with the second-best score. 

## Project Structure

- **`/app/training_code/best_hyper_parameters.json`**
  - **Purpose**: Saves the best hyperparameters identified through the hyperparameter search.

- **`/app/training_code/best_hyper_parameters.json`**
  - **Purpose**: Saves the best hyperparameters identified through the hyperparameter search.

- **`/data/train_data.csv`**
  - **Purpose**: Saves training data for the model.

- **`/app/training_code/load_data.ipynb`**
  - **Purpose**: Loads training data.
  - **Requirements**: 
    - `/data/train_data.csv`

- **`/app/training_code/build_features.ipynb`**
  - **Purpose**: Performs feature engineering, including feature extraction and selection.
  - **Requirements**: 
    - `/app/training_code/load_data.ipynb`
    - `/data/train_data.csv`

- **`/app/training_code/train.ipynb`**
  - **Purpose**: Train model and save it in a file.
  - **Requirements**: 
    - `/app/training_code/load_data.ipynb`
    - `/app/training_code/build_features.ipynb`
    - `/data/train_data.csv`
    - `/app/training_code/best_hyper_parameters.json`
  - **Output**: Model file.