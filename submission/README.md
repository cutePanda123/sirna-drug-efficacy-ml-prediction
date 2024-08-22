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


## How to Run

- Open and run the `code/main.ipynb` file in Jupyter Notebook

## Project Structure

- **`data/external_data/best_hyper_parameters.json`**
  - **Purpose**: Saves the best hyperparameters identified through the hyperparameter search.

- **`data/external_data/gru_weights`**
  - **Purpose**: Saved weights of the GRU model, which is used to generate features for downstream modeling.

- **`data/external_data/gru_features_predict_only.csv`**
  - **Purpose**: Predictions from the GRU model, which is used as a feature for downstream modeling.

- **`data/external_data/mrnafm_features.csv`**
  - **Purpose**: Embeddings generated from the pretrained mRNAFM model (https://huggingface.co/multimolecule/rnafm).

- **`data/external_data/rinalmo_features.csv`**
  - **Purpose**: Embeddings generated from the pretrained RiNALMo model (https://huggingface.co/multimolecule/rinalmo).

- **`data/external_data/pretrained_feature_predict.csv`**
  - **Purpose**: Predictions using the embeddings from mRNAFM and RiNALMo, which is then used as a feature for downstream modeling.

- **`code/main.ipynb`**
  - **Purpose**: Runs the entire process, including data loading, feature engineering, training, and predicting.
  - **Requirements**: 
    - `code/dependency_setup.ipynb`
    - `code/load_data.ipynb`
    - `code/build_features.ipynb`
    - `code/train_and_predict.ipynb`
    - `data/train_data.csv`
    - `data/sample_submission.csv`
    - `data/external_data/gru_features_predict_only.csv`
    - `data/external_data/pretrained_feature_predict.csv`
    - `data/external_data/best_hyper_parameters.json`
  - **Output**: Final submission file.

- **`code/dependency_setup.ipynb`**
  - **Purpose**: Installs all the required dependency packages.

- **`code/load_data.ipynb`**
  - **Purpose**: Loads training and evaluation data.
  - **Requirements**: 
    - `code/dependency_setup.ipynb`

- **`code/build_features.ipynb`**
  - **Purpose**: Performs feature engineering, including feature extraction and selection.
  - **Requirements**: 
    - `code/dependency_setup.ipynb`
    - `code/load_data.ipynb`

- **`code/hyper_parameter_selection.ipynb`**
  - **Purpose**: Searches for the best hyperparameters for the model.
  - **Requirements**: 
    - `code/dependency_setup.ipynb`
    - `code/load_data.ipynb`
    - `code/build_features.ipynb`
    - `data/train_data.csv`
    - `data/sample_submission.csv`
    - `data/external_data/gru_features_predict_only.csv`
    - `data/external_data/pretrained_feature_predict.csv`

- **`code/train_and_predict.ipynb`**
  - **Purpose**: Generates the final submission results.
  - **Requirements**: 
    - `code/dependency_setup.ipynb`
    - `code/load_data.ipynb`
    - `code/build_features.ipynb`
    - `data/train_data.csv`
    - `data/sample_submission.csv`
    - `data/external_data/gru_features_predict_only.csv`
    - `data/external_data/pretrained_feature_predict.csv`
    - `data/external_data/best_hyper_parameters.json`
  - **Output**: Final submission file.

- **`code/getting_gru_feature.ipynb`**
  - **Purpose**: Generates the `gru_features_predict_only.csv` file.
  - **Requirements**:
    - `data/train_data.csv`
    - `data/sample_submission.csv`
    - `data/external_data/gru_weights`
  - **Output**: `gru_features_predict_only.csv`
  - **Notes**: This notebook also contains code to generate the `gru_weights`.

- **`code/pretrained_feature_predict.ipynb`**
  - **Purpose**: Generates the `pretrained_feature_predict.csv` data file.
  - **Requirements**:
    - `data/train_data.csv`
    - `data/sample_submission.csv`
  - **Output**:
    - `pretrained_feature_predict.csv`
    - `rinalmo_features.csv`
    - `mrnafm_features.csv`

- **`code/pretrained_model_features.ipynb`**
  - **Purpose**: Uses two pretrained models to generate features.
  - **Requirements**:
    - `data/train_data.csv`
    - `data/sample_submission.csv`
  - **Output**:
    - `rinalmo_features.csv`
    - `mrnafm_features.csv`