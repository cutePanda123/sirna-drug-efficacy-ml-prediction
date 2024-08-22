
# Predictive Modeling Project

This repository contains a series of notebooks used for feature extraction and model training for a predictive modeling task. The project utilizes several methods, including GRU-based features and pretrained models, to generate predictions and submit results.

## Project Structure

- **current_best.ipynb**: 
  - Purpose: Generates the final submission results.
  - Requirements: 
    - `data/train_data.csv`
    - `data/sample_submission.csv`
    - `data/external_data/gru_features_predict_only.csv`
    - `data/external_data/pretrained_feature_predict.csv`
  - Output: Final submission file.

- **getting_gru_feature.ipynb**: 
  - Purpose: Generates the `gru_features_predict_only.csv` file.
  - Requirements:
    - `data/train_data.csv`
    - `data/sample_submission.csv`
    - `data/external_data/gru_weights`
  - Output: `gru_features_predict_only.csv`
  - Notes: This notebook also contains code to generate the `gru_weights`.

- **pretrained_feature_predict.ipynb**: 
  - Purpose: Generates the `pretrained_feature_predict.csv` data file.
  - Requirements:
    - `data/train_data.csv`
    - `data/sample_submission.csv`
  - Output:
    - `pretrained_feature_predict.csv`
    - `rinalmo_features.csv`
    - `mrnafm_features.csv`

- **pretrained_model_features.ipynb**: 
  - Purpose: Uses two pretrained models to generate features.
  - Requirements:
    - `data/train_data.csv`
    - `data/sample_submission.csv`
  - Output:
    - `rinalmo_features.csv`
    - `mrnafm_features.csv`

## Setup Instructions

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/your-repo-name.git
    ```

2. Install the necessary dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Place the required data files in the `data` folder.

4. Run the notebooks in the following order:
    1. `pretrained_model_features.ipynb`
    2. `pretrained_feature_predict.ipynb`
    3. `getting_gru_feature.ipynb`
    4. `current_best.ipynb`

5. After running the notebooks, your final submission file will be ready.

## Contributing

Feel free to open issues or submit pull requests for any bugs or improvements.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.











### How to run

torch: 2.0.0
python: 3.11.7
OS: macos 14.6.1

- **current_best.ipynb**: This notebook is used to generate the final submission results. To run it, you need the following files in the `data` folder: `train_data.csv`, `sample_submission.csv`, `gru_features_predict_only.csv`, and `pretrained_feature_predict.csv`.

- **getting_gru_feature.ipynb**: This notebook generates the `gru_features_predict_only.csv` file. It requires `train_data.csv`, `sample_submission.csv`, and `gru_weights` to be present in the `data` folder. The notebook also includes the code for generating the `gru_weights`.

- **pretrained_feature_predict.ipynb**: This notebook is used to generate the `pretrained_feature_predict.csv` data file. It requires `train_data.csv` and `sample_submission.csv` in the `data` folder. After running, it will generate two additional files: `rinalmo_features.csv` and `mrnafm_features.csv`.

- **pretrained_model_features.ipynb**: This notebook utilizes two pretrained models to generate features. After running, it will produce two data files: `rinalmo_features.csv` and `mrnafm_features.csv`.
