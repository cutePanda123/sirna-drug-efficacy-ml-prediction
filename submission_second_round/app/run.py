import lightgbm as lgb
import pandas as pd
import glob
import sys
from utils import siRNA_feat_builder, get_latest_model_file_name

is_docker_env = True
base_path = "/" if is_docker_env else "../"

model_file_path = get_latest_model_file_name(f"{base_path}model/")
loaded_model = lgb.Booster(model_file=model_file_path)

# Load testing data
csv_files = glob.glob(f"{base_path}tcdata/*.csv")
if len(csv_files) == 0:
    print("No CSV file found in the folder.")
    sys.exit()

csv_file = csv_files[0]
df_submit = pd.read_csv(csv_file)
df = pd.concat([df_submit], axis=0).reset_index(drop=True)

from feature_category import gene_target_symbol_name_category, gene_target_ncbi_id_category, gene_target_species_category, Transfection_method_category, Duration_after_transfection_h_category, cell_line_donor_category

df_gene_target_symbol_name = pd.get_dummies(df.gene_target_symbol_name.astype(pd.CategoricalDtype(categories=gene_target_symbol_name_category)))
df_gene_target_symbol_name.columns = [
    f"feat_gene_target_symbol_name_{c}" for c in df_gene_target_symbol_name.columns
]

df_gene_target_ncbi_id = pd.get_dummies(df.gene_target_ncbi_id.astype(pd.CategoricalDtype(categories=gene_target_ncbi_id_category)))
df_gene_target_ncbi_id.columns = [
    f"feat_gene_target_ncbi_id_{c}" for c in df_gene_target_ncbi_id.columns
]

df_gene_target_species = pd.get_dummies(df.gene_target_species.astype(pd.CategoricalDtype(categories=gene_target_species_category)))
df_gene_target_species.columns = [
    f"feat_gene_target_species_{c}" for c in df_gene_target_species.columns
]

siRNA_duplex_id_values = df.siRNA_duplex_id.str.split("-|\.").str[1].fillna(0).astype("int")
siRNA_duplex_id_values = (siRNA_duplex_id_values - siRNA_duplex_id_values.min()) / (
    #siRNA_duplex_id_values.max() - siRNA_duplex_id_values.min()
    1810839 - 62934
)
df_siRNA_duplex_id = pd.DataFrame(siRNA_duplex_id_values)

df_cell_line_donor = pd.get_dummies(df.cell_line_donor.astype(pd.CategoricalDtype(categories=cell_line_donor_category)))
df_cell_line_donor.columns = [
    f"feat_cell_line_donor_{c}" for c in df_cell_line_donor.columns
]
df_cell_line_donor["feat_cell_line_donor_hepatocytes"] = (
    (df.cell_line_donor.str.contains("Hepatocytes")).fillna(False).astype("int")
)
df_cell_line_donor["feat_cell_line_donor_cells"] = (
    df.cell_line_donor.str.contains("Cells").fillna(False).astype("int")
)

df_siRNA_concentration = df.siRNA_concentration.to_frame()
df_Transfection_method = pd.get_dummies(df.Transfection_method.astype(pd.CategoricalDtype(categories=Transfection_method_category)))
df_Transfection_method.columns = [
    f"feat_Transfection_method_{c}" for c in df_Transfection_method.columns
]
df_Duration_after_transfection_h = pd.get_dummies(df.Duration_after_transfection_h.astype(pd.CategoricalDtype(categories=Duration_after_transfection_h_category)))
df_Duration_after_transfection_h.columns = [
    f"feat_Duration_after_transfection_h_{c}"
    for c in df_Duration_after_transfection_h.columns
]

prepared_data = pd.concat(
    [
        df_gene_target_symbol_name,
        df_gene_target_ncbi_id,
        df_gene_target_species,
        df_siRNA_duplex_id,
        df_cell_line_donor,
        df_siRNA_concentration,
        df_Transfection_method,
        df_Duration_after_transfection_h,
        siRNA_feat_builder(df.siRNA_sense_seq, False),
        siRNA_feat_builder(df.siRNA_antisense_seq, True),
        df.iloc[:, -1].to_frame(),
    ],
    axis=1,
)

eval_data = prepared_data.iloc[:, :-1]
y_pred = loaded_model.predict(eval_data)
y_pred[y_pred > 100] = 100
y_pred[y_pred < 0] = 0

df_submit["mRNA_remaining_pct"] = y_pred

filename = f"{base_path}app/submit.csv"
print(f"File saved as {filename}")
df_submit.to_csv(filename, index=False)
