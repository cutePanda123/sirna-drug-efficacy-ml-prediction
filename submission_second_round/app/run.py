print("please write your program here...")


import lightgbm as lgb
import pandas as pd
import glob
import os
import sys

################################################
# 特征工程
################################################

def siRNA_feat_builder(s: pd.Series, anti: bool = False):
    name = "anti" if anti else "sense"
    df = s.to_frame()
    df[f"feat_siRNA_{name}_seq_len"] = s.str.len()
    for pos in [0, -1]:
        for c in list("AUGC"):
            df[f"feat_siRNA_{name}_seq_{c}_{'front' if pos == 0 else 'back'}"] = (
                s.str[pos] == c
            )
    df[f"feat_siRNA_{name}_seq_pattern_1"] = s.str.startswith("AA") & s.str.endswith(
        "UU"
    )
    df[f"feat_siRNA_{name}_seq_pattern_2"] = s.str.startswith("GA") & s.str.endswith(
        "UU"
    )
    df[f"feat_siRNA_{name}_seq_pattern_3"] = s.str.startswith("CA") & s.str.endswith(
        "UU"
    )
    df[f"feat_siRNA_{name}_seq_pattern_4"] = s.str.startswith("UA") & s.str.endswith(
        "UU"
    )
    df[f"feat_siRNA_{name}_seq_pattern_5"] = s.str.startswith("UU") & s.str.endswith(
        "AA"
    )
    df[f"feat_siRNA_{name}_seq_pattern_6"] = s.str.startswith("UU") & s.str.endswith(
        "GA"
    )
    df[f"feat_siRNA_{name}_seq_pattern_7"] = s.str.startswith("UU") & s.str.endswith(
        "CA"
    )
    df[f"feat_siRNA_{name}_seq_pattern_8"] = s.str.startswith("UU") & s.str.endswith(
        "UA"
    )
    df[f"feat_siRNA_{name}_seq_pattern_9"] = s.str[1] == "A"
    df[f"feat_siRNA_{name}_seq_pattern_10"] = s.str[-2] == "A"
    df[f"feat_siRNA_{name}_seq_pattern_GC_ratio_0"] = (
        s.str.count("G") + s.str.count("C")
    ) / s.str.len()

    df[f"feat_siRNA_{name}_len_range"] = (s.str.len() >= 21) & (s.str.len() <= 25)

    GC_ratio_1 = (s.str.count("G") + s.str.count("C")) / s.str.len()
    df[f"feat_siRNA_{name}_GC_ratio_1"] = (GC_ratio_1 >= 0.31) & (GC_ratio_1 <= 0.58)

    GC_ratio_2 = (s.str[1:7].str.count("G") + s.str[1:7].str.count("C")) / s.str[1:7].str.len()
    df[f"feat_siRNA_{name}_GC_ratio_2"] = (GC_ratio_2 == 0.19)

    GC_ratio_3 = (s.str[7:18].str.count("G") + s.str[7:18].str.count("C")) / s.str[7:18].str.len()
    df[f"feat_siRNA_{name}_GC_ratio_3"] = (GC_ratio_3 == 0.52)

    return df.iloc[:, 1:]

is_docker_env = False
base_path = "/" if is_docker_env else "./"

loaded_model = lgb.Booster(model_file=f"{base_path}model/lightgbm_model.txt")

# Load testing data
csv_files = glob.glob(f"{base_path}tcdata/*.csv")
if len(csv_files) == 0:
    print("No CSV file found in the folder.")
    sys.exit()

csv_file = csv_files[0]
df_submit = pd.read_csv(csv_file)
df = pd.concat([df_submit], axis=0).reset_index(drop=True)

print(df.tail(3))

# Prepare testing data 
df = pd.read_csv(f"{base_path}app/features/gru_features_predict_only.csv", index_col=0).tail(4876).merge(
    df,
    on='id'
)

# Features from other pretrained model
df = pd.read_csv(f"{base_path}app/features/pretrained_feature_predict.csv", index_col=0).tail(4876).merge(
    df,
    on='id'
)

print(df.tail(3))

df_publication_id = pd.get_dummies(df.publication_id)
df_publication_id.columns = [
    f"feat_publication_id_{c}" for c in df_publication_id.columns
]
df_gene_target_symbol_name = pd.get_dummies(df.gene_target_symbol_name)
df_gene_target_symbol_name.columns = [
    f"feat_gene_target_symbol_name_{c}" for c in df_gene_target_symbol_name.columns
]
df_gene_target_ncbi_id = pd.get_dummies(df.gene_target_ncbi_id)
df_gene_target_ncbi_id.columns = [
    f"feat_gene_target_ncbi_id_{c}" for c in df_gene_target_ncbi_id.columns
]
df_gene_target_species = pd.get_dummies(df.gene_target_species)
df_gene_target_species.columns = [
    f"feat_gene_target_species_{c}" for c in df_gene_target_species.columns
]
siRNA_duplex_id_values = df.siRNA_duplex_id.str.split("-|\.").str[1].astype("int")
siRNA_duplex_id_values = (siRNA_duplex_id_values - siRNA_duplex_id_values.min()) / (
    siRNA_duplex_id_values.max() - siRNA_duplex_id_values.min()
)
df_siRNA_duplex_id = pd.DataFrame(siRNA_duplex_id_values)
df_cell_line_donor = pd.get_dummies(df.cell_line_donor)
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
df_Transfection_method = pd.get_dummies(df.Transfection_method)
df_Transfection_method.columns = [
    f"feat_Transfection_method_{c}" for c in df_Transfection_method.columns
]
df_Duration_after_transfection_h = pd.get_dummies(df.Duration_after_transfection_h)
df_Duration_after_transfection_h.columns = [
    f"feat_Duration_after_transfection_h_{c}"
    for c in df_Duration_after_transfection_h.columns
]

df_GRU_pred = df[['GRU_predict']]
df_pretrained_pred = df[['Pretrained_feature_predict']]

prepared_data = pd.concat(
    [
        df_publication_id,
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
        df_GRU_pred,
        df_pretrained_pred,
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


# TODO: clean the testing change
tmp = pd.read_csv(f"{base_path}app/submit.csv")
print(tmp.head(3))