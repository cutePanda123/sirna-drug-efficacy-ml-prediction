import pandas as pd

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