import numpy as np
import pandas as pd

LIST_TYPES = (list, np.ndarray)


def unlist_columns(df):
    """Collapse list-typed columns back to scalars: single-element lists unwrap, longer ones join with '|'.

    Parquet round-trips list cells back as numpy.ndarray rather than list, so both are treated as list-like.
    """
    out = df.copy()
    for col in out.columns:
        if out[col].map(lambda v: isinstance(v, LIST_TYPES)).any():
            out[col] = out[col].map(
                lambda v: (v[0] if len(v) == 1 else "|".join(v)) if isinstance(v, LIST_TYPES) else v
            )
    return out


def cast_col(series, kind):
    if kind == "string":
        return series.astype("string")
    if kind == "float":
        return pd.to_numeric(series, errors="coerce").astype("float64")
    if kind == "int":
        return pd.to_numeric(series, errors="coerce").astype("Int64")
    if kind == "bool":
        return series.astype("boolean")
    raise ValueError(kind)


def coalesce_cols(df, cols):
    result = df[cols[0]]
    for col in cols[1:]:
        result = result.combine_first(df[col])
    return result


def build_atleast_two_technologies(seer_df, olink_df, soma_df):
    merged = (
        seer_df.merge(olink_df, on="Uni_Prot_ID", how="outer", suffixes=("", "_olink"))
               .merge(soma_df, on="Uni_Prot_ID", how="outer", suffixes=("", "_soma"))
    )
    merged = merged[
        (merged["Seer_ID"].notna() & merged["Olink_ID"].notna()) |
        (merged["Seer_ID"].notna() & merged["Soma_ID"].notna()) |
        (merged["Olink_ID"].notna() & merged["Soma_ID"].notna())
    ].copy()
    merged["gene_name"] = cast_col(coalesce_cols(merged, ["gene_name", "gene_name_olink", "gene_name_soma"]), "string")
    merged["length"] = cast_col(coalesce_cols(merged, ["length", "length_olink", "length_soma"]), "int")
    merged["covered_PINNACLE"] = cast_col(coalesce_cols(merged, ["covered_PINNACLE", "covered_PINNACLE_olink", "covered_PINNACLE_soma"]), "bool")
    merged["capable_for_NN_integration"] = cast_col(coalesce_cols(merged, ["capable_for_NN_integration", "capable_for_NN_integration_olink", "capable_for_NN_integration_soma"]), "bool")
    return merged[["Uni_Prot_ID", "gene_name", "length", "covered_PINNACLE", "capable_for_NN_integration", "Seer_ID", "Olink_ID", "Soma_ID"]]

def true_positives(df, prefix, spread, minimum, medium):
    well_covered = (df[f"{prefix}_spread"] <= spread) & (df[f"{prefix}_median_pred"] >= medium) & (df[f"{prefix}_p25_pred"] >= minimum)
    fraction = well_covered.groupby(df["Uni_Prot_ID"]).transform("mean")
    return df[fraction >= 1].copy()

def true_negatives(df, prefix, spread, minimum, medium):
    poorly_covered = (df[f"{prefix}_spread"] > spread) | (df[f"{prefix}_median_pred"] < medium) | (df[f"{prefix}_p25_pred"] < minimum)
    fraction = poorly_covered.groupby(df["Uni_Prot_ID"]).transform("mean")
    return df[fraction >= 1].copy()

def main():

    spread = 0.4
    minimum = 0.2
    medium = 0.4

    poorSpread = 0.8
    poorMinimum = 0.02
    poorMedium = 0.1

    seer = pd.read_parquet("/data/PHURI-Langenberg/people/Nat/Protein-Prediction/Data/Fixed/ST1_Seer_Cleaned.parquet")
    olink = pd.read_parquet("/data/PHURI-Langenberg/people/Nat/Protein-Prediction/Data/Fixed/ST2_Olink_Cleaned.parquet")
    soma = pd.read_parquet("/data/PHURI-Langenberg/people/Nat/Protein-Prediction/Data/Fixed/ST3_Soma_Cleaned.parquet")

    seer = unlist_columns(seer)
    olink = unlist_columns(olink)
    soma = unlist_columns(soma)

    seer_soma = seer.merge(soma, on="Uni_Prot_ID", how="inner")[["Uni_Prot_ID", "Seer_ID", "Soma_ID"]]
    seer_olink = seer.merge(olink, on="Uni_Prot_ID", how="inner")[["Uni_Prot_ID", "Seer_ID", "Olink_ID"]]
    olink_soma = olink.merge(soma, on="Uni_Prot_ID", how="inner")[["Uni_Prot_ID", "Olink_ID", "Soma_ID"]]
    all_technologies = (
        seer.merge(olink, on="Uni_Prot_ID", how="inner", suffixes=("", "_olink"))
            .merge(soma, on="Uni_Prot_ID", how="inner", suffixes=("", "_soma"))
    )

    seer_well_covered = true_positives(seer, "seer", spread, minimum, medium)
    soma_well_covered = true_positives(soma, "soma", spread, minimum, medium)
    olink_well_covered = true_positives(olink, "olink", spread, minimum, medium)

    seer_poorly_covered = true_negatives(seer, "seer", poorSpread, poorMinimum, poorMedium)
    soma_poorly_covered = true_negatives(soma, "soma", poorSpread, poorMinimum, poorMedium)
    olink_poorly_covered = true_negatives(olink, "olink", poorSpread, poorMinimum, poorMedium)

    seer_soma_well_covered = seer_well_covered.merge(soma_well_covered, on="Uni_Prot_ID", how="inner")
    seer_olink_well_covered = seer_well_covered.merge(olink_well_covered, on="Uni_Prot_ID", how="inner")
    olink_soma_well_covered = olink_well_covered.merge(soma_well_covered, on="Uni_Prot_ID", how="inner")

    seer_soma_poorly_covered = seer_poorly_covered.merge(soma_poorly_covered, on="Uni_Prot_ID", how="inner")
    seer_olink_poorly_covered = seer_poorly_covered.merge(olink_poorly_covered, on="Uni_Prot_ID", how="inner")
    olink_soma_poorly_covered = olink_poorly_covered.merge(soma_poorly_covered, on="Uni_Prot_ID", how="inner")

    atleast_two_technologies = build_atleast_two_technologies(seer, olink, soma)
    atleast_two_technologies_well_covered = build_atleast_two_technologies(seer_well_covered, olink_well_covered, soma_well_covered)
    atleast_two_technologies_poorly_covered = build_atleast_two_technologies(seer_poorly_covered, olink_poorly_covered, soma_poorly_covered)

    all_technologies_well_covered = seer_well_covered.merge(olink_well_covered, on="Uni_Prot_ID", how="inner", suffixes=("", "_olink")).merge(soma_well_covered, on="Uni_Prot_ID", how="inner", suffixes=("", "_soma"))
    all_technologies_poorly_covered = seer_poorly_covered.merge(olink_poorly_covered, on="Uni_Prot_ID", how="inner", suffixes=("", "_olink")).merge(soma_poorly_covered, on="Uni_Prot_ID", how="inner", suffixes=("", "_soma"))

    seer_count = seer["Uni_Prot_ID"].nunique(dropna=False)
    seer_well_covered_count = seer_well_covered["Uni_Prot_ID"].nunique(dropna=False)
    olink_count = olink["Uni_Prot_ID"].nunique(dropna=False)
    olink_well_covered_count = olink_well_covered["Uni_Prot_ID"].nunique(dropna=False)
    soma_count = soma["Uni_Prot_ID"].nunique(dropna=False)
    soma_well_covered_count = soma_well_covered["Uni_Prot_ID"].nunique(dropna=False)

    seer_poorly_covered_count = seer_poorly_covered["Uni_Prot_ID"].nunique(dropna=False)
    olink_poorly_covered_count = olink_poorly_covered["Uni_Prot_ID"].nunique(dropna=False)
    soma_poorly_covered_count = soma_poorly_covered["Uni_Prot_ID"].nunique(dropna=False)

    seer_olink_count = seer_olink[seer_olink["Seer_ID"].notna() & seer_olink["Olink_ID"].notna()]["Uni_Prot_ID"].nunique(dropna=False)
    seer_olink_well_covered_count = seer_olink_well_covered[seer_olink_well_covered["Olink_ID"].notna() & seer_olink_well_covered["Seer_ID"].notna()]["Uni_Prot_ID"].nunique(dropna=False)
    seer_soma_count = seer_soma[seer_soma["Seer_ID"].notna() & seer_soma["Soma_ID"].notna()]["Uni_Prot_ID"].nunique(dropna=False)
    seer_soma_well_covered_count = seer_soma_well_covered[seer_soma_well_covered["Soma_ID"].notna() & seer_soma_well_covered["Seer_ID"].notna()]["Uni_Prot_ID"].nunique(dropna=False)
    olink_soma_count = olink_soma[olink_soma["Olink_ID"].notna() & olink_soma["Soma_ID"].notna()]["Uni_Prot_ID"].nunique(dropna=False)
    olink_soma_well_covered_count = olink_soma_well_covered[olink_soma_well_covered["Olink_ID"].notna() & olink_soma_well_covered["Soma_ID"].notna()]["Uni_Prot_ID"].nunique(dropna=False)

    seer_olink_poorly_covered_count = seer_olink_poorly_covered[seer_olink_poorly_covered["Olink_ID"].notna() & seer_olink_poorly_covered["Seer_ID"].notna()]["Uni_Prot_ID"].nunique(dropna=False)
    seer_soma_poorly_covered_count = seer_soma_poorly_covered[seer_soma_poorly_covered["Soma_ID"].notna() & seer_soma_poorly_covered["Seer_ID"].notna()]["Uni_Prot_ID"].nunique(dropna=False)
    olink_soma_poorly_covered_count = olink_soma_poorly_covered[olink_soma_poorly_covered["Olink_ID"].notna() & olink_soma_poorly_covered["Soma_ID"].notna()]["Uni_Prot_ID"].nunique(dropna=False)

    all_technologies_count = all_technologies[all_technologies["Seer_ID"].notna() & all_technologies["Olink_ID"].notna() & all_technologies["Soma_ID"].notna()]["Uni_Prot_ID"].nunique(dropna=False)
    all_technologies_well_covered_count = all_technologies_well_covered[all_technologies_well_covered["Seer_ID"].notna() & all_technologies_well_covered["Olink_ID"].notna() & all_technologies_well_covered["Soma_ID"].notna()]["Uni_Prot_ID"].nunique(dropna=False)
    all_technologies_poorly_covered_count = all_technologies_poorly_covered[all_technologies_poorly_covered["Seer_ID"].notna() & all_technologies_poorly_covered["Olink_ID"].notna() & all_technologies_poorly_covered["Soma_ID"].notna()]["Uni_Prot_ID"].nunique(dropna=False)

    unique_atleast_two_technologies_count = atleast_two_technologies["Uni_Prot_ID"].nunique(dropna=False)
    unique_atleast_two_technologies_well_covered_count = atleast_two_technologies_well_covered["Uni_Prot_ID"].nunique(dropna=False)
    unique_atleast_two_technologies_poorly_covered_count = atleast_two_technologies_poorly_covered["Uni_Prot_ID"].nunique(dropna=False)

    unique_atleast_two_technologies_capable_count = atleast_two_technologies[atleast_two_technologies["capable_for_NN_integration"].fillna(False)]["Uni_Prot_ID"].nunique(dropna=False)
    unique_atleast_two_technologies_well_covered_capable_count = atleast_two_technologies_well_covered[atleast_two_technologies_well_covered["capable_for_NN_integration"].fillna(False)]["Uni_Prot_ID"].nunique(dropna=False)
    unique_atleast_two_technologies_poorly_covered_capable_count = atleast_two_technologies_poorly_covered[atleast_two_technologies_poorly_covered["capable_for_NN_integration"].fillna(False)]["Uni_Prot_ID"].nunique(dropna=False)

    print(f"-"*50)
    print(f"Filters Used Well Covered:")
    print(f"-"*50)
    print(f"Spread: {spread}")
    print(f"Median: {medium}")
    print(f"Minimum: {minimum}\n")

    print(f"-"*50)
    print(f"Proteins Covered by each Technology (Well Covered : Total Proteins):")
    print(f"-"*50)
    print(f"Seer: {seer_well_covered_count} : {seer_count}")
    print(f"Olink: {olink_well_covered_count} : {olink_count}")
    print(f"Soma: {soma_well_covered_count} : {soma_count}\n")

    print(f"-"*50)
    print(f"Proteins Covered by Two Technologies (Well Covered : Total Proteins):")
    print(f"-"*50)
    print(f"Seer + Olink: {seer_olink_well_covered_count} : {seer_olink_count}")
    print(f"Seer + Soma: {seer_soma_well_covered_count} : {seer_soma_count}")
    print(f"Olink + Soma: {olink_soma_well_covered_count} : {olink_soma_count}\n")

    print(f"-"*50)
    print(f"UniqueProteins Covered by At Least Two Technologies (Well Covered : Total Proteins):")
    print(f"-"*50)
    print(f"At Least Two Technologies: {unique_atleast_two_technologies_well_covered_count} : {unique_atleast_two_technologies_count}\n")
    print(f"Capable for NN Integration (Well Covered : All): {unique_atleast_two_technologies_well_covered_capable_count} : {unique_atleast_two_technologies_capable_count}\n")

    print(f"-"*50)
    print(f"Proteins Covered by All Three Technologies (Well Covered : Total Proteins):")
    print(f"-"*50)
    print(f"Seer + Olink + Soma: {all_technologies_well_covered_count} : {all_technologies_count}")

    print(f"\n\n\n")

    print(f"-"*50)
    print(f"Filters Used Poorly Covered:")
    print(f"-"*50)
    print(f"Spread: {poorSpread}")
    print(f"Median: {poorMedium}")
    print(f"Minimum: {poorMinimum}\n")

    print(f"-"*50)
    print(f"Proteins Poorly Covered by each Technology (Poorly Covered : Total Proteins):")
    print(f"-"*50)
    print(f"Seer: {seer_poorly_covered_count} : {seer_count}")
    print(f"Olink: {olink_poorly_covered_count} : {olink_count}")
    print(f"Soma: {soma_poorly_covered_count} : {soma_count}\n")

    print(f"-"*50)
    print(f"Proteins Poorly Covered by Two Technologies (Poorly Covered : Total Proteins):")
    print(f"-"*50)
    print(f"Seer + Olink: {seer_olink_poorly_covered_count} : {seer_olink_count}")
    print(f"Seer + Soma: {seer_soma_poorly_covered_count} : {seer_soma_count}")
    print(f"Olink + Soma: {olink_soma_poorly_covered_count} : {olink_soma_count}\n")

    print(f"-"*50)
    print(f"Unique Proteins Poorly Covered by At Least Two Technologies (Poorly Covered : Total Proteins):")
    print(f"-"*50)
    print(f"At Least Two Technologies: {unique_atleast_two_technologies_poorly_covered_count} : {unique_atleast_two_technologies_count}\n")
    print(f"Capable for NN Integration (Poorly Covered : All): {unique_atleast_two_technologies_poorly_covered_capable_count} : {unique_atleast_two_technologies_capable_count}\n")

    print(f"-"*50)
    print(f"Proteins Poorly Covered by All Three Technologies (Poorly Covered : Total Proteins):")
    print(f"-"*50)
    print(f"Seer + Olink + Soma: {all_technologies_poorly_covered_count} : {all_technologies_count}")

if __name__ == "__main__":
    main()
