import pandas as pd
import polars as pl
import numpy as np

def extract_true_positives(datasets, spread: float = 0.4, medium: float = 0.5, minimum: float = 0.25, coverage: int = 2):
    
    local_datasets = {}
    
    for name, dataset in datasets.items():
        d = dataset.copy()
        d[f"{name}_well_covered"] = (d[f"{name}_spread"] <= spread) & (d[f"{name}_median_pred"] >= medium) & (d[f"{name}_p25_pred"] >= minimum)
        local_datasets[name] = d

    merged = local_datasets["seer"].merge(local_datasets["olink"], on="Uni_Prot_ID", how="outer").merge(local_datasets["soma"], on="Uni_Prot_ID", how="outer")
    merged["well_covered_count"] = merged[["seer_well_covered", "olink_well_covered", "soma_well_covered"]].sum(axis=1)
    true_positives = merged[merged["well_covered_count"] >= coverage].copy()
    
    true_positives["label"] = 1
    true_positives.drop("well_covered_count", axis=1, inplace=True)
    true_positives.drop([f"{name}_well_covered" for name in local_datasets.keys()], axis=1, inplace=True)
    
    return true_positives
    
def extract_true_negatives(datasets, spread: float = 0.6, medium: float = 0.2, minimum: float = 0.1, coverage: int = 2):
    local_datasets = {}
    
    for name, dataset in datasets.items():
        d = dataset.copy()
        d[f"{name}_poorly_covered"] = (d[f"{name}_spread"] > spread) | (d[f"{name}_median_pred"] < medium) | (d[f"{name}_p25_pred"] < minimum)
        local_datasets[name] = d

    merged = local_datasets["seer"].merge(local_datasets["olink"], on="Uni_Prot_ID", how="outer").merge(local_datasets["soma"], on="Uni_Prot_ID", how="outer")
    merged["poorly_covered_count"] = merged[["seer_poorly_covered", "olink_poorly_covered", "soma_poorly_covered"]].sum(axis=1)
    true_negatives = merged[merged["poorly_covered_count"] >= coverage].copy()
    
    true_negatives["label"] = -1
    true_negatives.drop("poorly_covered_count", axis=1, inplace=True)
    true_negatives.drop([f"{name}_poorly_covered" for name in local_datasets.keys()], axis=1, inplace=True)

    return true_negatives

def extract_no_confidence(datasets, true_positives, true_negatives):
    merged = datasets["seer"].merge(datasets["olink"], on="Uni_Prot_ID", how="outer").merge(datasets["soma"], on="Uni_Prot_ID", how="outer")
    no_confidence = merged[~merged["Uni_Prot_ID"].isin(true_positives["Uni_Prot_ID"]) & ~merged["Uni_Prot_ID"].isin(true_negatives["Uni_Prot_ID"])].copy()
    
    no_confidence["label"] = 0
    
    return no_confidence
        

def prepare_data():
    seer = pd.read_parquet(
        "/data/PHURI-Langenberg/people/Nat/Protein-Prediction/Data/Fixed/ST1_Seer_Cleaned.parquet"
    ).rename(columns={
        "prot.id": "Seer_ID",
        "uniprotswissprot": "Uni_Prot_ID",
        "med.pred": "seer_median_pred",
        "p25.pred": "seer_p25_pred",
        "p75.pred": "seer_p75_pred"
    }).astype({
        "seer_median_pred": float,
        "seer_p25_pred": float,
        "seer_p75_pred": float
    })[[
        "Seer_ID", "Uni_Prot_ID", "seer_median_pred", "seer_p25_pred", "seer_p75_pred"
    ]]
    seer["seer_spread"] = seer["seer_p75_pred"] - seer["seer_p25_pred"]
    
    print("In parquet:", "prot.1093" in seer["Seer_ID"].values)
    print(seer["Uni_Prot_ID"].apply(lambda x: x == "" or x is None or (isinstance(x, float))).sum())

    olink = pd.read_parquet(
        "/data/PHURI-Langenberg/people/Nat/Protein-Prediction/Data/Fixed/ST2_Olink_Cleaned.parquet"
    ).rename(columns={
        "Olink ID": "Olink_ID",
        "UniProt": "Uni_Prot_ID",
        "med.pred": "olink_median_pred",
        "p25.pred": "olink_p25_pred",
        "p75.pred": "olink_p75_pred"
    }).astype({
        "olink_median_pred": float,
        "olink_p25_pred": float,
        "olink_p75_pred": float
    })[[
        "Olink_ID", "Uni_Prot_ID", "olink_median_pred", "olink_p25_pred", "olink_p75_pred"
    ]]
    olink["olink_spread"] = olink["olink_p75_pred"] - olink["olink_p25_pred"]

    print(olink["Uni_Prot_ID"].apply(lambda x: x == "" or x is None or (isinstance(x, float))).sum())

    soma = pd.read_parquet(
        "/data/PHURI-Langenberg/people/Nat/Protein-Prediction/Data/Fixed/ST3_Soma_Cleaned.parquet"
    ).rename(columns={
        "SomaLogic ID": "Soma_ID",
        "UniProt": "Uni_Prot_ID",
        "med.pred": "soma_median_pred",
        "p25.pred": "soma_p25_pred",
        "p75.pred": "soma_p75_pred"
    }).astype({
        "soma_median_pred": float,
        "soma_p25_pred": float,
        "soma_p75_pred": float
    })[[
        "Soma_ID", "Uni_Prot_ID", "soma_median_pred", "soma_p25_pred", "soma_p75_pred"
    ]]
    soma["soma_spread"] = soma["soma_p75_pred"] - soma["soma_p25_pred"]
    
    print(soma["Uni_Prot_ID"].apply(lambda x: x == "" or x is None or (isinstance(x, float))).sum())
    
    datasets = {"seer": seer, "olink": olink, "soma": soma}
    
    true_positives = extract_true_positives(datasets)
    true_negatives = extract_true_negatives(datasets)
    no_confidence = extract_no_confidence(datasets, true_positives, true_negatives)
    
    save_df(pd.concat([true_positives, true_negatives], ignore_index=True), "/data/PHURI-Langenberg/people/Nat/Protein-Prediction/Data/NN_input/true_positives_and_true_negatives")
    save_df(no_confidence, "/data/PHURI-Langenberg/people/Nat/Protein-Prediction/Data/NN_input/no_confidence")

def save_df(df: pd.DataFrame, path: str):
    df.to_parquet(f"{path}.parquet", index=False)
    
    csv_df = df.copy()
    list_cols = [col for col in csv_df.columns if csv_df[col].apply(lambda x: isinstance(x, np.ndarray)).any()]
    for col in list_cols:
        csv_df[col] = csv_df[col].apply(lambda x: "|".join(x) if isinstance(x, np.ndarray) else x)
        
    csv_df.to_csv(f"{path}.csv", index=False)

if __name__ == "__main__":
    prepare_data()
