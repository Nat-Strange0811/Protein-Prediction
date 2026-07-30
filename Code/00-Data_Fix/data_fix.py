import csv
import re
import time

import numpy as np
import pandas as pd
import requests

LIST_TYPES = (list, np.ndarray)


def read_csv_truncated(path, comment_prefix="#"):
    """Read a csv, truncating/padding ragged rows to the header width and mapping 'NA' to null (mirrors polars' truncate_ragged_lines + null_values)."""
    header = None
    rows = []
    with open(path, newline="") as f:
        for line in f:
            if line.startswith(comment_prefix):
                continue
            row = next(csv.reader([line]))
            if header is None:
                header = [c.strip() for c in row]
                continue
            if len(row) > len(header):
                row = row[: len(header)]
            elif len(row) < len(header):
                row = row + [None] * (len(header) - len(row))
            rows.append(row)
    df = pd.DataFrame(rows, columns=header)
    return df.replace("NA", None)


def infer_column_types(df):
    df = df.copy()
    for col in df.columns:
        converted = pd.to_numeric(df[col], errors="coerce")
        if converted.isna().sum() == df[col].isna().sum():
            df[col] = converted
    return df


def split_multi_id(value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    value = value.strip()
    if value == "NA":
        return None
    parts = value.split("|")
    if len(parts) == 1:
        parts = parts[0].split(";")
    if len(parts) == 1:
        parts = parts[0].split("_")
    return parts


def clean_dataframe(df):
    df = infer_column_types(df)
    string_cols = df.select_dtypes(include="object").columns
    for col in string_cols:
        df[col] = df[col].map(split_multi_id)
    return df


def join_list_columns(df):
    out = df.copy()
    for col in out.columns:
        if out[col].map(lambda v: isinstance(v, LIST_TYPES)).any():
            out[col] = out[col].map(lambda v: "|".join(v) if isinstance(v, LIST_TYPES) else v)
    return out


def check_UniProt(df, col, name):

    UNIPROT_PATTERN = re.compile(r'^([OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2})$')

    uniprot_ids = df[col].explode().unique().tolist()

    rejected_ids = [uid for uid in uniprot_ids if pd.notna(uid) and not UNIPROT_PATTERN.match(str(uid))]

    print(f"Rejected UniProt IDs in column '{col}' ({name}): {rejected_ids}")

    return rejected_ids


def query(df: pd.DataFrame, mapping: dict, uniprot: str) -> pd.DataFrame:

    df = df.copy()
    df["gene_name"] = [mapping.get(uid, {}).get("gene_name") for uid in df[uniprot]]
    df["length"] = [mapping.get(uid, {}).get("length") for uid in df[uniprot]]
    df["covered_PINNACLE"] = [mapping.get(uid, {}).get("covered_PINNACLE") for uid in df[uniprot]]
    df["capable_for_NN_integration"] = [
        (mapping.get(uid, {}).get("length") or 0) <= 2000 and (mapping.get(uid, {}).get("length") or 0) > 0 and mapping.get(uid, {}).get("covered_PINNACLE")
        for uid in df[uniprot]
    ]

    return df

def uniprot_lookup(uniprot_ids: list) -> dict:
    mapping = {}
    chunk_size = 250
    chunks = [uniprot_ids[i:i + chunk_size] for i in range(0, len(uniprot_ids), chunk_size)]

    url = "https://rest.uniprot.org/uniprotkb/accessions"

    for i, chunk in enumerate(chunks):

        print(chunk[0])

        params = {
            "accessions": ",".join(chunk),
            "fields": "accession,gene_names,length",
            "format": "tsv",
            "size": chunk_size
        }

        for attempt in range(3):
            response = requests.get(url, params=params, timeout=30)
            if response.status_code == 429 or response.status_code >= 500:
                if attempt < 2:
                    time.sleep(5 * 2 ** attempt)  # 5s, 10s
                    continue
            response.raise_for_status()
            break

        for line in response.text.strip().split("\n")[1:]:
            parts = line.split("\t")
            if len(parts) >= 3:
                uniprotId = parts[0]
                gene_name = parts[1].split(" ")[0] if parts[1] else None
                length = int(parts[2]) if parts[2].isdigit() else None
                mapping[uniprotId] = {"gene_name": gene_name, "length": length}

        time.sleep(0.5)

    return mapping


def covered_PINNACLE(mapping: dict) -> dict:
    # Implementation for checking PINNACLE coverage
    pinnacle_genes = set()

    with open("Data/PINNACLE/networks/networks/global_ppi_edgelist.txt", "r") as f:
        for line in f:
            a, b = line.strip().split()
            pinnacle_genes.add(a)
            pinnacle_genes.add(b)

    for uniprot_id in mapping:
        mapping[uniprot_id]["covered_PINNACLE"] = mapping[uniprot_id]["gene_name"] in pinnacle_genes
        print(uniprot_id, mapping[uniprot_id]["gene_name"], mapping[uniprot_id]["length"], mapping[uniprot_id]["covered_PINNACLE"])

    return mapping

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

def fix_data():

    seer_uniprot = "uniprotswissprot"
    olink_uniprot = "UniProt"
    soma_uniprot = "UniProt"

    seer = read_csv_truncated("/data/PHURI-Langenberg/people/Nat/Protein-Prediction/Data/Raw/Supplementary_Tables_Seer_pQTL_20251027(ST1).csv")
    seer = clean_dataframe(seer)
    seer = seer.explode(seer_uniprot)

    seer_fails = check_UniProt(seer, seer_uniprot, "Seer")
    seer = seer[~seer[seer_uniprot].isin(seer_fails)]
    seer = seer[seer[seer_uniprot].notna() & (seer[seer_uniprot] != "")]

    olink = read_csv_truncated("/data/PHURI-Langenberg/people/Nat/Protein-Prediction/Data/Raw/Supplementary_Tables_Seer_pQTL_20251027(ST2).csv")
    olink = clean_dataframe(olink)
    olink = olink.explode(olink_uniprot)

    olink_fails = check_UniProt(olink, olink_uniprot, "Olink")
    olink = olink[~olink[olink_uniprot].isin(olink_fails)]
    olink = olink[olink[olink_uniprot].notna() & (olink[olink_uniprot] != "")]

    soma = read_csv_truncated("/data/PHURI-Langenberg/people/Nat/Protein-Prediction/Data/Raw/Supplementary_Tables_Seer_pQTL_20251027(ST3).csv")
    soma = clean_dataframe(soma)
    soma = soma.explode(soma_uniprot)

    soma_fails = check_UniProt(soma, soma_uniprot, "Soma")
    soma = soma[~soma[soma_uniprot].isin(soma_fails)]
    soma = soma[soma[soma_uniprot].notna() & (soma[soma_uniprot] != "")]

    ids = {uid for uid in
        seer[seer_uniprot].tolist() +
        olink[olink_uniprot].tolist() +
        soma[soma_uniprot].tolist()
        if uid}

    mapping = uniprot_lookup(list(ids))
    mapping = covered_PINNACLE(mapping)

    seer = query(seer, mapping, seer_uniprot)
    olink = query(olink, mapping, olink_uniprot)
    soma = query(soma, mapping, soma_uniprot)

    # Collapse the pipe/semicolon/underscore-split list columns (incl. med.pred/p25.pred/p75.pred,
    # which don't cleanly numeric-infer due to whitespace-padded "NA" values) back to scalars
    # before renaming/casting - casting a still-listified column silently NaNs/mis-stringifies it.
    seer = join_list_columns(seer)
    olink = join_list_columns(olink)
    soma = join_list_columns(soma)

    seer = seer.rename(columns={
        "prot.id": "Seer_ID",
        "uniprotswissprot": "Uni_Prot_ID",
        "med.pred": "seer_median_pred",
        "p25.pred": "seer_p25_pred",
        "p75.pred": "seer_p75_pred",
    })[["Seer_ID", "Uni_Prot_ID", "seer_median_pred", "seer_p25_pred", "seer_p75_pred", "gene_name", "length", "covered_PINNACLE", "capable_for_NN_integration"]]
    seer["Seer_ID"] = cast_col(seer["Seer_ID"], "string")
    seer["Uni_Prot_ID"] = cast_col(seer["Uni_Prot_ID"], "string")
    seer["seer_median_pred"] = cast_col(seer["seer_median_pred"], "float")
    seer["seer_p25_pred"] = cast_col(seer["seer_p25_pred"], "float")
    seer["seer_p75_pred"] = cast_col(seer["seer_p75_pred"], "float")
    seer["gene_name"] = cast_col(seer["gene_name"], "string")
    seer["length"] = cast_col(seer["length"], "int")
    seer["covered_PINNACLE"] = cast_col(seer["covered_PINNACLE"], "bool")
    seer["capable_for_NN_integration"] = cast_col(seer["capable_for_NN_integration"], "bool")
    seer["seer_spread"] = seer["seer_p75_pred"] - seer["seer_p25_pred"]

    olink = olink.rename(columns={
        "Olink ID": "Olink_ID",
        "UniProt": "Uni_Prot_ID",
        "med.pred": "olink_median_pred",
        "p25.pred": "olink_p25_pred",
        "p75.pred": "olink_p75_pred",
    })[["Olink_ID", "Uni_Prot_ID", "olink_median_pred", "olink_p25_pred", "olink_p75_pred", "gene_name", "length", "covered_PINNACLE", "capable_for_NN_integration"]]
    olink["Olink_ID"] = cast_col(olink["Olink_ID"], "string")
    olink["Uni_Prot_ID"] = cast_col(olink["Uni_Prot_ID"], "string")
    olink["olink_median_pred"] = cast_col(olink["olink_median_pred"], "float")
    olink["olink_p25_pred"] = cast_col(olink["olink_p25_pred"], "float")
    olink["olink_p75_pred"] = cast_col(olink["olink_p75_pred"], "float")
    olink["gene_name"] = cast_col(olink["gene_name"], "string")
    olink["length"] = cast_col(olink["length"], "int")
    olink["covered_PINNACLE"] = cast_col(olink["covered_PINNACLE"], "bool")
    olink["capable_for_NN_integration"] = cast_col(olink["capable_for_NN_integration"], "bool")
    olink["olink_spread"] = olink["olink_p75_pred"] - olink["olink_p25_pred"]

    soma = soma.rename(columns={
        "SomaLogic ID": "Soma_ID",
        "UniProt": "Uni_Prot_ID",
        "med.pred": "soma_median_pred",
        "p25.pred": "soma_p25_pred",
        "p75.pred": "soma_p75_pred",
    })[["Soma_ID", "Uni_Prot_ID", "soma_median_pred", "soma_p25_pred", "soma_p75_pred", "gene_name", "length", "covered_PINNACLE", "capable_for_NN_integration"]]
    soma["Soma_ID"] = cast_col(soma["Soma_ID"], "string")
    soma["Uni_Prot_ID"] = cast_col(soma["Uni_Prot_ID"], "string")
    soma["soma_median_pred"] = cast_col(soma["soma_median_pred"], "float")
    soma["soma_p25_pred"] = cast_col(soma["soma_p25_pred"], "float")
    soma["soma_p75_pred"] = cast_col(soma["soma_p75_pred"], "float")
    soma["gene_name"] = cast_col(soma["gene_name"], "string")
    soma["length"] = cast_col(soma["length"], "int")
    soma["covered_PINNACLE"] = cast_col(soma["covered_PINNACLE"], "bool")
    soma["capable_for_NN_integration"] = cast_col(soma["capable_for_NN_integration"], "bool")
    soma["soma_spread"] = soma["soma_p75_pred"] - soma["soma_p25_pred"]

    seer.to_csv("/data/PHURI-Langenberg/people/Nat/Protein-Prediction/Data/Fixed/ST1_Seer_Cleaned.csv", index=False)
    olink.to_csv("/data/PHURI-Langenberg/people/Nat/Protein-Prediction/Data/Fixed/ST2_Olink_Cleaned.csv", index=False)
    soma.to_csv("/data/PHURI-Langenberg/people/Nat/Protein-Prediction/Data/Fixed/ST3_Soma_Cleaned.csv", index=False)

    seer.to_parquet("/data/PHURI-Langenberg/people/Nat/Protein-Prediction/Data/Fixed/ST1_Seer_Cleaned.parquet", index=False)
    olink.to_parquet("/data/PHURI-Langenberg/people/Nat/Protein-Prediction/Data/Fixed/ST2_Olink_Cleaned.parquet", index=False)
    soma.to_parquet("/data/PHURI-Langenberg/people/Nat/Protein-Prediction/Data/Fixed/ST3_Soma_Cleaned.parquet", index=False)


if __name__ == "__main__":
    fix_data()
