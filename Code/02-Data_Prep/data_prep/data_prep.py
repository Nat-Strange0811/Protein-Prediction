from functools import reduce

import numpy as np
import pandas as pd

#Requests is a library that allows us to send HTTP requests in Python. It provides a simple and elegant way to interact with web services and APIs. We can use it to make GET, POST, PUT, DELETE, and other types of HTTP requests, and it also supports features like authentication, sessions, and retries.
import requests
from configs import ExtractorConfig
from embedding_extraction import DScriptExtractor, ESMExtractor
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

ANNOTATION_COLS = ["gene_name", "length", "covered_PINNACLE", "capable_for_NN_integration"]

def coalesce_annotation_columns(merged: pd.DataFrame) -> pd.DataFrame:
    """
    Merging seer/olink/soma (which all carry the same gene_name/length/covered_PINNACLE/
    capable_for_NN_integration columns from the UniProt mapping) leaves duplicate _x/_y
    columns from the merge suffixing. Since these columns describe the protein, not the
    technology, they're identical wherever more than one is non-null - so collapse them
    back into a single column per Uni_Prot_ID.
    """
    for col in ANNOTATION_COLS:
        candidates = [c for c in (f"{col}_x", f"{col}_y", col) if c in merged.columns]
        if len(candidates) > 1:
            merged[col] = reduce(lambda a, b: a.combine_first(b), [merged[c] for c in candidates])
            merged.drop(columns=[c for c in candidates if c != col], inplace=True)
    return merged

def extract_true_positives(datasets, spread: float = 0.4, medium: float = 0.4, minimum: float = 0.2, coverage: int = 2):

    local_datasets = {}

    for name, dataset in datasets.items():
        d = dataset.copy()
        d[f"{name}_well_covered"] = (d[f"{name}_spread"] <= spread) & (d[f"{name}_median_pred"] >= medium) & (d[f"{name}_p25_pred"] >= minimum)
        # gene_name/length/covered_PINNACLE/capable_for_NN_integration are constant per Uni_Prot_ID (they come
        # from the same UniProt mapping regardless of probe), so "first" preserves them through the collapse
        # to one row per protein instead of dropping them like a bare Series groupby would.
        agg = {f"{name}_well_covered": "mean", **{col: "first" for col in ANNOTATION_COLS}}
        d = d.groupby("Uni_Prot_ID").agg(agg).reset_index()
        d[f"{name}_well_covered"] = d[f"{name}_well_covered"] >= 1

        local_datasets[name] = d

    merged = local_datasets["seer"].merge(local_datasets["olink"], on="Uni_Prot_ID", how="outer").merge(local_datasets["soma"], on="Uni_Prot_ID", how="outer")
    merged = coalesce_annotation_columns(merged)
    merged["well_covered_count"] = merged[["seer_well_covered", "olink_well_covered", "soma_well_covered"]].sum(axis=1)
    true_positives = merged[merged["well_covered_count"] >= coverage].copy()
    
    true_positives["label"] = 1
    true_positives.drop("well_covered_count", axis=1, inplace=True)
    true_positives.drop([f"{name}_well_covered" for name in local_datasets.keys()], axis=1, inplace=True)
    
    return true_positives
    
def extract_true_negatives(datasets, spread: float = 0.6, medium: float = 0.15, minimum: float = 0.05, coverage: int = 2):
    local_datasets = {}
    
    for name, dataset in datasets.items():
        d = dataset.copy()
        d[f"{name}_poorly_covered"] = (d[f"{name}_spread"] > spread) | (d[f"{name}_median_pred"] < medium) | (d[f"{name}_p25_pred"] < minimum)
        agg = {f"{name}_poorly_covered": "mean", **{col: "first" for col in ANNOTATION_COLS}}
        d = d.groupby("Uni_Prot_ID").agg(agg).reset_index()
        d[f"{name}_poorly_covered"] = d[f"{name}_poorly_covered"] >= 1
        local_datasets[name] = d

    merged = local_datasets["seer"].merge(local_datasets["olink"], on="Uni_Prot_ID", how="outer").merge(local_datasets["soma"], on="Uni_Prot_ID", how="outer")
    merged = coalesce_annotation_columns(merged)
    merged["poorly_covered_count"] = merged[["seer_poorly_covered", "olink_poorly_covered", "soma_poorly_covered"]].sum(axis=1)
    true_negatives = merged[merged["poorly_covered_count"] >= coverage].copy()
    
    true_negatives["label"] = 0
    true_negatives.drop("poorly_covered_count", axis=1, inplace=True)
    true_negatives.drop([f"{name}_poorly_covered" for name in local_datasets.keys()], axis=1, inplace=True)

    return true_negatives

def extract_no_confidence(datasets, true_positives, true_negatives):
    merged = datasets["seer"].merge(datasets["olink"], on="Uni_Prot_ID", how="outer").merge(datasets["soma"], on="Uni_Prot_ID", how="outer")
    merged = coalesce_annotation_columns(merged)
    no_confidence = merged[~merged["Uni_Prot_ID"].isin(true_positives["Uni_Prot_ID"]) & ~merged["Uni_Prot_ID"].isin(true_negatives["Uni_Prot_ID"])].copy()

    # Unlike true_positives/true_negatives, this merge is over the raw (un-collapsed) per-technology
    # frames, so a protein with multiple probes in any technology produces multiple rows here - collapse
    # back to one row per protein to match the contract the rest of the NN_input pipeline expects.
    no_confidence = no_confidence.drop_duplicates(subset="Uni_Prot_ID", keep="first")

    no_confidence["label"] = -1

    return no_confidence

def save_df(df: pd.DataFrame, path: str):
    df.to_parquet(f"{path}.parquet", index=False)
    
    csv_df = df.copy()
    list_cols = [col for col in csv_df.columns if csv_df[col].apply(lambda x: isinstance(x, np.ndarray)).any()]
    for col in list_cols:
        csv_df[col] = csv_df[col].apply(lambda x: "|".join(x) if isinstance(x, np.ndarray) else x)
        
    csv_df.to_csv(f"{path}.csv", index=False)

def _build_session(max_retries: int, backoff_factor: float) -> requests.Session:
    """
    Method - Build Session:
    
    Args:
    - max_retries: An integer representing the maximum number of retries for failed HTTP requests.
    - backoff_factor: A float representing the backoff factor for retrying failed HTTP requests (e.g., 0.5 means that the delay between retries will be 0.5 seconds, then 1 second, then 2 seconds, etc.).
    
    Returns:
    - A configured requests.Session object with retry logic.
    """
    
    #The retry strategy utilises the library import Retry from urllib3.util.retry to build our http session with retry logic. We specify the total number of retries, the backoff factor, 
    #and the HTTP codes to retry on. 
    retry_strategy = Retry(
        total=max_retries,
        backoff_factor=backoff_factor,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False
    )
    
    #The HTTPAdapter is then used to mount the retry strategy to both http and https requests in the session.
    adapter = HTTPAdapter(max_retries=retry_strategy)
    
    #We then create a requests.Session and mount the adapter to both http and https requests, ensuring that all requests made through this session will have the retry logic applied.
    session = requests.Session()
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    
    return session

def _get(session, url: str) -> requests.Response:
    """
    Method - Get:
    
    Args:
    - url: A string representing the URL to which the GET request should be made.
    
    Returns:
    - A requests.Response object containing the response from the GET request.
    
    Raises:
    - requests.HTTPError: If the HTTP request fails (e.g., due to network issues, server errors, or invalid URL).
    """
    
    response = session.get(url, timeout = 30)
    
    if not response.ok:
        raise requests.HTTPError(
            f"Request failed [{response.status_code}] for URL: {url}",
            response=response
        )
        
    return response

def fetch_sequence(session, UNIPROT_FASTA_URL, uniprot_id: str) -> str:
    """
    Method - Fetch Sequence:
    
    Args:
    - session: A requests.Session object with retry logic.
    - UNIPROT_FASTA_URL: A string representing the URL template for fetching UniProt sequences.
    - uniprot_id: A string representing the UniProt ID of the protein for which we want to fetch the amino acid sequence.
    
    Returns:
    - A string containing the amino acid sequence for the specified UniProt ID.
    
    Raises:
    - requests.HTTPError: If the HTTP request to retrieve the sequence fails (e.g., due to network issues, server errors, or invalid UniProt ID).
    """
    
    url = UNIPROT_FASTA_URL.format(uniprot_id=uniprot_id)
    response = _get(session, url)
    
    #Fasta format is >header\nseqeuence, so we split on newline and take the second part as the sequence.
    lines = response.text.strip().split('\n')
    sequence = ''.join(lines[1:])  # Join all lines after the header to get the full sequence
    
    return sequence

def _panel_uniprot_ids(configs: list[ExtractorConfig] = None) -> set[str]:
    """
    D_Script scores each protein against a fixed interactor panel (Data/Panels/panel_<n>.txt)
    that's independent of the true_positives/true_negatives set - those IDs need to be in the
    UniProt cache too, or DScriptExtractor.extract_panel_embeddings() will fail to look them up.
    Derived from the configs actually in use so we don't fetch panels nobody asked for.
    """
    ids = set()
    for config in configs or []:
        if config.extractor_type != "dscript":
            continue
        panel_path = f"/data/PHURI-Langenberg/people/Nat/Protein-Prediction/Data/Panels/panel_{config.panel}.txt"
        with open(panel_path) as f:
            ids.update(line.strip() for line in f if line.strip())
    return ids

def create_uni_prot_cache(df: pd.DataFrame, extra_uniprot_ids: set[str] = frozenset(), cache_path: str = "/data/PHURI-Langenberg/people/Nat/Protein-Prediction/Data/NN_input/uni_prot_cache.parquet", backoff_factor: float = 0.5, retries: int = 3):
    #UNIPROT_FASTA_URL for amino acid fetching
    UNIPROT_FASTA_URL = "https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta"

    session = _build_session(max_retries=retries, backoff_factor=backoff_factor)

    uni_prot_cache = pd.DataFrame(columns=["Uni_Prot_ID", "sequence", "gene_name", "length"])

    # extra_uniprot_ids (e.g. D_Script panel members) aren't part of df and have no gene_name -
    # fetched and cached the same way, just without that annotation.
    all_ids = list(dict.fromkeys(df["Uni_Prot_ID"].tolist())) + [uid for uid in extra_uniprot_ids if uid not in set(df["Uni_Prot_ID"])]

    for uniprot_id in all_ids:
        gene_rows = df.loc[df["Uni_Prot_ID"] == uniprot_id, "gene_name"]
        gene_name = gene_rows.iloc[0] if not gene_rows.empty else None
        try:
            sequence = fetch_sequence(session, UNIPROT_FASTA_URL, uniprot_id)
            length = len(sequence)
        except requests.HTTPError as e:
            # Keep the ID in the cache with an empty sequence rather than dropping the row -
            # a missing row causes a KeyError in fetch_sequence(), while an empty sequence is
            # already excluded by the "length > 0" check every extractor's filter() applies.
            print(f"Failed to fetch sequence for UniProt ID {uniprot_id}: {e}")
            sequence, length = "", 0
        uni_prot_cache = pd.concat([uni_prot_cache, pd.DataFrame({"Uni_Prot_ID": [uniprot_id], "sequence": [sequence], "gene_name": [gene_name], "length": [length]})], ignore_index=True)

    uni_prot_cache.to_parquet(cache_path, index=False)

def filter_NN_compatible(df: pd.DataFrame, configs: list[ExtractorConfig] = None) -> pd.DataFrame:
    extractor_registry = {
        "esm": ESMExtractor,
        "dscript": DScriptExtractor
    }
    
    for config in configs:
        extractor_class = extractor_registry.get(config.extractor_type)
        if not extractor_class:
            raise ValueError(f"Unknown extractor: {config.extractor_type}")
        
        extractor = extractor_class(config)
        df = extractor.filter(df)
        
    return df

def prepare_data(extractor_configs: list[ExtractorConfig] = None, raw_csv: str = "/data/PHURI-Langenberg/people/Nat/Protein-Prediction/Data/NN_input/true_positives_and_true_negatives", backoff_factor: float = 0.5, retries: int = 3):
    seer = pd.read_parquet(
        "/data/PHURI-Langenberg/people/Nat/Protein-Prediction/Data/Fixed/ST1_Seer_Cleaned.parquet"
    )[[
        "Seer_ID", "Uni_Prot_ID", "seer_median_pred", "seer_p25_pred", "seer_p75_pred", *ANNOTATION_COLS
    ]]
    seer["seer_spread"] = seer["seer_p75_pred"] - seer["seer_p25_pred"]
    
    print(seer["Uni_Prot_ID"].apply(lambda x: x == "" or x is None or (isinstance(x, float))).sum())

    olink = pd.read_parquet(
        "/data/PHURI-Langenberg/people/Nat/Protein-Prediction/Data/Fixed/ST2_Olink_Cleaned.parquet"
    )[[
        "Olink_ID", "Uni_Prot_ID", "olink_median_pred", "olink_p25_pred", "olink_p75_pred", *ANNOTATION_COLS
    ]]
    olink["olink_spread"] = olink["olink_p75_pred"] - olink["olink_p25_pred"]

    print(olink["Uni_Prot_ID"].apply(lambda x: x == "" or x is None or (isinstance(x, float))).sum())

    soma = pd.read_parquet(
        "/data/PHURI-Langenberg/people/Nat/Protein-Prediction/Data/Fixed/ST3_Soma_Cleaned.parquet"
    )[[
        "Soma_ID", "Uni_Prot_ID", "soma_median_pred", "soma_p25_pred", "soma_p75_pred", *ANNOTATION_COLS
    ]]
    soma["soma_spread"] = soma["soma_p75_pred"] - soma["soma_p25_pred"]
    
    print(soma["Uni_Prot_ID"].apply(lambda x: x == "" or x is None or (isinstance(x, float))).sum())
    
    datasets = {"seer": seer, "olink": olink, "soma": soma}
    
    true_positives = extract_true_positives(datasets)
    true_negatives = extract_true_negatives(datasets)
    no_confidence = extract_no_confidence(datasets, true_positives, true_negatives)
    
    combined = pd.concat([true_positives, true_negatives], ignore_index=True)
    
    create_uni_prot_cache(combined, extra_uniprot_ids=_panel_uniprot_ids(extractor_configs), cache_path="/data/PHURI-Langenberg/people/Nat/Protein-Prediction/Data/NN_input/uni_prot_cache.parquet", backoff_factor=backoff_factor, retries=retries)
    
    positives_negatives = filter_NN_compatible(combined, configs=extractor_configs)
    
    save_df(positives_negatives, raw_csv)
    save_df(no_confidence, "/data/PHURI-Langenberg/people/Nat/Protein-Prediction/Data/NN_input/no_confidence")

if __name__ == "__main__":
    prepare_data()
