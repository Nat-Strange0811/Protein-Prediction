import polars as pl
import requests
import time

def main():
    
    spread = 0.3
    minimum = 0.2
    medium = 0.4
    
    poorSpread = 0.6
    poorMinimum = 0.1
    poorMedium = 0.2
    
    seer = pl.read_parquet("/data/PHURI-Langenberg/people/Nat/Protein-Prediction/Data/Fixed/ST1_Seer_Cleaned.parquet")
    olink = pl.read_parquet("/data/PHURI-Langenberg/people/Nat/Protein-Prediction/Data/Fixed/ST2_Olink_Cleaned.parquet")
    soma = pl.read_parquet("/data/PHURI-Langenberg/people/Nat/Protein-Prediction/Data/Fixed/ST3_Soma_Cleaned.parquet")
    
    seer = seer.with_columns(
        pl.when(pl.col(pl.List(pl.String)).list.len() == 1)
            .then(pl.col(pl.List(pl.String)).list.first())
            .otherwise(pl.col(pl.List(pl.String)).list.join("|"))
    )
    
    olink = olink.with_columns(
        pl.when(pl.col(pl.List(pl.String)).list.len() == 1)
            .then(pl.col(pl.List(pl.String)).list.first())
            .otherwise(pl.col(pl.List(pl.String)).list.join("|"))
    )
    
    soma = soma.with_columns(
        pl.when(pl.col(pl.List(pl.String)).list.len() == 1)
            .then(pl.col(pl.List(pl.String)).list.first())
            .otherwise(pl.col(pl.List(pl.String)).list.join("|"))
    )
    
    seer = seer.select(
        pl.col("prot.id").alias("Seer_ID").cast(pl.Utf8),
        pl.col("uniprotswissprot").alias("Uni_Prot_ID").cast(pl.Utf8),
        pl.col("med.pred").alias("seer_median_pred").cast(pl.Float64),
        pl.col("p25.pred").alias("seer_p25_pred").cast(pl.Float64),
        pl.col("p75.pred").alias("seer_p75_pred").cast(pl.Float64)
    ).with_columns(
        (pl.col("seer_p75_pred") - pl.col("seer_p25_pred")).alias("seer_spread").cast(pl.Float64)
    )
    olink = olink.select(
        pl.col("Olink ID").alias("Olink_ID").cast(pl.Utf8),
        pl.col("UniProt").alias("Uni_Prot_ID").cast(pl.Utf8),
        pl.col("med.pred").alias("olink_median_pred").cast(pl.Float64),
        pl.col("p25.pred").alias("olink_p25_pred").cast(pl.Float64),
        pl.col("p75.pred").alias("olink_p75_pred").cast(pl.Float64)
    ).with_columns(
        (pl.col("olink_p75_pred") - pl.col("olink_p25_pred")).alias("olink_spread").cast(pl.Float64)
    )
    soma = soma.select(
        pl.col("SomaLogic ID").alias("Soma_ID").cast(pl.Utf8),
        pl.col("UniProt").alias("Uni_Prot_ID").cast(pl.Utf8),
        pl.col("med.pred").alias("soma_median_pred").cast(pl.Float64),
        pl.col("p25.pred").alias("soma_p25_pred").cast(pl.Float64),
        pl.col("p75.pred").alias("soma_p75_pred").cast(pl.Float64)
    ).with_columns(
        (pl.col("soma_p75_pred") - pl.col("soma_p25_pred")).alias("soma_spread").cast(pl.Float64)
    )
    
    ids = {id for id in
        seer["Uni_Prot_ID"].to_list() +
        olink["Uni_Prot_ID"].to_list() +
        soma["Uni_Prot_ID"].to_list()
        if id}
    
    mapping = uniprot_lookup(list(ids))
    mapping = covered_PINNACLE(mapping)
    
    seer = query(seer, mapping)
    olink = query(olink, mapping)
    soma = query(soma, mapping)
    
    seer_soma = seer.join(soma, on="Uni_Prot_ID", how="inner").select(pl.col("Uni_Prot_ID"), pl.col("Seer_ID"), pl.col("Soma_ID"))
    seer_olink = seer.join(olink, on="Uni_Prot_ID", how="inner").select(pl.col("Uni_Prot_ID"), pl.col("Seer_ID"), pl.col("Olink_ID"))
    olink_soma = olink.join(soma, on="Uni_Prot_ID", how="inner").select(pl.col("Uni_Prot_ID"), pl.col("Olink_ID"), pl.col("Soma_ID"))
    all_technologies = seer.join(olink, on="Uni_Prot_ID", how="inner", suffix="_olink").join(soma, on="Uni_Prot_ID", how="inner", suffix="_soma")
    
    
    seer_well_covered = seer.filter((pl.col("seer_spread") < spread) & (pl.col("seer_median_pred") > medium) & (pl.col("seer_p25_pred") > minimum)).select(pl.col("Uni_Prot_ID"), pl.col("Seer_ID"), pl.col("gene_name"), pl.col("length"), pl.col("covered_PINNACLE"), pl.col("capable_for_NN_integration"))
    soma_well_covered = soma.filter((pl.col("soma_spread") < spread) & (pl.col("soma_median_pred") > medium) & (pl.col("soma_p25_pred") > minimum)).select(pl.col("Uni_Prot_ID"), pl.col("Soma_ID"), pl.col("gene_name"), pl.col("length"), pl.col("covered_PINNACLE"), pl.col("capable_for_NN_integration"))
    olink_well_covered = olink.filter((pl.col("olink_spread") < spread) & (pl.col("olink_median_pred") > medium) & (pl.col("olink_p25_pred") > minimum)).select(pl.col("Uni_Prot_ID"), pl.col("Olink_ID"), pl.col("gene_name"), pl.col("length"), pl.col("covered_PINNACLE"), pl.col("capable_for_NN_integration"))
    
    seer_poorly_covered = seer.filter((pl.col("seer_spread") > poorSpread) | (pl.col("seer_median_pred") < poorMedium) | (pl.col("seer_p25_pred") < poorMinimum)).select(pl.col("Uni_Prot_ID"), pl.col("Seer_ID"), pl.col("gene_name"), pl.col("length"), pl.col("covered_PINNACLE"), pl.col("capable_for_NN_integration"))
    soma_poorly_covered = soma.filter((pl.col("soma_spread") > poorSpread) | (pl.col("soma_median_pred") < poorMedium) | (pl.col("soma_p25_pred") < poorMinimum)).select(pl.col("Uni_Prot_ID"), pl.col("Soma_ID"), pl.col("gene_name"), pl.col("length"), pl.col("covered_PINNACLE"), pl.col("capable_for_NN_integration"))
    olink_poorly_covered = olink.filter((pl.col("olink_spread") > poorSpread) | (pl.col("olink_median_pred") < poorMedium) | (pl.col("olink_p25_pred") < poorMinimum)).select(pl.col("Uni_Prot_ID"), pl.col("Olink_ID"), pl.col("gene_name"), pl.col("length"), pl.col("covered_PINNACLE"), pl.col("capable_for_NN_integration"))
    
    seer_soma_well_covered = seer_well_covered.join(soma_well_covered, on="Uni_Prot_ID", how="inner")
    seer_olink_well_covered = seer_well_covered.join(olink_well_covered, on="Uni_Prot_ID", how="inner")
    olink_soma_well_covered = olink_well_covered.join(soma_well_covered, on="Uni_Prot_ID", how="inner")
    
    seer_soma_poorly_covered = seer_poorly_covered.join(soma_poorly_covered, on="Uni_Prot_ID", how="inner")
    seer_olink_poorly_covered = seer_poorly_covered.join(olink_poorly_covered, on="Uni_Prot_ID", how="inner")
    olink_soma_poorly_covered = olink_poorly_covered.join(soma_poorly_covered, on="Uni_Prot_ID", how="inner")
    
    atleast_two_technologies = (
        seer.join(olink, on="Uni_Prot_ID", how="full", suffix="_olink")
        .with_columns(
            pl.coalesce(["Uni_Prot_ID", "Uni_Prot_ID_olink"]).alias("Uni_Prot_ID")
        )
        .join(soma, on="Uni_Prot_ID", how="full", suffix="_soma")
        .filter(
            (pl.col("Seer_ID").is_not_null() & pl.col("Olink_ID").is_not_null()) |
            (pl.col("Seer_ID").is_not_null() & pl.col("Soma_ID").is_not_null()) |
            (pl.col("Olink_ID").is_not_null() & pl.col("Soma_ID").is_not_null())
        ).select(
            pl.coalesce(pl.col("Uni_Prot_ID"), pl.col("Uni_Prot_ID_soma")).alias("Uni_Prot_ID"),
            pl.coalesce(pl.col("gene_name"), pl.col("gene_name_olink"), pl.col("gene_name_soma")).alias("gene_name"),
            pl.coalesce(pl.col("length"), pl.col("length_olink"), pl.col("length_soma")).alias("length"),
            pl.coalesce(pl.col("covered_PINNACLE"), pl.col("covered_PINNACLE_olink"), pl.col("covered_PINNACLE_soma")).alias("covered_PINNACLE"),
            pl.coalesce(pl.col("capable_for_NN_integration"), pl.col("capable_for_NN_integration_olink"), pl.col("capable_for_NN_integration_soma")).alias("capable_for_NN_integration"),
            pl.col("Seer_ID"),
            pl.col("Olink_ID"),
            pl.col("Soma_ID")
        )
    )
    
    atleast_two_technologies_well_covered = (
        seer_well_covered.join(olink_well_covered, on="Uni_Prot_ID", how="full", suffix="_olink")
        .with_columns(
            pl.coalesce(["Uni_Prot_ID", "Uni_Prot_ID_olink"]).alias("Uni_Prot_ID")
        )
        .join(soma_well_covered, on="Uni_Prot_ID", how="full", suffix="_soma")
        .filter(
            (pl.col("Seer_ID").is_not_null() & pl.col("Olink_ID").is_not_null()) |
            (pl.col("Seer_ID").is_not_null() & pl.col("Soma_ID").is_not_null()) |
            (pl.col("Olink_ID").is_not_null() & pl.col("Soma_ID").is_not_null())
        ).select(
            pl.coalesce(pl.col("Uni_Prot_ID"), pl.col("Uni_Prot_ID_soma")).alias("Uni_Prot_ID"),
            pl.coalesce(pl.col("gene_name"), pl.col("gene_name_olink"), pl.col("gene_name_soma")).alias("gene_name"),
            pl.coalesce(pl.col("length"), pl.col("length_olink"), pl.col("length_soma")).alias("length"),
            pl.coalesce(pl.col("covered_PINNACLE"), pl.col("covered_PINNACLE_olink"), pl.col("covered_PINNACLE_soma")).alias("covered_PINNACLE"),
            pl.coalesce(pl.col("capable_for_NN_integration"), pl.col("capable_for_NN_integration_olink"), pl.col("capable_for_NN_integration_soma")).alias("capable_for_NN_integration"),
            pl.col("Seer_ID"),
            pl.col("Olink_ID"),
            pl.col("Soma_ID")
        )
    )
    
    atleast_two_technologies_poorly_covered = (
        seer_poorly_covered.join(olink_poorly_covered, on="Uni_Prot_ID", how="full", suffix="_olink")
        .with_columns(
            pl.coalesce(["Uni_Prot_ID", "Uni_Prot_ID_olink"]).alias("Uni_Prot_ID")
        )
        .join(soma_poorly_covered, on="Uni_Prot_ID", how="full", suffix="_soma").filter(
            (pl.col("Seer_ID").is_not_null() & pl.col("Olink_ID").is_not_null()) |
            (pl.col("Seer_ID").is_not_null() & pl.col("Soma_ID").is_not_null()) |
            (pl.col("Olink_ID").is_not_null() & pl.col("Soma_ID").is_not_null())
        ).select(
            pl.coalesce(pl.col("Uni_Prot_ID"), pl.col("Uni_Prot_ID_soma")).alias("Uni_Prot_ID"),
            pl.coalesce(pl.col("gene_name"), pl.col("gene_name_olink"), pl.col("gene_name_soma")).alias("gene_name"),
            pl.coalesce(pl.col("length"), pl.col("length_olink"), pl.col("length_soma")).alias("length"),
            pl.coalesce(pl.col("covered_PINNACLE"), pl.col("covered_PINNACLE_olink"), pl.col("covered_PINNACLE_soma")).alias("covered_PINNACLE"),
            pl.coalesce(pl.col("capable_for_NN_integration"), pl.col("capable_for_NN_integration_olink"), pl.col("capable_for_NN_integration_soma")).alias("capable_for_NN_integration"),
            pl.col("Seer_ID"),
            pl.col("Olink_ID"),
            pl.col("Soma_ID")
        )
    )
    
    all_technologies_well_covered = seer_well_covered.join(olink_well_covered, on="Uni_Prot_ID", how="inner", suffix="_olink").join(soma_well_covered, on="Uni_Prot_ID", how="inner", suffix="_soma")
    all_technologies_poorly_covered = seer_poorly_covered.join(olink_poorly_covered, on="Uni_Prot_ID", how="inner", suffix="_olink").join(soma_poorly_covered, on="Uni_Prot_ID", how="inner", suffix="_soma")
    
    seer_count = seer.select(pl.col("Uni_Prot_ID").n_unique())[0,0]
    seer_well_covered_count = seer_well_covered.select(pl.col("Uni_Prot_ID").n_unique())[0,0]
    olink_count = olink.select(pl.col("Uni_Prot_ID").n_unique())[0,0]
    olink_well_covered_count = olink_well_covered.select(pl.col("Uni_Prot_ID").n_unique())[0,0]
    soma_count = soma.select(pl.col("Uni_Prot_ID").n_unique())[0,0]
    soma_well_covered_count = soma_well_covered.select(pl.col("Uni_Prot_ID").n_unique())[0,0]
    
    seer_poorly_covered_count = seer_poorly_covered.select(pl.col("Uni_Prot_ID").n_unique())[0,0]
    olink_poorly_covered_count = olink_poorly_covered.select(pl.col("Uni_Prot_ID").n_unique())[0,0]
    soma_poorly_covered_count = soma_poorly_covered.select(pl.col("Uni_Prot_ID").n_unique())[0,0]
    
    seer_olink_count = seer_olink.filter(pl.col("Seer_ID").is_not_null() & pl.col("Olink_ID").is_not_null()).select(pl.col("Uni_Prot_ID").n_unique())[0,0]
    seer_olink_well_covered_count = seer_olink_well_covered.filter(pl.col("Olink_ID").is_not_null() & pl.col("Seer_ID").is_not_null()).select(pl.col("Uni_Prot_ID").n_unique())[0,0]
    seer_soma_count = seer_soma.filter(pl.col("Seer_ID").is_not_null() & pl.col("Soma_ID").is_not_null()).select(pl.col("Uni_Prot_ID").n_unique())[0,0]
    seer_soma_well_covered_count = seer_soma_well_covered.filter(pl.col("Soma_ID").is_not_null() & pl.col("Seer_ID").is_not_null()).select(pl.col("Uni_Prot_ID").n_unique())[0,0]
    olink_soma_count = olink_soma.filter(pl.col("Olink_ID").is_not_null() & pl.col("Soma_ID").is_not_null()).select(pl.col("Uni_Prot_ID").n_unique())[0,0]
    olink_soma_well_covered_count = olink_soma_well_covered.filter(pl.col("Olink_ID").is_not_null() & pl.col("Soma_ID").is_not_null()).select(pl.col("Uni_Prot_ID").n_unique())[0,0]
    
    seer_olink_poorly_covered_count = seer_olink_poorly_covered.filter(pl.col("Olink_ID").is_not_null() & pl.col("Seer_ID").is_not_null()).select(pl.col("Uni_Prot_ID").n_unique())[0,0]
    seer_soma_poorly_covered_count = seer_soma_poorly_covered.filter(pl.col("Soma_ID").is_not_null() & pl.col("Seer_ID").is_not_null()).select(pl.col("Uni_Prot_ID").n_unique())[0,0]
    olink_soma_poorly_covered_count = olink_soma_poorly_covered.filter(pl.col("Olink_ID").is_not_null() & pl.col("Soma_ID").is_not_null()).select(pl.col("Uni_Prot_ID").n_unique())[0,0]
    
    all_technologies_count = all_technologies.filter(pl.col("Seer_ID").is_not_null() & pl.col("Olink_ID").is_not_null() & pl.col("Soma_ID").is_not_null()).select(pl.col("Uni_Prot_ID").n_unique())[0,0]
    all_technologies_well_covered_count = all_technologies_well_covered.filter(pl.col("Seer_ID").is_not_null() & pl.col("Olink_ID").is_not_null() & pl.col("Soma_ID").is_not_null()).select(pl.col("Uni_Prot_ID").n_unique())[0,0]
    all_technologies_poorly_covered_count = all_technologies_poorly_covered.filter(pl.col("Seer_ID").is_not_null() & pl.col("Olink_ID").is_not_null() & pl.col("Soma_ID").is_not_null()).select(pl.col("Uni_Prot_ID").n_unique())[0,0]
    
    unique_atleast_two_technologies_count = atleast_two_technologies.select(pl.col("Uni_Prot_ID").n_unique())[0,0]
    unique_atleast_two_technologies_well_covered_count = atleast_two_technologies_well_covered.select(pl.col("Uni_Prot_ID").n_unique())[0,0]
    unique_atleast_two_technologies_poorly_covered_count = atleast_two_technologies_poorly_covered.select(pl.col("Uni_Prot_ID").n_unique())[0,0]
    
    unique_atleast_two_technologies_capable_count = atleast_two_technologies.filter(pl.col("capable_for_NN_integration")).select(pl.col("Uni_Prot_ID").n_unique())[0,0]
    unique_atleast_two_technologies_well_covered_capable_count = atleast_two_technologies_well_covered.filter(pl.col("capable_for_NN_integration")).select(pl.col("Uni_Prot_ID").n_unique())[0,0]
    unique_atleast_two_technologies_poorly_covered_capable_count = atleast_two_technologies_poorly_covered.filter(pl.col("capable_for_NN_integration")).select(pl.col("Uni_Prot_ID").n_unique())[0,0]
    
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
    
    
def query(df: pl.DataFrame, mapping: dict) -> pl.DataFrame:
    
    df = df.with_columns(
        pl.Series("gene_name", [mapping.get(uniprot_id, {}).get("gene_name") for uniprot_id in df["Uni_Prot_ID"]]),
        pl.Series("length", [mapping.get(uniprot_id, {}).get("length") for uniprot_id in df["Uni_Prot_ID"]]),
        pl.Series("covered_PINNACLE", [mapping.get(uniprot_id, {}).get("covered_PINNACLE") for uniprot_id in df["Uni_Prot_ID"]]),
        pl.Series("capable_for_NN_integration", [(mapping.get(uniprot_id, {}).get("length") or 0) <= 1024 and mapping.get(uniprot_id, {}).get("covered_PINNACLE") for uniprot_id in df["Uni_Prot_ID"]])
    )
    
    
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

if __name__ == "__main__":
    main()