import polars as pl

def main():
    
    spread = 0.4
    minimum = 0.05
    medium = 0.25
    
    seer = pl.scan_csv("/data/PHURI-Langenberg/people/Nat/Protein-Prediction/Data/Supplementary_Tables_Seer_pQTL_20251027(ST1).csv", truncate_ragged_lines=True, separator=",", comment_prefix="#", null_values="NA").select(
        pl.col("prot.id").alias("Seer_ID").cast(pl.Utf8),
        pl.col("uniprotswissprot").alias("Uni_Prot_ID").cast(pl.Utf8),
        pl.col("med.pred").alias("seer_median_pred").cast(pl.Float64),
        pl.col("p25.pred").alias("seer_p25_pred").cast(pl.Float64),
        pl.col("p75.pred").alias("seer_p75_pred").cast(pl.Float64),
        (pl.col("p75.pred") - pl.col("p25.pred")).alias("seer_spread").cast(pl.Float64)
    )
    olink = pl.scan_csv("/data/PHURI-Langenberg/people/Nat/Protein-Prediction/Data/Supplementary_Tables_Seer_pQTL_20251027(ST2).csv", truncate_ragged_lines=True, separator=",", comment_prefix="#", null_values="NA").select(
        pl.col("Olink ID").alias("Olink_ID").cast(pl.Utf8),
        pl.col("UniProt").alias("Uni_Prot_ID").cast(pl.Utf8),
        pl.col("med.pred").alias("olink_median_pred").cast(pl.Float64),
        pl.col("p25.pred").alias("olink_p25_pred").cast(pl.Float64),
        pl.col("p75.pred").alias("olink_p75_pred").cast(pl.Float64),
        (pl.col("p75.pred") - pl.col("p25.pred")).alias("olink_spread").cast(pl.Float64)
    )
    soma = pl.scan_csv("/data/PHURI-Langenberg/people/Nat/Protein-Prediction/Data/Supplementary_Tables_Seer_pQTL_20251027(ST3).csv", truncate_ragged_lines=True, separator=",", comment_prefix="#", null_values="NA").select(
        pl.col("SomaLogic ID").alias("Soma_ID").cast(pl.Utf8),
        pl.col("UniProt").alias("Uni_Prot_ID").cast(pl.Utf8),
        pl.col("med.pred").alias("soma_median_pred").cast(pl.Float64),
        pl.col("p25.pred").alias("soma_p25_pred").cast(pl.Float64),
        pl.col("p75.pred").alias("soma_p75_pred").cast(pl.Float64),
        (pl.col("p75.pred") - pl.col("p25.pred")).alias("soma_spread").cast(pl.Float64)
    )
    
    seer_well_covered = seer.filter((pl.col("seer_spread") < spread) & (pl.col("seer_median_pred") > medium) & (pl.col("seer_p25_pred") > minimum)).select(pl.col("Uni_Prot_ID"), pl.col("Seer_ID"))
    soma_well_covered = soma.filter((pl.col("soma_spread") < spread) & (pl.col("soma_median_pred") > medium) & (pl.col("soma_p25_pred") > minimum)).select(pl.col("Uni_Prot_ID"), pl.col("Soma_ID"))
    olink_well_covered = olink.filter((pl.col("olink_spread") < spread) & (pl.col("olink_median_pred") > medium) & (pl.col("olink_p25_pred") > minimum)).select(pl.col("Uni_Prot_ID"), pl.col("Olink_ID"))
    
    seer_soma = seer.join(soma, on="Uni_Prot_ID", how="inner").select(pl.col("Uni_Prot_ID"), pl.col("Seer_ID"), pl.col("Soma_ID"))
    seer_olink = seer.join(olink, on="Uni_Prot_ID", how="inner").select(pl.col("Uni_Prot_ID"), pl.col("Seer_ID"), pl.col("Olink_ID"))
    olink_soma = olink.join(soma, on="Uni_Prot_ID", how="inner").select(pl.col("Uni_Prot_ID"), pl.col("Olink_ID"), pl.col("Soma_ID"))
    seer_soma_well_covered = seer_well_covered.join(soma_well_covered, on="Uni_Prot_ID", how="inner")
    seer_olink_well_covered = seer_well_covered.join(olink_well_covered, on="Uni_Prot_ID", how="inner")
    olink_soma_well_covered = olink_well_covered.join(soma_well_covered, on="Uni_Prot_ID", how="inner")
    
    atleast_two_technologies = seer.join(olink, on="Uni_Prot_ID", how="full", suffix="_olink").join(soma, on="Uni_Prot_ID", how="full", suffix="_soma").filter(
        (pl.col("Seer_ID").is_not_null() & pl.col("Olink_ID").is_not_null()) |
        (pl.col("Seer_ID").is_not_null() & pl.col("Soma_ID").is_not_null()) |
        (pl.col("Olink_ID").is_not_null() & pl.col("Soma_ID").is_not_null())
    ).select(
        pl.coalesce(pl.col("Uni_Prot_ID"), pl.col("Uni_Prot_ID_olink"), pl.col("Uni_Prot_ID_soma")).alias("Uni_Prot_ID"),
        pl.col("Seer_ID"),
        pl.col("Olink_ID"),
        pl.col("Soma_ID")
    )
    
    atleast_two_technologies_well_covered = seer_well_covered.join(olink_well_covered, on="Uni_Prot_ID", how="full", suffix="_olink").join(soma_well_covered, on="Uni_Prot_ID", how="full", suffix="_soma").filter(
        (pl.col("Seer_ID").is_not_null() & pl.col("Olink_ID").is_not_null()) |
        (pl.col("Seer_ID").is_not_null() & pl.col("Soma_ID").is_not_null()) |
        (pl.col("Olink_ID").is_not_null() & pl.col("Soma_ID").is_not_null())
    ).select(
        pl.coalesce(pl.col("Uni_Prot_ID"), pl.col("Uni_Prot_ID_olink"), pl.col("Uni_Prot_ID_soma")).alias("Uni_Prot_ID"),
        pl.col("Seer_ID"),
        pl.col("Olink_ID"),
        pl.col("Soma_ID")
    )
    
    all_technologies = seer.join(olink, on="Uni_Prot_ID", how="inner").join(soma, on="Uni_Prot_ID", how="inner")
    all_technologies_well_covered = seer_well_covered.join(olink_well_covered, on="Uni_Prot_ID", how="inner").join(soma_well_covered, on="Uni_Prot_ID", how="inner")
    
    seer_count = seer.select(pl.col("Uni_Prot_ID").n_unique()).collect()[0,0]
    seer_well_covered_count = seer_well_covered.select(pl.col("Uni_Prot_ID").n_unique()).collect()[0,0]
    olink_count = olink.select(pl.col("Uni_Prot_ID").n_unique()).collect()[0,0]
    olink_well_covered_count = olink_well_covered.select(pl.col("Uni_Prot_ID").n_unique()).collect()[0,0]
    soma_count = soma.select(pl.col("Uni_Prot_ID").n_unique()).collect()[0,0]
    soma_well_covered_count = soma_well_covered.select(pl.col("Uni_Prot_ID").n_unique()).collect()[0,0]
    
    seer_olink_count = seer_olink.filter(pl.col("Seer_ID").is_not_null() & pl.col("Olink_ID").is_not_null()).select(pl.col("Uni_Prot_ID").n_unique()).collect()[0,0]
    seer_olink_well_covered_count = seer_olink_well_covered.filter(pl.col("Olink_ID").is_not_null() & pl.col("Seer_ID").is_not_null()).select(pl.col("Uni_Prot_ID").n_unique()).collect()[0,0]
    seer_soma_count = seer_soma.filter(pl.col("Seer_ID").is_not_null() & pl.col("Soma_ID").is_not_null()).select(pl.col("Uni_Prot_ID").n_unique()).collect()[0,0]
    seer_soma_well_covered_count = seer_soma_well_covered.filter(pl.col("Soma_ID").is_not_null() & pl.col("Seer_ID").is_not_null()).select(pl.col("Uni_Prot_ID").n_unique()).collect()[0,0]
    olink_soma_count = olink_soma.filter(pl.col("Olink_ID").is_not_null() & pl.col("Soma_ID").is_not_null()).select(pl.col("Uni_Prot_ID").n_unique()).collect()[0,0]
    olink_soma_well_covered_count = olink_soma_well_covered.filter(pl.col("Olink_ID").is_not_null() & pl.col("Soma_ID").is_not_null()).select(pl.col("Uni_Prot_ID").n_unique()).collect()[0,0]
    
    all_technologies_count = all_technologies.filter(pl.col("Seer_ID").is_not_null() & pl.col("Olink_ID").is_not_null() & pl.col("Soma_ID").is_not_null()).select(pl.col("Uni_Prot_ID").n_unique()).collect()[0,0]
    all_technologies_well_covered_count = all_technologies_well_covered.filter(pl.col("Seer_ID").is_not_null() & pl.col("Olink_ID").is_not_null() & pl.col("Soma_ID").is_not_null()).select(pl.col("Uni_Prot_ID").n_unique()).collect()[0,0]
    
    unique_atleast_two_technologies_count = atleast_two_technologies.select(pl.col("Uni_Prot_ID").n_unique()).collect()[0,0]
    unique_atleast_two_technologies_well_covered_count = atleast_two_technologies_well_covered.select(pl.col("Uni_Prot_ID").n_unique()).collect()[0,0]
    
    print(f"-"*50)
    print(f"Filters Used:")
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
    
    print(f"-"*50)
    print(f"Proteins Covered by All Three Technologies (Well Covered : Total Proteins):")
    print(f"-"*50)
    print(f"Seer + Olink + Soma: {all_technologies_well_covered_count} : {all_technologies_count}")
    
    
if __name__ == "__main__":
    main()