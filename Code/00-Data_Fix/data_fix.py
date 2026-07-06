import polars as pl
import re

def fix_data():
    
    seer_uniprot = "uniprotswissprot"
    olink_uniprot = "UniProt"
    soma_uniprot = "UniProt"
    
    seer = pl.read_csv("/data/PHURI-Langenberg/people/Nat/Protein-Prediction/Data/Raw/Supplementary_Tables_Seer_pQTL_20251027(ST1).csv", truncate_ragged_lines=True, separator=",", comment_prefix="#", null_values="NA")
    seer.columns = [col.strip() for col in seer.columns]
    seer = seer.with_columns(
        pl.col(pl.String).str.strip_chars()
    ).with_columns(
        pl.when(pl.col(pl.String) == "NA").then(None).otherwise(pl.col(pl.String)).name.keep()
    ).with_columns(
        pl.col(pl.String).str.split("|").name.keep()
    ).with_columns(
        pl.when(pl.col(pl.List(pl.String)).list.len() == 1)
            .then(pl.col(pl.List(pl.String)).list.first().str.split(";"))
            .otherwise(pl.col(pl.List(pl.String)))
            .name.keep()
    ).with_columns(
        pl.when(pl.col(pl.List(pl.String)).list.len() == 1)
            .then(pl.col(pl.List(pl.String)).list.first().str.split("_"))
            .otherwise(pl.col(pl.List(pl.String)))
            .name.keep()
    )

    seer = seer.explode(seer_uniprot)
    
    seer_fails = check_UniProt(seer, seer_uniprot, "Seer")
    seer = seer.filter(~pl.col(seer_uniprot).is_in(seer_fails))
    seer = seer.filter(pl.col(seer_uniprot).is_not_null() & (pl.col(seer_uniprot) != ""))
    
    olink = pl.read_csv("/data/PHURI-Langenberg/people/Nat/Protein-Prediction/Data/Raw/Supplementary_Tables_Seer_pQTL_20251027(ST2).csv", truncate_ragged_lines=True, separator=",", comment_prefix="#", null_values="NA")
    olink.columns = [col.strip() for col in olink.columns]
    olink = olink.with_columns(
        pl.col(pl.String).str.strip_chars()
    ).with_columns(
        pl.when(pl.col(pl.String) == "NA").then(None).otherwise(pl.col(pl.String)).name.keep()
    ).with_columns(
        pl.col(pl.String).str.split("|").name.keep()
    ).with_columns(
        pl.when(pl.col(pl.List(pl.String)).list.len() == 1)
            .then(pl.col(pl.List(pl.String)).list.first().str.split(";"))
            .otherwise(pl.col(pl.List(pl.String)))
            .name.keep()
    ).with_columns(
        pl.when(pl.col(pl.List(pl.String)).list.len() == 1)
            .then(pl.col(pl.List(pl.String)).list.first().str.split("_"))
            .otherwise(pl.col(pl.List(pl.String)))
            .name.keep()
    )
    
    olink = olink.explode(olink_uniprot)
    
    olink_fails = check_UniProt(olink, olink_uniprot, "Olink")
    olink = olink.filter(~pl.col(olink_uniprot).is_in(olink_fails))
    olink = olink.filter(pl.col(olink_uniprot).is_not_null() & (pl.col(olink_uniprot) != ""))
    
    soma = pl.read_csv("/data/PHURI-Langenberg/people/Nat/Protein-Prediction/Data/Raw/Supplementary_Tables_Seer_pQTL_20251027(ST3).csv", truncate_ragged_lines=True, separator=",", comment_prefix="#", null_values="NA")
    soma.columns = [col.strip() for col in soma.columns]
    soma = soma.with_columns(
        pl.col(pl.String).str.strip_chars()
    ).with_columns(
        pl.when(pl.col(pl.String) == "NA").then(None).otherwise(pl.col(pl.String)).name.keep()
    ).with_columns(
        pl.col(pl.String).str.split("|").name.keep()
    ).with_columns(
        pl.when(pl.col(pl.List(pl.String)).list.len() == 1)
            .then(pl.col(pl.List(pl.String)).list.first().str.split(";"))
            .otherwise(pl.col(pl.List(pl.String)))
            .name.keep()
    ).with_columns(
        pl.when(pl.col(pl.List(pl.String)).list.len() == 1)
            .then(pl.col(pl.List(pl.String)).list.first().str.split("_"))
            .otherwise(pl.col(pl.List(pl.String)))
            .name.keep()
    )
    
    soma = soma.explode(soma_uniprot)
    
    soma_fails = check_UniProt(soma, soma_uniprot, "Soma")
    soma = soma.filter(~pl.col(soma_uniprot).is_in(soma_fails))
    
    
    cols_soma = ["SomaId", "TargetFullName", "Target", "UniProt", "EntrezGeneID", "EntrezGeneSymbol"]
    
    seer_csv = seer.with_columns(
        pl.col(pl.List(pl.String)).list.join("|")
    )
    
    olink_csv = olink.with_columns(
        pl.col(pl.List(pl.String)).list.join("|")
    )
    
    soma_csv = soma.with_columns(
        pl.col(pl.List(pl.String)).list.join("|")
    )
    
    
    seer_csv.write_csv("/data/PHURI-Langenberg/people/Nat/Protein-Prediction/Data/Fixed/ST1_Seer_Cleaned.csv")
    olink_csv.write_csv("/data/PHURI-Langenberg/people/Nat/Protein-Prediction/Data/Fixed/ST2_Olink_Cleaned.csv")
    soma_csv.write_csv("/data/PHURI-Langenberg/people/Nat/Protein-Prediction/Data/Fixed/ST3_Soma_Cleaned.csv")

    seer.write_parquet("/data/PHURI-Langenberg/people/Nat/Protein-Prediction/Data/Fixed/ST1_Seer_Cleaned.parquet")
    olink.write_parquet("/data/PHURI-Langenberg/people/Nat/Protein-Prediction/Data/Fixed/ST2_Olink_Cleaned.parquet")
    soma.write_parquet("/data/PHURI-Langenberg/people/Nat/Protein-Prediction/Data/Fixed/ST3_Soma_Cleaned.parquet")
    

def check_UniProt(df, col, name):
    
    UNIPROT_PATTERN = re.compile(r'^([OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2})$')

    uniprot_ids = df.select(pl.col(col)).to_series().explode().unique().to_list()
    
    rejected_ids = [id for id in uniprot_ids if id is not None and not UNIPROT_PATTERN.match(str(id))]

    
    print(f"Rejected UniProt IDs in column '{col}' ({name}): {rejected_ids}")
    
    return rejected_ids




if __name__ == "__main__":
    fix_data()
    
    
