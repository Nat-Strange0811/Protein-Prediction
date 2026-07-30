"""
One-off panel-building script.
Run ONCE to generate panel_<N>.txt — a static list of UniProt IDs.
Downstream pipeline (extract_panel_embeddings) never needs to know
how this list was constructed.

Tier 1: mechanistic secretory/export machinery (gene-symbol -> UniProt lookup)
Tier 2: agnostic general-binding panel (GO-slim cellular-component stratified sample)
"""

import logging
import os
import random
import time

import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

UNIPROT_SEARCH_URL = "https://rest.uniprot.org/uniprotkb/search"
UNIPROT_GO_STREAM_URL = "https://rest.uniprot.org/uniprotkb/stream"

RANDOM_SEED = 42  # fixed — panel must be reproducible
random.seed(RANDOM_SEED)

# ---------------------------------------------------------------------------
# TIER 1 — mechanistic secretory/export machinery, by gene symbol
# ---------------------------------------------------------------------------

TIER1_GENE_SYMBOLS = {
    "translocon": ["SEC61A1", "SEC61B", "SRP54", "SRP9", "SRP14"],
    "copii_coat": ["SEC23A", "SEC24A", "SEC24B", "SAR1A"],
    "golgi_unconventional": ["GORASP1", "GORASP2"],
    "autophagy_secretion": ["ATG5", "ATG7", "ATG16L1", "SEC22B"],
    "escrt_exosome": ["TSG101", "CHMP4B", "VPS4A"],
    "signal_peptidase": ["SPCS1", "SPCS3"],
    "carrier_stabilization": ["ALB", "TTR", "SERPINA1"],
}


def resolve_gene_to_uniprot(gene_symbol: str, organism_id: int = 9606) -> str | None:
    """Resolve a human gene symbol to a single reviewed (Swiss-Prot) UniProt accession."""
    query = f"gene:{gene_symbol} AND organism_id:{organism_id} AND reviewed:true"
    params = {"query": query, "fields": "accession,gene_names", "format": "json", "size": 5}

    resp = requests.get(UNIPROT_SEARCH_URL, params=params, timeout=15)
    resp.raise_for_status()
    results = resp.json().get("results", [])

    if not results:
        logger.warning("No reviewed UniProt entry found for gene %s", gene_symbol)
        return None

    if len(results) > 1:
        logger.warning(
            "Multiple reviewed entries for gene %s — taking first (%s). Verify manually.",
            gene_symbol,
            results[0]["primaryAccession"],
        )

    return results[0]["primaryAccession"]


def build_tier1_panel() -> list[str]:
    accessions = []
    for stage, genes in TIER1_GENE_SYMBOLS.items():
        for gene in genes:
            acc = resolve_gene_to_uniprot(gene)
            if acc is None:
                logger.error("Skipping unresolved gene %s (stage=%s)", gene, stage)
                continue
            accessions.append(acc)
            time.sleep(0.2)  # be polite to the API
    logger.info("Tier 1 (secretory) resolved: %d / %d genes",
                len(accessions), sum(len(v) for v in TIER1_GENE_SYMBOLS.values()))
    return accessions


# ---------------------------------------------------------------------------
# TIER 2 — agnostic general-binding panel, GO-slim compartment stratified
# ---------------------------------------------------------------------------

# GO-slim cellular component terms used as strata (generic GO slim subset)
GO_SLIM_COMPARTMENTS = {
    "nucleus": "GO:0005634",
    "cytoplasm": "GO:0005737",
    "plasma_membrane": "GO:0005886",
    "mitochondrion": "GO:0005739",
    "endoplasmic_reticulum": "GO:0005783",
    "golgi_apparatus": "GO:0005794",
    "extracellular_region": "GO:0005576",
    "cytoskeleton": "GO:0005856",
}

N_PER_STRATUM = 12  # ~100 total across 8 strata

# Pure outlier guard only — NOT a representativeness filter. A tight length
# window (e.g. 100-1200) would bias the "general" panel toward medium-length
# proteins and quietly contradict the agnostic-sampling goal. This cap exists
# only to exclude the rare pathological cases (titin-scale, thousands of
# residues) that would blow up D-SCRIPT's contact-map memory (scales with
# L_query x L_panel_member) for every future query protein scored against
# that panel slot. Compute concerns for the query side are handled at
# inference time via D-SCRIPT's --blocks memory management, not here.
MAX_LEN = 2000


def fetch_candidates_for_go_term(go_id: str, organism_id: int = 9606, batch_size: int = 500) -> list[dict]:
    """Pull a batch of reviewed human proteins annotated to a given GO-slim term,
    with length already included so we can filter without a second call per protein."""
    query = f"go:{go_id.removeprefix('GO:')} AND organism_id:{organism_id} AND reviewed:true"
    params = {
        "query": query,
        "fields": "accession,length",
        "format": "json",
        "size": batch_size,
    }
    resp = requests.get(UNIPROT_SEARCH_URL, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json().get("results", [])


def build_tier2_panel() -> list[str]:
    accessions = []
    seen = set()
    for compartment, go_id in GO_SLIM_COMPARTMENTS.items():
        candidates = fetch_candidates_for_go_term(go_id)

        # Only exclude pathological-length outliers — no lower bound, and the
        # upper bound is set well above typical protein length so it almost
        # never triggers. Deliberately NOT filtering by annotation richness/
        # citations, and deliberately NOT narrowing to a "typical" length
        # range, since either would bias the sample away from a genuine
        # cross-section of the compartment. Missing length data defaults to
        # MAX_LEN + 1 (excluded) rather than 0 (included), since we can't
        # vouch for a protein's size when UniProt didn't report one.
        filtered = [
            c["primaryAccession"]
            for c in candidates
            if c["primaryAccession"] not in seen
            and c.get("sequence", {}).get("length", MAX_LEN + 1) <= MAX_LEN
        ]

        if len(filtered) < N_PER_STRATUM:
            logger.warning(
                "Compartment %s: only %d candidates passed length filter (wanted %d)",
                compartment, len(filtered), N_PER_STRATUM,
            )

        sampled = random.sample(filtered, min(N_PER_STRATUM, len(filtered)))
        seen.update(sampled)
        accessions.extend(sampled)
        logger.info("Compartment %s: sampled %d proteins", compartment, len(sampled))
        time.sleep(0.2)

    return accessions


# ---------------------------------------------------------------------------
# Build and write panel
# ---------------------------------------------------------------------------

def main(secretory_panel_number: int, general_panel_number: int, output_dir: str):
    tier1 = build_tier1_panel()
    tier2 = build_tier2_panel()

    overlap = set(tier1) & set(tier2)
    if overlap:
        logger.warning("Overlap between tier1 and tier2: %s — removing duplicates from tier2", overlap)
        tier2 = [a for a in tier2 if a not in overlap]

    secretory_file = f"{output_dir}/panel_{secretory_panel_number}.txt"
    general_file = f"{output_dir}/panel_{general_panel_number}.txt"
    
    if os.path.exists(secretory_file) or os.path.exists(general_file):
        logger.error("Output files already exist. Please remove them before running this script.")
        return
    else:
        os.makedirs(output_dir, exist_ok=True)
        
    with open(secretory_file, "w") as f:
        for acc in tier1:
            f.write(f"{acc}\n")

    with open(general_file, "w") as f:
        for acc in tier2:
            f.write(f"{acc}\n")

    logger.info(
        "Wrote %d secretory IDs to %s (panel_%d) and %d general IDs to %s (panel_%d), seed=%d",
        len(tier1), secretory_file, secretory_panel_number,
        len(tier2), general_file, general_panel_number, RANDOM_SEED,
    )


if __name__ == "__main__":
    SECRETORY_PANEL_NUMBER = 1
    GENERAL_PANEL_NUMBER = 2
    OUTPUT_DIR = "/data/PHURI-Langenberg/people/Nat/Protein-Prediction/Data/Panels"
    main(SECRETORY_PANEL_NUMBER, GENERAL_PANEL_NUMBER, OUTPUT_DIR)
    
    