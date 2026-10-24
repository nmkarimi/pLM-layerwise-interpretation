# Graph-Transformer-Protein-Design

This repository contains the core implementation scripts for our study on layer-wise protein language model representations, semantic coloring, and structural alignment in the Twilight Zone.

## Repository Structure

1. **`get_data_from_RCSB.py`** 
   - Retrieves protein sequence pairs from the RCSB PDB database using `SequenceQuery` (identity cutoff 0.7) and submits structural alignment jobs via the `smith-waterman-3d` API to filter pairs within the Twilight Zone ($\text{RMSD} > 3.0\text{ Å}$ and $\text{Sequence Similarity} > 0.7$).

2. **`filter_after_get_data_from_RCSB.py`**
   - Post-processes and cleans the retrieved dataset by removing duplicate and symmetric pairs (yielding the final filtered dataset of $N=1,133$ pairs).

3. **`extract_embedding.py`**
   - Extracts per-residue hidden state embeddings from the encoder-only **ProtT5** model (`Rostlab/prot_t5_xl_half_uniref50-enc`, Layer 24).

4. **`extract_semantic_color.py`**
   - Implements the core semantic coloring pipeline:
     - Maps per-residue embeddings into 7 clusters using **Fuzzy C-Means**.
     - Optimizes color palettes in the **CIELAB** color space using **NSGA-III** to maintain structural distance and target $\Delta E$ ranges ($30\text{--}45$).
     - Performs Smith-Waterman alignment on semantic colors with CIEDE2000 scoring.
     - Optimizes the color threshold via **Bayesian Optimization** on the training set (900 pairs) for maximum correlation with TM-score, and evaluates on the test set (233 pairs).
