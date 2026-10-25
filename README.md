# Layer-wise Topological Interpretation of Protein Language Models through Semantic Visualization

This repository contains the core implementation scripts, pipelines, and visual examples associated with our study on layer-wise protein language model (pLM) representations and semantic visualization.

---

## 🔬 Overview & Contributions

The study has two complementary components:

1. **Layer-wise Diagnostic & Interpretability Advisor:** We systematically probe intermediate transformer layers across modern protein language models (ESM-2, ProtT5, and ProtBERT). By evaluating challenging protein pairs exhibiting high sequence similarity yet structural divergence, this framework serves as an interpretive and diagnostic aid to examine how internal representations balance sequence fidelity versus 3D topological preservation; it is not intended as a general-purpose alignment benchmark or black-box predictor.

2. **Semantic Coloring & Visualization Framework:** We propose a structure-informed visualization paradigm that translates high-dimensional per-residue embeddings into perceptual CIELAB colors through a three-stage pipeline:
   - **Fuzzy Clustering:** Grouping per-residue embeddings into latent clusters using Fuzzy C-Means.
   - **Cluster-Center Optimization:** Mapping and optimizing only the cluster centroids into the CIELAB color space via NSGA-III multi-objective optimization to guarantee maximal perceptual contrast and rank-order distance preservation.
   - **Color Propagation:** Calculating the continuous color of every individual residue as a weighted combination of cluster centroid colors, driven by its fuzzy membership degrees.
   
   This enables intuitive, residue-level visual inspection of functional and structural properties beyond classical monochrome alignments.

---

## 🎨 Semantic Coloring Pipeline

The end-to-end framework mapping layer embeddings into optimized perceptual color palettes:

<p align="center">
  <img src="pipeline.png" alt="Semantic Coloring Pipeline" width="850"/>
</p>

**Figure 1.** The semantic coloring pipeline consists of three main stages: (1) **Clustering:** Application of fuzzy clustering on embedding vectors to identify latent structures; (2) **Coloring:** Optimization of cluster centers in the CIELAB color space, constrained by rank-order preservation and visual contrast requirements; and (3) **Color Propagation:** Final assignment of color values to all data points based on their respective membership degrees to the optimized clusters.

---

## 📊 Layer-wise Network Visualizations

The following network visualizations show the semantic relationships and structural alignments obtained at five selected model layers (**Layers 3, 12, 16, 20, and 24**). They illustrate how the network representation evolves across the model depth.

### Layer 3
<p align="center">
  <img src="layer_03.png" alt="Network visualization at Layer 3" width="700">
</p>

### Layer 12
<p align="center">
  <img src="layer_12.png" alt="Network visualization at Layer 12" width="700">
</p>

### Layer 16
<p align="center">
  <img src="layer_16.png" alt="Network visualization at Layer 16" width="700">
</p>

### Layer 20
<p align="center">
  <img src="layer_20.png" alt="Network visualization at Layer 20" width="700">
</p>

### Layer 24
<p align="center">
  <img src="layer_24.png" alt="Network visualization at Layer 24" width="700">
</p>

**Figure 6. Residue-level graph visualization of ProtT5 attention across representative layers for the 6A0W sequence.** Weighted residue-level graphs were derived from ProtT5 attention matrices. Nodes correspond to residues; edges represent attention-derived contextual associations; edge thickness is proportional to attention weight; node colors indicate embedding-based clusters; and node size reflects betweenness centrality. The observed topology changes across layers: Layer 3 is dominated by mostly local connectivity; Layers 12 and 16 show increased centralization and hub formation; and Layers 20 and 24 display more modular organization, with bridge residues connecting separate clusters. These patterns suggest that ProtT5 progressively reorganizes residue-level information from local contextual dependencies toward a more globally structured representation.

---

## 📂 Repository Structure

1. **`get_data_from_RCSB.py`**  
   - Retrieves protein sequence pairs from the RCSB PDB database using `SequenceQuery` and submits structural alignment jobs via the `smith-waterman-3d` API to filter twilight-zone pairs.

2. **`filter_after_get_data_from_RCSB.py`**  
   - Post-processes and cleans the retrieved dataset by removing duplicate and symmetric pairs (yielding the filtered dataset of $N=1,133$ pairs: 900 training and 233 test pairs).

3. **`extract_embedding.py`**  
   - Extracts per-residue hidden state embeddings from the encoder-only **ProtT5** model (`Rostlab/prot_t5_xl_half_uniref50-enc`).

4. **`extract_semantic_color.py`**  
   - Implements the core semantic coloring pipeline:
     - Maps per-residue embeddings into clusters using **Fuzzy C-Means**.
     - Optimizes color palettes in the **CIELAB** color space using **NSGA-III**.
     - Performs Smith-Waterman alignment on semantic colors with CIEDE2000 scoring.
     - Optimizes color thresholds via **Bayesian Optimization** on the training set.

---

## 📋 Citation

If you use this code or findings in your research, please cite our paper:

> [Add full citation details here once available]
