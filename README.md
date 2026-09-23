# Mitochondrial RNA Lifespan Profiling & Alignment Pipeline

An end-to-end bioinformatic pipeline designed to profile mitochondrial RNA (mtRNA) expression across human lifespan cohorts from high-throughput FASTQ datasets. 

The tool implements a dual-verification sequence alignment algorithm—combining positional $k$-mer seed indexing with dynamic programming Levenshtein edit distance filtering—to suppress Nuclear Mitochondrial Pseudogene (NUMT) artifacts. It then applies unsupervised $K$-Means clustering ($K=3$) on standardized genomic coverage parameters to classify bioenergetic phenotypes across aging stages.

---

## 🌟 Key Features

* **Dual-Verification Alignment Filtering:** Combines positional $k$-mer seed indexing ($k=5$, threshold $\ge 0.7$) with dynamic programming Levenshtein edit distance calculation ($e \le 2$) to ensure high alignment identity and eliminate NUMT noise.
* **Circular Origin Wrap Buffer:** Appends a 500 bp buffer prefix to the origin of the human mitochondrial reference to seamlessly map reads straddling the circular junction.
* **Strand-Aware Alignment:** Automatically evaluates both forward and reverse-complement read strands against the mitochondrial reference sequence.
* **Algorithm R Reservoir Sampling:** Uniformly subsamples large FASTQ files up to a user-defined threshold (`MAX_READS = 500,000`) in single-pass $O(N)$ time.
* **1D Positional Coverage Accumulation:** Tracks nucleotide-level read mapping across the 16,569 bp human mitochondrial genome to measure **Coverage Breadth (%)** and **Mean Coverage Depth**.
* **Unsupervised $K$-Means Phenotypic Clustering:** Uses `StandardScaler` feature normalization and $K$-Means ($K=3$) on coverage breadth and depth to discover emergent lifespan profiles without observational bias.
* **Control Baseline Analysis:** Includes an unfiltered raw baseline control option to verify that sample separation is driven by mitochondrial transcriptional dynamics rather than library composition or raw sequence depth.
* **Automated Caching & Reporting:** Caches processed results to `analysis_cache.json` for rapid re-visualization and generates detailed text reports and high-resolution plots.

---

## 📁 Repository Structure

```text
.
├── main.py                 # Primary pipeline script
├── reference/
│   └── reference.fasta     # Human mitochondrial reference sequence (NC_012920.1)
├── samples/                # Directory containing input .fastq files
│   ├── w21_SRR6706578_500k.fastq
│   └── ...
├── output_plots/           # Output directory for generated figures (.png)
├── output_reports/          # Output directory for text summary reports
│   └── sample_statistics.txt
├── analysis_cache.json     # Cached alignment outputs (auto-generated)
└── README.md
