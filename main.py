from collections import defaultdict
import numpy as np
import random
import os
import time
import json

# Machine Learning components for Clustering
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Fix PyCharm rendering bug by forcing a stable standalone window backend
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# =====================================================
# PARAMETERS & PATHS
# =====================================================

MAX_READS = 500000
K = 10
ALIGNMENT_THRESHOLD = 0.7
MAX_EDIT_DISTANCE = 2

SAMPLES_FOLDER = "samples"
REFERENCE_FILE = os.path.join("reference", "reference.fasta")
CACHE_FILE = "analysis_cache.json"

OUTPUT_PLOTS_DIR = "output_plots"
OUTPUT_REPORTS_DIR = "output_reports"
STATS_REPORT_FILE = os.path.join(OUTPUT_REPORTS_DIR, "sample_statistics.txt")

random.seed(42)


# =====================================================
# FILTER MODE SELECTOR PROMPT
# =====================================================

def prompt_filter_mode():
    """Prompts the user to select the analysis or filtering option via console input 1-4."""
    print("\n" + "=" * 65)
    print("SELECT ALIGNMENT FILTERING OR CONTROL MODE")
    print("=" * 65)
    print("  1. K-mer Alignment Score only (Score >= Threshold)")
    print("  2. Levenshtein Edit Distance only (Edit Dist <= Max Distance)")
    print("  3. Both Dual-Verification (K-mer Score AND Edit Distance)")
    print("  4. Unfiltered Raw Baseline Control (K-Means on Raw Sequence Metrics)")

    while True:
        choice = input("\nEnter choice (1, 2, 3, or 4) [default: 3]: ").strip()
        if choice == "1":
            return "kmer"
        elif choice == "2":
            return "edit_distance"
        elif choice == "3" or choice == "":
            return "both"
        elif choice == "4":
            return "raw_unfiltered"
        else:
            print("[Error] Invalid selection. Please enter 1, 2, 3, or 4.")


# =====================================================
# REFERENCE LOADING
# =====================================================

def load_reference(path, circular_wrap=500):
    seq = ""
    with open(path, "r") as f:
        for line in f:
            if not line.startswith(">"):
                seq += line.strip()
    return seq + seq[:circular_wrap]


# =====================================================
# KMER INDEX
# =====================================================

def build_kmer_index(reference, k):
    index = defaultdict(list)
    for i in range(len(reference) - k +1):
        kmer = reference[i:i + k]
        index[kmer].append(i)
    return index


# =====================================================
# FASTQ READER
# =====================================================

def read_fastq(path, max_reads):
    reservoir = []
    with open(path, "r") as f:
        i = 0
        while True:
            header = f.readline()
            if not header:
                break
            header = header.strip()
            seq = f.readline().strip()
            f.readline()  # +
            f.readline()  # quality

            read_data = {"header": header, "seq": seq}

            if i < max_reads:
                reservoir.append(read_data)
            else:
                j = random.randint(0, i)
                if j < max_reads:
                    reservoir[j] = read_data
            i += 1
    return reservoir


# =====================================================
# REVERSE COMPLEMENT
# =====================================================

def reverse_complement(seq):
    comp = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', 'N': 'N'}
    return ''.join(comp[b] for b in reversed(seq))


# =====================================================
# LEVENSHTEIN DYNAMIC PROGRAMMING EDIT DISTANCE
# =====================================================

def compute_edit_distance(s1, s2):
    """Calculates Levenshtein edit distance between read sequence s1 and reference segment s2."""
    if len(s1) < len(s2):
        return compute_edit_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (0 if c1 == c2 else 1)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


# =====================================================
# ALIGNMENT SCORE
# =====================================================

def score_alignment(read, segment):
    matches = sum(1 for a, b in zip(read, segment) if a == b)
    return matches / len(read)


# =====================================================
# KMER ALIGNER
# =====================================================

def align_read_kmer(read, reference, kmer_index, k):
    candidates = defaultdict(int)
    for i in range(len(read) - k):
        kmer = read[i:i + k]
        if kmer in kmer_index:
            for pos in kmer_index[kmer]:
                candidates[pos - i] += 1

    if not candidates:
        return -1, -1

    best_positions = sorted(candidates.items(), key=lambda x: -x[1])[:10]
    best_score = -1
    best_pos = -1

    for pos, _ in best_positions:
        if pos < 0 or pos + len(read) >= len(reference):
            continue
        segment = reference[pos:pos + len(read)]
        score = score_alignment(read, segment)
        if score > best_score:
            best_score = score
            best_pos = pos

    return best_pos, best_score


# =====================================================
# SAMPLE ANALYSIS (FILTERED & UNFILTERED BASELINE)
# =====================================================

def analyze_sample(reads, reference, kmer_index, k, filter_mode="both"):
    aligned_reads = 0
    scores = []
    edit_distances = []
    gc_contents = []
    read_lengths = []

    coverage_array = np.zeros(len(reference), dtype=int)

    for read in reads:
        seq = read["seq"]
        read_lengths.append(len(seq))

        g_count = seq.count('G') + seq.count('g')
        c_count = seq.count('C') + seq.count('c')
        gc_contents.append((g_count + c_count) / len(seq) * 100 if seq else 0)

        # Fast path for raw baseline: skip compute-heavy sequence alignment checks
        if filter_mode == "raw_unfiltered":
            continue

        rc = reverse_complement(seq)
        best_pos_f, score_f = align_read_kmer(seq, reference, kmer_index, k)
        best_pos_r, score_r = align_read_kmer(rc, reference, kmer_index, k)

        if score_f >= score_r:
            best_pos, score, aligned_seq = best_pos_f, score_f, seq
        else:
            best_pos, score, aligned_seq = best_pos_r, score_r, rc

        if best_pos != -1 and best_pos + len(aligned_seq) <= len(reference):
            segment = reference[best_pos:best_pos + len(aligned_seq)]
            edit_dist = compute_edit_distance(aligned_seq, segment)
        else:
            edit_dist = float('inf')

        kmer_pass = (score >= ALIGNMENT_THRESHOLD)
        edit_pass = (edit_dist <= MAX_EDIT_DISTANCE)

        if filter_mode == "kmer":
            is_aligned = kmer_pass
        elif filter_mode == "edit_distance":
            is_aligned = edit_pass
        elif filter_mode == "both":
            is_aligned = kmer_pass and edit_pass
        else:
            raise ValueError(f"Unknown filter_mode: {filter_mode}")

        if is_aligned:
            aligned_reads += 1
            scores.append(score)
            if edit_dist != float('inf'):
                edit_distances.append(edit_dist)

            for bp_idx in range(best_pos, best_pos + len(seq)):
                if 0 <= bp_idx < len(reference):
                    coverage_array[bp_idx] += 1

    if filter_mode == "raw_unfiltered":
        aligned_reads = len(reads)

    mean_score = float(np.mean(scores)) if scores else 0.0
    min_score = float(np.min(scores)) if scores else 0.0
    max_score = float(np.max(scores)) if scores else 0.0

    mean_edit_dist = float(np.mean(edit_distances)) if edit_distances else 0.0
    min_edit_dist = int(np.min(edit_distances)) if edit_distances else 0
    max_edit_dist = int(np.max(edit_distances)) if edit_distances else 0

    return {
        "aligned_reads": aligned_reads,
        "total_reads": len(reads),
        "mt_fraction": aligned_reads / len(reads) if reads else 0,
        "mean_score": mean_score,
        "min_score": min_score,
        "max_score": max_score,
        "mean_edit_distance": mean_edit_dist,
        "min_edit_distance": min_edit_dist,
        "max_edit_distance": max_edit_dist,
        "scores": scores,
        "edit_distances": edit_distances,
        "gc_contents": gc_contents,
        "read_lengths": read_lengths,
        "coverage_array": coverage_array.tolist()
    }


# =====================================================
# UNFILTERED RAW BASELINE K-MEANS CLUSTERING
# =====================================================

def plot_raw_kmeans_control(plot_data, output_dir):
    """Clusters samples purely using baseline unfiltered metrics (Total Reads vs Mean GC Content %)."""
    os.makedirs(output_dir, exist_ok=True)
    sample_names = sorted(plot_data.keys())

    features = []
    raw_metrics = []

    for sample in sample_names:
        total_reads = plot_data[sample]["total_reads"]
        mean_gc = float(np.mean(plot_data[sample]["gc_contents"])) if plot_data[sample]["gc_contents"] else 0.0

        raw_metrics.append((total_reads, mean_gc))
        features.append([total_reads, mean_gc])

    X = np.array(features)
    X_raw = np.array(raw_metrics)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)

    plt.figure(num="Unfiltered Baseline K-Means", figsize=(9, 7))

    colors = ['#FF5733', '#2ECC71', '#3498DB']
    markers = ['o', 's', '^']

    for cluster_idx in range(3):
        indices = np.where(cluster_labels == cluster_idx)[0]
        plt.scatter(
            X_raw[indices, 0],
            X_raw[indices, 1],
            c=colors[cluster_idx],
            marker=markers[cluster_idx],
            s=140,
            edgecolor='black',
            label=f'Cluster {cluster_idx + 1}',
            alpha=0.9,
            zorder=3
        )

    for idx, sample in enumerate(sample_names):
        clean_label = sample.replace('_500k.fastq', '')
        plt.annotate(
            clean_label,
            (X_raw[idx, 0], X_raw[idx, 1]),
            textcoords="offset points",
            xytext=(0, 10),
            ha='center',
            fontsize=8,
            fontweight='semibold',
            zorder=4
        )

    plt.title("Baseline Unfiltered Sample Grouping (Control)\n(K-Means on Raw Read Count vs Mean GC %)", fontsize=11,
              fontweight='bold', pad=15)
    plt.xlabel("Total Raw Read Count", fontweight='bold')
    plt.ylabel("Mean GC Content (%)", fontweight='bold')
    plt.legend(loc='best', frameon=True, shadow=True)
    plt.grid(True, linestyle=':', alpha=0.6, zorder=1)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "raw_unfiltered_kmeans_clustering.png"), dpi=300)


# =====================================================
# 2D SCATTER PLOT K-MEANS CLUSTERING (MITOCHONDRIAL)
# =====================================================

def plot_kmeans_coverage(plot_data, output_dir):
    """Clusters and plots each sample using mitochondrial genome coverage breadth and depth."""
    os.makedirs(output_dir, exist_ok=True)
    sample_names = sorted(plot_data.keys())

    features = []
    raw_metrics = []

    for sample in sample_names:
        cov_array = np.array(plot_data[sample]["coverage_array"])

        coverage_rate = (np.sum(cov_array > 0) / len(cov_array)) * 100 if len(cov_array) > 0 else 0.0
        mean_depth = np.mean(cov_array) if len(cov_array) > 0 else 0.0

        raw_metrics.append((coverage_rate, mean_depth))
        features.append([coverage_rate, mean_depth])

    X = np.array(features)
    X_raw = np.array(raw_metrics)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)

    plt.figure(num="6. KMeans Coverage Clustering", figsize=(9, 7))

    colors = ['#FF5733', '#2ECC71', '#3498DB']
    markers = ['o', 's', '^']

    for cluster_idx in range(3):
        indices = np.where(cluster_labels == cluster_idx)[0]
        plt.scatter(
            X_raw[indices, 0],
            X_raw[indices, 1],
            c=colors[cluster_idx],
            marker=markers[cluster_idx],
            s=140,
            edgecolor='black',
            label=f'Cluster {cluster_idx + 1}',
            alpha=0.9,
            zorder=3
        )

    for idx, sample in enumerate(sample_names):
        clean_label = sample.replace('_500k.fastq', '')
        plt.annotate(
            clean_label,
            (X_raw[idx, 0], X_raw[idx, 1]),
            textcoords="offset points",
            xytext=(0, 10),
            ha='center',
            fontsize=8,
            fontweight='semibold',
            zorder=4
        )

    plt.title("Sample Grouping Matrix via Concrete Coverage Metrics\n(K-Means Clustering)", fontsize=11,
              fontweight='bold', pad=15)
    plt.xlabel("Coverage Breadth (% of Mitochondrial Genome Covered)", fontweight='bold')
    plt.ylabel("Mean Coverage Depth (Average Reads per Base Position)", fontweight='bold')
    plt.legend(loc='best', frameon=True, shadow=True)
    plt.grid(True, linestyle=':', alpha=0.6, zorder=1)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "6_kmeans_coverage_clustering.png"), dpi=300)


# =====================================================
# INDEPENDENT GRAPH GENERATION VISUALIZER
# =====================================================

def plot_all_data(plot_data, output_dir):
    """Generates and saves separate independent windows for sequence data metrics."""
    os.makedirs(output_dir, exist_ok=True)
    sorted_samples = sorted(plot_data.keys())

    # Window 1: Aligned Reads Bar Graph
    plt.figure(num="1. Aligned Reads Count", figsize=(8, 6))
    aligned_counts = [plot_data[s]['aligned_reads'] for s in sorted_samples]
    bars = plt.bar(sorted_samples, aligned_counts, color='skyblue', edgecolor='black')
    plt.title("Aligned Reads per Sample (Sorted)", fontsize=12, fontweight='bold')
    plt.ylabel("Read Count", fontweight='bold')
    plt.xticks(rotation=30, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2,
                 yval + (max(aligned_counts) * 0.01 if max(aligned_counts) > 0 else 1),
                 f'{int(yval):,}', ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "1_aligned_reads_count.png"), dpi=300)

    # Window 2: GC Content Distribution Histogram
    plt.figure(num="2. GC Content Distribution", figsize=(8, 6))
    for sample in sorted_samples:
        plt.hist(plot_data[sample]['gc_contents'], bins=20, alpha=0.5, label=sample, histtype='stepfilled')
    plt.title("GC Content Per-Read Distribution", fontsize=12, fontweight='bold')
    plt.xlabel("GC Content (%)", fontweight='bold')
    plt.ylabel("Count", fontweight='bold')
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "2_gc_content_distribution.png"), dpi=300)

    # Window 3: Read Length Distribution Histogram
    plt.figure(num="3. Read Length Distribution", figsize=(8, 6))
    for sample in sorted_samples:
        plt.hist(plot_data[sample]['read_lengths'], bins=15, alpha=0.5, label=sample, histtype='step', linewidth=2)
    plt.title("Read Length Distribution", fontsize=12, fontweight='bold')
    plt.xlabel("Length (bp)", fontweight='bold')
    plt.ylabel("Count", fontweight='bold')
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "3_read_length_distribution.png"), dpi=300)

    # Window 4: Alignment Score Density Distribution
    plt.figure(num="4. Alignment Score Distribution", figsize=(8, 6))
    for sample in sorted_samples:
        if plot_data[sample]['scores']:
            plt.hist(plot_data[sample]['scores'], bins=15, alpha=0.5, label=sample, density=True)
    plt.title("Alignment Score Distribution Density", fontsize=12, fontweight='bold')
    plt.xlabel("Identity Match Score", fontweight='bold')
    plt.ylabel("Density", fontweight='bold')
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "4_alignment_score_distribution.png"), dpi=300)

    # Window 5: Edit Distance Frequency Distribution
    plt.figure(num="5. Edit Distance Distribution", figsize=(8, 6))
    max_dist_found = max(
        [max(plot_data[s]['edit_distances']) if plot_data[s]['edit_distances'] else 0 for s in sorted_samples],
        default=MAX_EDIT_DISTANCE)
    bins = np.arange(-0.5, max_dist_found + 1.5, 1.0)

    for sample in sorted_samples:
        edits = plot_data[sample].get('edit_distances', [])
        if edits:
            plt.hist(edits, bins=bins, alpha=0.5, label=sample, density=True, rwidth=0.85)

    plt.title("Per-Read Levenshtein Edit Distance Distribution Density", fontsize=12, fontweight='bold')
    plt.xlabel("Edit Distance (Mismatches / Indels)", fontweight='bold')
    plt.ylabel("Density", fontweight='bold')
    plt.xticks(range(0, max_dist_found + 1))
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "5_edit_distance_distribution.png"), dpi=300)


# =====================================================
# SUMMARY STATISTICS EXPORTER
# =====================================================

def save_summary_statistics(plot_data, filepath, filter_mode):
    """Calculates coverage parameters and saves all alignment & edit distance metrics to a text file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    sorted_samples = sorted(plot_data.keys())

    with open(filepath, "w") as f:
        f.write("=" * 155 + "\n")
        f.write(" " * 45 + f"MITOCHONDRIAL RNA ALIGNMENT REPORT (FILTER MODE: {filter_mode.upper()})\n")
        f.write("=" * 155 + "\n\n")

        header = (
            f"{'Sample Name':<28} | {'Total':<7} | {'Aligned':<7} | "
            f"{'MT Frac':<7} | {'Mean Scr':<8} | {'Min Scr':<8} | {'Max Scr':<8} | "
            f"{'Mean Edit':<9} | {'Min Edit':<8} | {'Max Edit':<8} | "
            f"{'Cov %':<7} | {'Mean Depth':<10}\n"
        )
        f.write(header)
        f.write("-" * 155 + "\n")

        for sample in sorted_samples:
            data = plot_data[sample]

            cov_array = np.array(data["coverage_array"])
            cov_breadth = (np.sum(cov_array > 0) / len(cov_array)) * 100 if len(cov_array) > 0 else 0.0
            mean_depth = float(np.mean(cov_array)) if len(cov_array) > 0 else 0.0

            scores = data.get("scores", [])
            edits = data.get("edit_distances", [])

            mean_s = data.get("mean_score", float(np.mean(scores)) if scores else 0.0)
            min_s = data.get("min_score", float(np.min(scores)) if scores else 0.0)
            max_s = data.get("max_score", float(np.max(scores)) if scores else 0.0)

            mean_e = data.get("mean_edit_distance", float(np.mean(edits)) if edits else 0.0)
            min_e = data.get("min_edit_distance", int(np.min(edits)) if edits else 0)
            max_e = data.get("max_edit_distance", int(np.max(edits)) if edits else 0)

            line = (
                f"{sample:<28} | {data['total_reads']:<7d} | {data['aligned_reads']:<7d} | "
                f"{data['mt_fraction']:<7.4f} | {mean_s:<8.4f} | {min_s:<8.4f} | {max_s:<8.4f} | "
                f"{mean_e:<9.2f} | {min_e:<8d} | {max_e:<8d} | "
                f"{cov_breadth:<7.2f}% | {mean_depth:<10.2f}\n"
            )
            f.write(line)

        f.write("-" * 155 + "\n")

    print(f"\n[Info] Comprehensive summary statistics report saved to: '{filepath}'")


# =====================================================
# MAIN RUNTIME CONTROL
# =====================================================

def main():
    plot_data = {}
    use_cache = False
    filter_mode = prompt_filter_mode()

    if filter_mode != "raw_unfiltered" and os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r") as f:
                temp_data = json.load(f)

            first_entry = next(iter(temp_data.values()))
            if "coverage_array" in first_entry:
                print("=" * 60)
                print("FOUND VALID PREVIOUSLY SAVED RUN DATA WITH COVERAGE ARRAYS!")
                print("=" * 60)
                choice = input("Load database cache and view updated graphs instantly? (y/n): ").strip().lower()
                if choice in ['y', 'yes']:
                    plot_data = temp_data
                    use_cache = True
            else:
                print("\n[Notice] Old cache file detected without genomic coverage metrics. Overwriting database...")
        except Exception:
            print("\n[Notice] Error checking cache file. Re-running full sequence alignment...")

    if not use_cache:
        print(f"\nRunning analysis pipeline with MODE: '{filter_mode.upper()}'...")
        reference = load_reference(REFERENCE_FILE) if filter_mode != "raw_unfiltered" else ""
        kmer_index = build_kmer_index(reference, K) if filter_mode != "raw_unfiltered" else {}

        sample_files = sorted([
            f for f in os.listdir(SAMPLES_FOLDER)
            if f.endswith(".fastq")
        ])

        print(f"\nFound {len(sample_files)} FASTQ files")

        for sample in sample_files:
            sample_path = os.path.join(SAMPLES_FOLDER, sample)
            print("\n" + "=" * 60)
            print(f"Processing: {sample}")

            start = time.time()
            reads = read_fastq(sample_path, MAX_READS)
            result = analyze_sample(reads, reference, kmer_index, K, filter_mode=filter_mode)
            elapsed = time.time() - start

            print(f"Total reads parsed: {result['total_reads']}")
            if filter_mode != "raw_unfiltered":
                print(f"Aligned reads: {result['aligned_reads']}/{result['total_reads']}")
                print(
                    f"Score stats    -> Mean: {result['mean_score']:.4f} | Min: {result['min_score']:.4f} | Max: {result['max_score']:.4f}")
                print(
                    f"Edit Dist stats -> Mean: {result['mean_edit_distance']:.2f} | Min: {result['min_edit_distance']} | Max: {result['max_edit_distance']}")
            print(f"Runtime: {elapsed:.2f} seconds")

            plot_data[sample] = result

        if filter_mode != "raw_unfiltered":
            print(f"\nSaving detailed genomic profiling results to database file: '{CACHE_FILE}'...")
            with open(CACHE_FILE, "w") as f:
                json.dump(plot_data, f)

    if filter_mode == "raw_unfiltered":
        print("\nGenerating Unfiltered Raw Baseline Control K-Means plot...")
        plot_raw_kmeans_control(plot_data, OUTPUT_PLOTS_DIR)
    else:
        for sample, data in plot_data.items():
            scores = data.get("scores", [])
            edits = data.get("edit_distances", [])

            if "min_score" not in data or "max_score" not in data:
                data["min_score"] = float(np.min(scores)) if scores else 0.0
                data["max_score"] = float(np.max(scores)) if scores else 0.0

            if "mean_edit_distance" not in data:
                data["mean_edit_distance"] = float(np.mean(edits)) if edits else 0.0
                data["min_edit_distance"] = int(np.min(edits)) if edits else 0
                data["max_edit_distance"] = int(np.max(edits)) if edits else 0

        save_summary_statistics(plot_data, STATS_REPORT_FILE, filter_mode)

        print("\nGenerating dashboards and saving plots automatically...")
        plot_all_data(plot_data, OUTPUT_PLOTS_DIR)
        plot_kmeans_coverage(plot_data, OUTPUT_PLOTS_DIR)

    print(f"[Info] Plots successfully saved to directory: '{OUTPUT_PLOTS_DIR}/'")

    plt.show()
    print("\nProcess finalized successfully.")


if __name__ == "__main__":
    main()