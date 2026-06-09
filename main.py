from collections import defaultdict
import numpy as np
import random
import os
import time
import json  # Added for saving/loading results

# Fix PyCharm rendering bug by forcing a stable standalone window backend
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# =====================================================
# PARAMETERS
# =====================================================

MAX_READS = 500000
K = 10
ALIGNMENT_THRESHOLD = 0.7

SAMPLES_FOLDER = "samples"
REFERENCE_FILE = os.path.join("reference", "reference.fasta")
CACHE_FILE = "analysis_cache.json"  # File where analysis metadata is saved

random.seed(42)


# =====================================================
# REFERENCE LOADING
# =====================================================

def load_reference(path):
    seq = ""
    with open(path, "r") as f:
        for line in f:
            if not line.startswith(">"):
                seq += line.strip()
    return seq


# =====================================================
# KMER INDEX
# =====================================================

def build_kmer_index(reference, k):
    index = defaultdict(list)
    for i in range(len(reference) - k):
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
# SAMPLE ANALYSIS
# =====================================================

def analyze_sample(reads, reference, kmer_index, k):
    aligned_reads = 0
    scores = []
    gc_contents = []
    read_lengths = []

    for read in reads:
        seq = read["seq"]
        read_lengths.append(len(seq))

        # Calculate GC content percentage
        g_count = seq.count('G') + seq.count('g')
        c_count = seq.count('C') + seq.count('c')
        gc_contents.append((g_count + c_count) / len(seq) * 100 if seq else 0)

        rc = reverse_complement(seq)
        _, score_f = align_read_kmer(seq, reference, kmer_index, k)
        _, score_r = align_read_kmer(rc, reference, kmer_index, k)
        score = max(score_f, score_r)

        if score >= ALIGNMENT_THRESHOLD:
            aligned_reads += 1
            scores.append(score)

    return {
        "aligned_reads": aligned_reads,
        "total_reads": len(reads),
        "mt_fraction": aligned_reads / len(reads) if reads else 0,
        "mean_score": float(np.mean(scores)) if scores else 0.0,
        "scores": scores,  # Stored for detailed metrics distribution
        "gc_contents": gc_contents,  # Stored for quality metric graph
        "read_lengths": read_lengths  # Stored for quality metric graph
    }


# =====================================================
# MULTI-GRAPH GENERATION VISUALIZER
# =====================================================

# =====================================================
# INDEPENDENT GRAPH GENERATION VISUALIZER (NEW)
# =====================================================

def plot_all_data(plot_data):
    """Generates four separate independent windows for sequence data metrics."""
    sorted_samples = sorted(plot_data.keys())

    # -------------------------------------------------
    # Window 1: Aligned Reads Bar Graph
    # -------------------------------------------------
    plt.figure(num="1. Aligned Reads Count", figsize=(8, 6))
    aligned_counts = [plot_data[s]['aligned_reads'] for s in sorted_samples]
    bars = plt.bar(sorted_samples, aligned_counts, color='skyblue', edgecolor='black')
    plt.title("Aligned Reads per Sample (Sorted)", fontsize=12, fontweight='bold')
    plt.ylabel("Read Count", fontweight='bold')
    plt.xticks(rotation=30, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    # Add exact numeric values on top of bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + (max(aligned_counts) * 0.01),
                 f'{int(yval):,}', ha='center', va='bottom', fontsize=9)
    plt.tight_layout()

    # -------------------------------------------------
    # Window 2: GC Content Distribution Histogram
    # -------------------------------------------------
    plt.figure(num="2. GC Content Distribution", figsize=(8, 6))
    for sample in sorted_samples:
        plt.hist(plot_data[sample]['gc_contents'], bins=20, alpha=0.5, label=sample, histtype='stepfilled')
    plt.title("GC Content Per-Read Distribution", fontsize=12, fontweight='bold')
    plt.xlabel("GC Content (%)", fontweight='bold')
    plt.ylabel("Count", fontweight='bold')
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # -------------------------------------------------
    # Window 3: Read Length Distribution Histogram
    # -------------------------------------------------
    plt.figure(num="3. Read Length Distribution", figsize=(8, 6))
    for sample in sorted_samples:
        plt.hist(plot_data[sample]['read_lengths'], bins=15, alpha=0.5, label=sample, histtype='step', linewidth=2)
    plt.title("Read Length Distribution", fontsize=12, fontweight='bold')
    plt.xlabel("Length (bp)", fontweight='bold')
    plt.ylabel("Count", fontweight='bold')
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # -------------------------------------------------
    # Window 4: Alignment Score Density Distribution
    # -------------------------------------------------
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

    # Calling plt.show() once at the end opens all 4 active windows at the exact same time
    plt.show()


# =====================================================
# MAIN RUNTIME CONTROL
# =====================================================

def main():
    plot_data = {}
    use_cache = False

    # Check if a cache file already exists to skip calculation steps
    if os.path.exists(CACHE_FILE):
        print("=" * 60)
        print("FOUND PREVIOUSLY SAVED ANALYSIS RUN DATA!")
        print("=" * 60)
        choice = input("Would you like to load cache and display graphs instantly? (y/n): ").strip().lower()
        if choice in ['y', 'yes']:
            use_cache = True

    if use_cache:
        print("\nLoading cached results from disk...")
        with open(CACHE_FILE, "r") as f:
            plot_data = json.load(f)
    else:
        print("\nRunning full sequence alignment processing...")
        print("Loading reference...")
        reference = load_reference(REFERENCE_FILE)

        print("Building k-mer index...")
        kmer_index = build_kmer_index(reference, K)

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
            result = analyze_sample(reads, reference, kmer_index, K)
            elapsed = time.time() - start

            print(f"Aligned reads: {result['aligned_reads']}/{result['total_reads']}")
            print(f"Mean alignment score: {result['mean_score']:.4f}")
            print(f"Runtime: {elapsed:.1f} seconds")

            plot_data[sample] = result

        # Save results to a structured JSON database cache file
        print(f"\nSaving results to database file: '{CACHE_FILE}'...")
        with open(CACHE_FILE, "w") as f:
            json.dump(plot_data, f)

    print("\nGenerating dashboard windows visualization...")
    plot_all_data(plot_data)
    print("\nProcess finalized successfully.")


# =====================================================
# ENTRY POINT
# =====================================================
if __name__ == "__main__":
    main()