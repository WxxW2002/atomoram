# AtomORAM: Achieving $O(1)$ Online Server I/O for Low Access Latency under Sparse Workloads

## Introduction
AtomORAM is a novel Oblivious RAM (ORAM) construction that achieves $O(1)$ online server I/O and $O(1)$ online bandwidth within a single round trip. By decoupling the costly access obfuscation process from actual data retrieval via a timer-driven *Virtual Access Insertion* mechanism, AtomORAM amortizes the overhead required to hide access patterns across inter-request intervals.

This repository contains a conceptual prototype for academic research and is designed to produce the theoretical claims and experimental results presented in our paper. It is not optimized or intended for production deployment.

This implementation is open-sourced under the Apache v2 license([License](LICENSE)).

---

## Artifact Evaluation

### Overview

* [Installation](#installation)
* [Figure 1: Sparse Slack CDF](#figure-1-sparse-slack-cdf)
* [Figure 2: Mechanism Validation (Online Server I/O & Latency)](#figure-2-mechanism-validation-online-server-io--latency)
* [Figure 3: Synthetic Sparsity Sweep](#figure-3-synthetic-sparsity-sweep)
* [Figure 4: Real-World Trace Comparison](#figure-4-real-world-trace-comparison)
* [Figure 5: Burst Recovery](#figure-5-burst-recovery)
* [Figure A1 & A2: Stash and Queue Empirical Distributions](#figure-a1--a2-stash-and-queue-empirical-distributions)

Requirements
* **Environment**: Ubuntu 22.04 LTS (Recommended)
* **Software**: Python 3.10+, with dependencies listed in `requirements.txt`.
* **Hardware**: We tested on a 16-core 32GB ubuntu machine in Tencent Cloud. Recommended at least 16GB RAM and 50GB storage.

The tree structure of the project is as follows:
```text
.
├── artifacts/          # Generated outputs (PDF figures and CSV data)
├── data/               # Raw and processed trace datasets (MSRC, AliCloud)
├── scripts/            # Experiment execution scripts (E1-E5, A1-A2)
├── src/                # Core ORAM protocols and simulation engine
│   ├── backend/        # B-Tree storage abstractions
│   ├── protocols/      # Implementations of AtomORAM, PathORAM, RingORAM
│   ├── sim/            # Event-driven and timer-driven runner logic
│   └── traces/         # Trace loading and schema definitions
└── tests/              # Unit tests
└── .gitignore
└── LICENSE
└── README.md
└── requirements.txt
```

The evaluation compares AtomORAM with the following baselines:Non-recursive Path ORAM、Ring ORAM, and non-ORAM lower bound storage, all have a single round trip of communication. The links to the papers for the three ORAMs implemented are as follows:
* **Path ORAM**: https://eprint.iacr.org/2013/280.pdf
* **Ring ORAM**: https://eprint.iacr.org/archive/2014/997/1418874204.pdf

---

### Installation

1. Clone the repository and navigate to the project root:
    ```bash
    git clone https://github.com/WxxW2002/atomoram.git
    cd atomoram
    ```

2. Set up the Python environment and install dependencies. We use Python 3.10.:
    ```bash
    pip install -r requirements.txt
    ```

3. Our raw trace datasets are in data/raw/, including MSRC src1_0_tripped.csv subset and AliCloud io_traces_32.csv subset. To simplify our evaluation, we provide subsets of these datasets. The full datasets can be obtained from the following links:
    - MSRC: https://iotta.snia.org/traces/block-io/388
    - AliCloud: https://github.com/alibaba/block-traces

4. Run tests to verify the correctness:
    ```bash
    pytest -q
    ```

---

### Experiment
The following commands will produce the results of the experiment. All generated figures (PDF) will be saved in artifacts/outputs/ and the corresponding raw data (CSV) in artifacts/csv/.

---

#### Figure 1: Sparse Slack CDF
```bash
python3 scripts/exp_e1_slack_cdf.py
```

This command verifies the feasibility of the Sparse Access Assumption on real-world cloud traces by showing the Cumulative Distribution Function (CDF) of the idle gaps between real requests.

The generated figure is saved in artifacts/figs/Fig1_Sparse_Slack_CDF.pdf, and the raw data is saved in artifacts/csv/E1_MSRC_CDF.csv, artifacts/csv/E1_Alic_CDF.csv.

Sample figure 1 output:
![Figure 1: Sparse Slack CDF](artifacts/figs/Fig1_Sparse_Slack_CDF.pdf)

---

#### Figure 2: Mechanism Validation (Online Server I/O & Latency)
```bash
python3 scripts/exp_e2_mechanism.py
```

This command directly validates the core contribution: AtomORAM maintains $O(1)$ online server I/O and critical-path latency as the tree capacity ($N$) scales, in contrast to the $O(\log N)$ PathORAM and RingORAM.

The generated figure is saved in artifacts/figs/Fig2_Mechanism_Validation.pdf, and the raw data is saved in artifacts/csv/E2_Online_IO.csv, artifacts/csv/E2_Online_Latency.csv.

Sample figure 2 output:
![Figure 2: Mechanism Validation](artifacts/figs/Fig2_Mechanism_Validation.pdf)

---

#### Figure 3: Synthetic Sparsity Sweep
```bash
python3 scripts/exp_e3_sparsity_sweep.py
```

This command runs a boundary sweep experiment demonstrating system behavior across different sparsity ratios ($\alpha$). Shows near-zero queuing delay when $\alpha \ge 1$ (assumption holds) and expected graceful degradation when $\alpha < 1$ (assumption violated).

The generated figure is saved in artifacts/figs/Fig3_Sparsity_Sweep.pdf, and the raw data is saved in artifacts/csv/E3_Sparsity_Sweep.csv.

Sample figure 3 output:
![Figure 3: Sparsity Sweep](artifacts/figs/Fig3_Sparsity_Sweep.pdf)

---

#### Figure 4: Real-World Trace Comparison
```bash
python3 scripts/exp_e4_real_trace.py
```

This command performs an end-to-end P95 latency evaluation comparing AtomORAM against Path ORAM and Ring ORAM on MSRC (sparse) and AliCloud (dense) workloads.

The generated figure is saved in artifacts/figs/Fig4_Real_Trace_Comparison.pdf, and the raw data is saved in artifacts/csv/E4_Real_Trace_Comparison.csv.

Sample figure 4 output:
![Figure 4: Real-World Trace Comparison](artifacts/figs/Fig4_Real_Trace_Comparison.pdf)

---

#### Figure 5: Burst Recovery
```bash
python3 scripts/exp_e5_burst_recovery.py
```

This command demonstrates the resilience and self-healing capability of AtomORAM, showing how accumulated queue lengths during traffic bursts are digested during subsequent idle periods.

The generated figure is saved in artifacts/figs/Fig5_Burst_Recovery.pdf, and the raw data is saved in artifacts/csv/E5_Burst_Recovery.csv.

Sample figure 5 output:
![Figure 5: Burst Recovery](artifacts/figs/Fig5_Burst_Recovery.pdf)

---

#### Figure A1 & A2: Stash and Queue Empirical Distributions
```bash
python3 scripts/exp_a1_a2_distributions.py
```

This command runs appendix experiments demonstrating the physical bounds of the system in steady-state. Fig A1 proves that client stash size is tightly bounded ($O(1)$ physical memory), refuting trivial caching concerns. Fig A2 shows the long-tail queue distribution under dense workloads.

The generated figures are saved in artifacts/figs/FigA1_Stash_Distribution.pdf and artifacts/figs/FigA2_Queue_Distribution.pdf, and the raw data is saved in artifacts/csv/A1_A2_*_Distribution.csv.

Sample figure A1 output:
![Figure A1: Stash Distribution](artifacts/figs/FigA1_Stash_Distribution.pdf)

Sample figure A2 output:
![Figure A2: Queue Distribution](artifacts/figs/FigA2_Queue_Distribution.pdf)
















