# InverseNet: Physics-Constrained Deep Learning for EEG Source Imaging

This repository contains the code for my final project in **BME 548L: Machine Learning and Imaging** at Duke University (Spring 2026).

## Overview

EEG source imaging is a classical inverse problem: recovering the activity of 7,498 cortical sources from only 60 scalp electrodes. The problem is severely underdetermined, and traditional regularization-based methods (MNE, sLORETA) lose spatial focus at low signal-to-noise ratios (SNR).

We propose **InverseNet**, a deep learning framework that embeds the physics of EEG signal generation directly into the network. The first layer's weights are initialized with the pseudoinverse of the lead field matrix (*L*†), giving the network a physically grounded starting point before any training begins. Three subsequent fully-connected layers then learn nonlinear corrections on top of this physical anchor.

## Key Results

Across SNR levels from 0 to 20 dB:

| SNR (dB) | InverseNet | sLORETA | MNE |
|:---:|:---:|:---:|:---:|
| 15 | **1.00 mm** | 13.86 mm | 35.81 mm |
| 10 | **3.91 mm** | 20.40 mm | 38.29 mm |
| 5 | **13.24 mm** | 28.52 mm | 42.77 mm |
| 0 | **36.22 mm** | 40.18 mm | 48.99 mm |

At SNR = 15 dB, InverseNet localizes 96.8% of samples within the 10 mm clinical threshold, compared to 66.1% for sLORETA and 12.4% for MNE.

## Method

The core idea is to embed the lead field matrix's pseudoinverse *L*† as a learnable initialization for the first network layer, rather than relying on simulated data alone to teach the network the physics. *L*† represents the best linear inverse solution to the EEG inverse problem; using it as initialization gives the network a physical anchor in the otherwise enormous solution space.

The PhysicsLayer remains fully trainable, allowing physical and data-driven knowledge to work together. A cosine similarity analysis confirms that the network preserves the physical prior in well-observed cortical regions while making larger data-driven corrections in deep and medial regions where EEG observability is inherently low — demonstrating that the network learns when to trust the physics.

## Repository Contents

- `EEG_VSCode_clean.ipynb` — Jupyter notebook containing data generation, model definition, training, and evaluation

## Dependencies

- Python 3.10+
- PyTorch
- MNE-Python
- NumPy, SciPy, Matplotlib
- Nilearn (for visualization)

## Dataset

The forward model uses a three-shell boundary element model (BEM) built from the MNE Sample Dataset. Training data consists of 200,000 simulated EEG-source pairs with mixed SNR (5–25 dB).


## Author

**Yuxin Zhou**
Department of Biomedical Engineering, Duke University
yuxin.zhou@duke.edu
