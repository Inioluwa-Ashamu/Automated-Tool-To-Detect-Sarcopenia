Automated Tool to Detect Sarcopenia from CT Scans

This repository contains the full implementation of my MSc Artificial Intelligence dissertation project at Manchester Metropolitan University: “Automated Tool to Detect Sarcopenia from CT Scans.”

The project develops a deep learning and rule-based pipeline for automated segmentation and quantification of skeletal muscle at the L3 vertebra level, a standard landmark for sarcopenia diagnosis. It integrates manual ground truth annotations, deep learning (U-Net), and TotalSegmentator outputs, and provides a clinician-facing Streamlit application for opportunistic screening.

🔑 Key Features

Data Digitisation

Convert anonymised DICOM scans into NIfTI and extract metadata.

Write and validate L3 vertebra indices for reproducible processing.

Automated Segmentation

Deep Learning (U-Net): trained to predict L3 slice and muscle mask.

TotalSegmentator: baseline multi-organ segmentation framework.

Rule-based application of tissue masks at L3.

Evaluation

Compute CSA (Cross-Sectional Area) and SMRA (Skeletal Muscle Radiation Attenuation).

Dice similarity between manual annotations, U-Net predictions, and TS outputs.

Bland–Altman plots, correlation scatterplots, histograms, and boxplots.

Clinician-facing Deployment

Streamlit app to load scans, overlay segmentations, compute metrics, visualise results, and export reports.

Reproducible Outputs

All figures in dissertation generated automatically by make_all_plots.py.

Summary statistics and metrics written to CSV and text files for auditability.

📂 Repository Structure
.
├── digitize_dicom.py              # Convert and anonymise raw DICOMs, extract metadata
├── write_l3_index_from_mask.py    # Extract and store L3 vertebra index
├── run_all_segmentation.py        # Orchestrates full segmentation pipeline
├── train_l3_unet.py               # Train U-Net model for L3 slice/muscle detection
├── predict_l3_unet.py             # Predict L3 segmentations with trained model
├── apply_tissue_masks_at_L3.py    # Apply TotalSegmentator tissue masks at L3 level
├── export_l3_pairs.py             # Export paired CT/mask slices for DL training
├── compute_all_csa_smra.py        # Compute CSA & SMRA metrics
├── eval_dice_manual.py            # Evaluate Dice between manual & DL segmentations
├── eval_dice_manual_vs_TS.py      # Evaluate Dice between manual & TotalSegmentator
├── summarise_and_plot.py          # Generate basic plots and summary statistics
├── make_all_plots.py              # Full research-grade figure generation
├── L3 Sarcopenia Streamlit App-proV5.py  # Streamlit app for clinician-facing deployment
├── requirements.txt               # Python dependencies
└── README.md                      # Project documentation (this file)
⚙️ Installation
1. Clone the Repository
git clone https://github.com/<your-username>/sarcopenia-detection.git
cd sarcopenia-detection

2. Install Dependencies
pip install -r requirements.txt

3. Environment Notes

Python 3.9+ recommended.

For GPU training/inference, install PyTorch with the correct CUDA version from pytorch.org
.

TotalSegmentator requires nnU-Net v2; it is installed automatically with pip install TotalSegmentator.

🚀 Usage
1. Data Preparation

Convert and digitise raw DICOMs:

python digitize_dicom.py
python write_l3_index_from_mask.py

2. Segmentation

Run the full segmentation pipeline:

python run_all_segmentation.py


Train a U-Net model:

python train_l3_unet.py


Predict with a trained model:

python predict_l3_unet.py

3. Metric Computation

Compute CSA & SMRA metrics:

python compute_all_csa_smra.py


Evaluate Dice overlap:

python eval_dice_manual.py
python eval_dice_manual_vs_TS.py

4. Analysis & Plotting

Generate summary plots:

python summarise_and_plot.py
python make_all_plots.py

5. Deployment App

Launch Streamlit tool:

streamlit run "L3 Sarcopenia Streamlit App-proV5.py"

📊 Outputs

comparison_all.csv → Combined manual, DL, and TS metrics per patient.

eval_dice_manual_vs_DL.csv / eval_dice_manual_vs_TS.csv → Dice similarity scores.

dl_runs/figs/ → Automatically generated plots for dissertation.

Streamlit app → Interactive overlays, metric tables, cohort summaries, and report export.

🧪 Reproducibility

All scripts are modular, documented, and produce deterministic outputs when run with the same data.

Every figure in the dissertation (histograms, scatter plots, Bland–Altman, boxplots) can be regenerated with make_all_plots.py.

The pipeline supports opportunistic screening — analysis can be applied to routine CT without requiring full clinical metadata (height/sex).

📋 Requirements

See requirements.txt
:

numpy>=1.22
pandas>=1.5
matplotlib>=3.6
SimpleITK>=2.2
nibabel>=5.1
pydicom>=2.4
torch>=2.0
streamlit>=1.26
reportlab>=4.0
TotalSegmentator

📖 Citation

If you use this repository in academic work, please cite:

Ashamu, I. (2025). Automated Tool to Detect Sarcopenia from CT Scans. MSc Artificial Intelligence Dissertation, Manchester Metropolitan University.

⚠️ Disclaimer

This software is provided for research purposes only.
It is not a medical device and must not be used for clinical decision-making.