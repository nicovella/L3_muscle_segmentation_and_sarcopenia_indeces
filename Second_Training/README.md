# Second Training — retraining with more data

This folder is the retraining of the same model, with more data. It does not replace the thesis version — that one is in `First_Training/`.

## What each file is for, in order

1. `model_retraining_.ipynb` — retrain and evaluate the model. Dataset: 211 patients (125 original + 86 new), split 147 train / 32 validation / 32 test. Same U-Net as the first training, 256x256 input, HU window [-30, 150], 50 epochs. Internal test results: abdominals Dice 0.9185, IoU 0.8518, bAHD 1.21 mm, HD95 10.14 mm; psoas Dice 0.9257, IoU 0.8634, bAHD 0.97 mm, HD95 4.09 mm. Best validation loss 0.13520, final validation Dice 0.9088. This notebook also includes descriptive statistics tables and best-case and worst-case audits.
2. `04_second_external_evaluation_30patients.ipynb` — evaluation on a separate batch of 30 patients, not used in training.
3. `unet_sarcopenia_best.keras` — the retrained weights from this training. You don't need to retrain to use them. This file is in `.keras` format, not `.h5` like the first training.

## How to use the trained model

You only need to download the weights file and the U-Net application file: `Second_Training/unet_sarcopenia_best.keras` and `U-Net_application.ipynb` from the repository root. Put them in the same folder, open the notebook, and run it — it takes one unseen L3 CT scan (`.nii`) and gives you back a predicted mask (`.nii`) and a text report with total muscle area in cm², psoas area in cm², and mean density in HU for each. SMI is then area in cm² divided by height in metres squared.

## Requirements

Python 3.8+, TensorFlow, nibabel, numpy, scikit-learn, scipy, matplotlib, seaborn, pandas. The first cell of the retraining notebook installs them with `%pip install`. Run the notebooks in Jupyter, in the order above.

## Privacy

No patient scans or names are in this folder, or in the repo. Keep your `Dataset/` folder and the `mapping_pazienti.csv` file local only — never add them to GitHub.
