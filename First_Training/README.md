# First Training — thesis version

This folder is the original thesis training. Keep it together and don't overwrite it — the retraining is in `Second_Training/`.

## What each file is for, in order

1. `Patient_engeneering_sanity_check.ipynb` — prepare and check the data first. It decompresses the `.nii.gz` scans, anonymises patients to `subject_X`, and checks every mask only contains labels 0 (background), 1 (abdominals), 2 (psoas). It expects one CT slice and one mask per patient, as `.nii` files in local `slices/` and `masks/` folders.
2. `model_training_and_evaluation.ipynb` — train and evaluate the model. Dataset: 125 patients, split 87 train / 19 validation / 19 test. Custom U-Net, 256x256 input, HU window [-30, 150], 50 epochs. Test results: abdominals Dice 0.9124, IoU 0.8412, bAHD 1.12 mm, HD95 6.40 mm; psoas Dice 0.9157, IoU 0.8461, bAHD 1.05 mm, HD95 4.99 mm.
3. `unet_sarcopenia_best.h5` — the trained weights from this training. You don't need to retrain to use them.

## How to use the trained model

You only need to download the weights file and the U-Net application file: `First_Training/unet_sarcopenia_best.h5` and `U-Net_application.ipynb` from the repository root. Put them in the same folder, open the notebook, and run it — it takes one unseen L3 CT scan (`.nii`) and gives you back a predicted mask (`.nii`) and a text report with total muscle area in cm², psoas area in cm², and mean density in HU for each. SMI is then area in cm² divided by height in metres squared.

## Requirements

Python 3.8+, TensorFlow, nibabel, numpy, scikit-learn, scipy, matplotlib, seaborn, pandas. Run the notebooks in Jupyter, in the order above.

## Privacy

No patient scans or names are in this folder, or in the repo. Keep your `Dataset/` folder and the `mapping_pazienti.csv` file local only — never add them to GitHub.
