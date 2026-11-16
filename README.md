# L3 Muscle Segmentation and Sarcopenia Indices

Automatic sarcopenia assessment from L3 CT scans. A custom U-Net segments the abdominal muscle wall (class 1) and the psoas muscles (class 2) at the L3 vertebral level, and calculates cross-sectional area in cm² and mean radiodensity in HU — the inputs for the Skeletal Muscle Index (SMI = area in cm² / height in m²).

Model: custom U-Net, 256×256 input, HU window [-30, 150], loss = Sparse Categorical Crossentropy + foreground Dice. No patient scans are in this repo.

## What's in this repo

```
First_Training/        Thesis version — 125 patients
Second_Training/       Retraining — 211 patients (125 original + 86 new)
app/                   Streamlit interface — drag and drop a scan, get the results
U-Net_application.ipynb   Notebook version of the same inference
```

**First_Training/** — the original thesis training. See [First_Training/README.md](First_Training/README.md) for the files, the results, and how to use it.

**Second_Training/** — the same model retrained with more data, plus an external evaluation on a separate batch of 30 patients. See [Second_Training/README.md](Second_Training/README.md) for the files, the results, and how to use it.

**app/** — the easiest way to use the retrained model. The weights file is already inside the folder, next to `app.py`. See below for how to run it.

## Results (internal test set)

| | Abdominals Dice / IoU | Abdominals bAHD / HD95 | Psoas Dice / IoU | Psoas bAHD / HD95 |
|---|---|---|---|---|
| First Training (125 patients) | 0.9124 / 0.8412 | 1.12 mm / 6.40 mm | 0.9157 / 0.8461 | 1.05 mm / 4.99 mm |
| Second Training (211 patients) | 0.9185 / 0.8518 | 1.21 mm / 10.14 mm | 0.9257 / 0.8634 | 0.97 mm / 4.09 mm |

## How to use it

**Easiest — the app:** download the `app/` folder. The retrained weights (`unet_sarcopenia_best.keras`) are already inside it, next to `app.py`. Then run `streamlit run app.py` (or double-click `run_app.bat` on Windows). Drop a `.nii` or `.nii.gz` L3 slice in, and download the predicted mask and the text report.

**Notebook:** you only need to download the weights file and the U-Net application file — `Second_Training/unet_sarcopenia_best.keras` and `U-Net_application.ipynb` — put them in the same folder and run the notebook. For the thesis model instead, use `First_Training/unet_sarcopenia_best.h5`.

**Requirements:** Python 3.8+, TensorFlow, nibabel, numpy, scikit-learn, scipy, matplotlib, seaborn, pandas, streamlit (only for the app).

**Privacy:** no patient scans or names are in this repo. Keep your `Dataset/` folder and `mapping_pazienti.csv` local only — never add them to GitHub.
