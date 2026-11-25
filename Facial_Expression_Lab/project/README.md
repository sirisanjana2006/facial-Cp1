# Facial Expression Lab

An 8-class facial expression classifier trained on **CK+ Extended**, with
a baseline CNN, a CNN + channel-attention model, and a small website that
lets you upload a photo and see a live prediction, plus a results
dashboard built from your own training run.

```
project/
├── data_pipeline/
│   └── preprocessing.py            # CSV -> images -> preprocessed_data.npz
├── models/
│   ├── train_CNN.py                # baseline CNN
│   ├── CNN_Attention_improved.py   # CNN + channel attention (main model)
│   ├── comparison.py                # baseline vs attention, printed report
│   └── generate_dashboard_data.py   # exports results.json for the website
├── predict_emotion.py               # command-line single-image prediction
├── app/
│   ├── app.py                       # Flask server
│   ├── templates/index.html
│   └── static/
│       ├── css/style.css
│       ├── js/main.js
│       └── data/results.json        # created by generate_dashboard_data.py
└── requirements.txt
```

## 1. Setup

```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

All scripts currently point at:

```
C:\Users\91994\PycharmProjects\PythonProject7\CK+
```

via a `BASE_PATH` variable at the top of each file. Update that one line
in each script if your project lives elsewhere — everything else (data
file names, model file names) is built from it, so you only edit it in
one place per file.

You'll need `ckextended.csv` (the CK+ Extended pixel CSV) inside
`BASE_PATH` before step 2.

## 2. Run the pipeline, in order

```bash
python data_pipeline/preprocessing.py          # -> preprocessed_data.npz
python models/train_CNN.py                     # -> baseline_cnn.h5
python models/CNN_Attention_improved.py         # -> cnn_attention.h5 / cnn_attention_best.h5
python models/comparison.py                     # prints accuracy + confusion matrices
python models/generate_dashboard_data.py         # -> app/static/data/results.json
```

Each step reads the previous step's output from `BASE_PATH`, so run them
in this order the first time. After that, you can re-run
`CNN_Attention_improved.py` alone to retrain, then re-run `comparison.py`
and `generate_dashboard_data.py` to refresh the website's numbers.

## 3. Run the website

```bash
cd app
python app.py
```

Open **http://127.0.0.1:5000**. You can:

- Drop or choose a face photo and get a live prediction from your
  trained `cnn_attention` model, with a confidence bar for all 8
  expressions.
- Scroll to **Results** to see baseline-vs-attention accuracy,
  per-expression accuracy, and how often the two models agree — all
  pulled from `results.json`, i.e. your actual test-set numbers.
- Scroll to **Dataset** to see the class distribution in your training
  set (useful for spotting how imbalanced classes like contempt are).

If you open the site before running `generate_dashboard_data.py`, the
Results and Dataset sections show a note telling you which script to run
— they don't show fake numbers.

## 4. Command-line prediction (no website)

```bash
python predict_emotion.py
```

Edit `IMAGE_PATH` at the top of the file to point at the image you want
to classify.

## Notes on accuracy

CK+ Extended is a small dataset (under 1,000 images across 8 imbalanced
classes). `CNN_Attention_improved.py` already applies data augmentation,
class weighting, batch normalization, and L2/dropout regularization to
get the most out of that data — but the practical ceiling is set by
dataset size. If you want to push further, look at transfer learning
from a small pretrained backbone (e.g. MobileNetV2) or k-fold
cross-validation instead of a single train/val/test split.
