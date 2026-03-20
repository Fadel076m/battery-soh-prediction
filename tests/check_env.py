import sys
import numpy as np
import pandas as pd
import matplotlib
import seaborn as sns
import sklearn
import tensorflow as tf
import plotly
import streamlit
import tqdm
import joblib

print(f"Python version: {sys.version}")
print(f"TensorFlow version: {tf.__version__}")
gpu_devices = tf.config.list_physical_devices('GPU')
print(f"GPU Available: {len(gpu_devices) > 0}")
if gpu_devices:
    print(f"GPU devices: {gpu_devices}")

libs = {
    "numpy": np.__version__,
    "pandas": pd.__version__,
    "matplotlib": matplotlib.__version__,
    "seaborn": sns.__version__,
    "scikit-learn": sklearn.__version__,
    "plotly": plotly.__version__,
    "streamlit": streamlit.__version__,
    "tqdm": tqdm.__version__,
    "joblib": joblib.__version__
}

for lib, version in libs.items():
    print(f"{lib}: {version}")
