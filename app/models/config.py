import os

UPLOADS_FOLDER = 'assets/uploads'
# os.makedirs(UPLOADS_FOLDER, exist_ok=True)
PKL_FOLDER = 'assets/pickles'
os.makedirs(PKL_FOLDER, exist_ok=True)
DB_FOLDER = 'assets/db'

PLOT_FOLDER = 'assets/docs/plot'
os.makedirs(PLOT_FOLDER, exist_ok=True)

LABEL_COL = 'Class/ASD Traits'
CATEGORICAL_COLS = ['Sex', 'Ethnicity', 'Jaundice', 'Family_mem_with_ASD']