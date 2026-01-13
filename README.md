## Dataset Handling

The original IMDb dataset is larger than GitHub's file size limit.
Therefore, the dataset was split into three smaller CSV files:

- IMDB_part1.csv
- IMDB_part2.csv
- IMDB_part3.csv

The main scripts automatically load and merge these files during execution.

To train and evaluate classical Machine Learning models (Naive Bayes, Logistic Regression, SVM), run:
python classic_ml.py

To train and evaluate the LSTM-based Deep Learning model, run:
python lstm_dl.py


