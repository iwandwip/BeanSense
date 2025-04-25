import pandas as pd

data = pd.read_csv('dataset8.csv')

nama_labels = data['NAMA'].unique().tolist()

sorted_nama_labels = sorted(nama_labels)

for index, label in enumerate(sorted_nama_labels, 1):
    print(f"{index}. {label}")
