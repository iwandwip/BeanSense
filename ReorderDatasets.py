import pandas as pd
import os


def reorder_datasets():
    label_order = [
        'aKaw-D', 'aKaw-L', 'aKaw-M',
        'aSem-D', 'aSem-L', 'aSem-M',
        'rGed-D', 'rGed-L', 'rGed-M',
        'rTir-D', 'rTir-L', 'rTir-M'
    ]

    column_orders = {
        'dataset4.csv': ['NAMA', 'MQ135', 'MQ2', 'MQ3', 'MQ6'],
        'dataset6.csv': ['NAMA', 'MQ135', 'MQ2', 'MQ3', 'MQ6', 'MQ138', 'MQ7'],
        'dataset8.csv': ['NAMA', 'MQ135', 'MQ2', 'MQ3', 'MQ6', 'MQ138', 'MQ7', 'MQ136', 'MQ5']
    }

    for filename, columns in column_orders.items():
        filepath = f"datasets/{filename}"

        if not os.path.exists(filepath):
            print(f"File not found: {filepath}")
            continue

        df = pd.read_csv(filepath)

        existing_columns = df.columns.tolist()
        valid_columns = [col for col in columns if col in existing_columns]

        if len(valid_columns) != len(columns):
            missing_columns = [col for col in columns if col not in existing_columns]
            print(f"Warning: Missing columns in {filename}: {missing_columns}")

        df = df[valid_columns]

        cat_type = pd.CategoricalDtype(categories=label_order, ordered=True)
        df['NAMA'] = df['NAMA'].astype(cat_type)

        df = df.sort_values('NAMA')

        df.to_csv(filepath, index=False)

        print(f"Processed {filename}")


if __name__ == "__main__":
    reorder_datasets()
