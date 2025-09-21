import pandas as pd
import glob

csv_files = glob.glob(r'..\data\*.csv')

df_list = []
for file in csv_files:
    try:
        df = pd.read_csv(file, encoding='utf-8')
    except UnicodeDecodeError:
        print(f"[Warning] UTF-8 decoding failed for {file}. Trying ISO-8859-1 encoding...")
        df = pd.read_csv(file, encoding='ISO-8859-1')
    df_list.append(df)

combined_df = pd.concat(df_list, ignore_index=True)

cleaned_df = combined_df.drop_duplicates(subset='text').reset_index(drop=True)

cleaned_df.to_csv('merged_cleaned_dataset.csv', index=False)

print(f'Merged {len(csv_files)} files successfully. Final dataset shape: {cleaned_df.shape}')
