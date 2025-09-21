import pandas as pd
import re

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)  
    text = re.sub(r'\@\w+|\#', '', text)               
    text = re.sub(r'[^A-Za-z0-9\s]', '', text)        
    text = text.strip()
    return text

def main():
    
    real_data = pd.read_csv('../data/real.csv')
    fake_data = pd.read_csv('../data/fake.csv')

    
    for df, name in [(real_data, 'real.csv'), (fake_data, 'fake.csv')]:
        if 'text' not in df.columns:
            raise Exception(f"'text' column missing in {name}")

    
    real_data['label'] = 0  # Real
    fake_data['label'] = 1  # Fake

    
    data = pd.concat([real_data, fake_data], ignore_index=True)

    
    data = data.dropna(subset=['text'])

    
    data = data.drop_duplicates()

    
    data['text'] = data['text'].apply(clean_text)

    
    data = data[data['text'].str.strip() != '']

    
    data.to_csv('../data/cleaned_dataset.csv', index=False)

    print(f"Combining and cleaning complete. {len(data)} rows saved to data/cleaned_dataset.csv")

if __name__ == '__main__':
    main()
