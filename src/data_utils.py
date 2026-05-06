import os
import re

def load_and_clean_txt(folder_data):
    txt_all = ""

    files = [f for f in os.listdir(folder_data) if f.endswith('.txt')]
    for file in files:
        ruta = os.path.join(folder_data, file)
        with open(ruta, 'r', encoding='utf-8') as f:
            text = f.read()
            
            text = re.sub(r'(?i)chapter [0-9ivxlc]+[\.]*', '', text)
            text = re.sub(r'([* ])+([*])', '', text)
            text = re.sub(r'\s+', ' ', text)
            txt_all += text + " "
            
    print(f"Total books: {len(files)}")
    print(f"Total characters: {len(txt_all)}")
    return txt_all


if __name__ == "__main__":
    corpus = load_and_clean_txt('data/archive')
    print(corpus[0:100])
    with open('data/corpus_clean.txt', 'w', encoding='utf-8') as f:
        f.write(corpus)