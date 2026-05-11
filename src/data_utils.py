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

            # Normalize characters and dashes
            text = re.sub(r'[“”„«»]', '"', text)
            text = re.sub(r'[ -]+\'', ' "', text)
            text = re.sub(r'\' ', '" ', text)
            text = re.sub(r'[‘’]', '\'', text)
            text = re.sub(r'--', ' ', text)
            text = re.sub(r'[—–]', '-', text)
            text = re.sub(r'- -', '-', text)
            text = re.sub(r'_', '', text)

            #references and [Illustration]
            text = re.sub(r'[\]+[0-9]+[\]]+', '', text)
            text = re.sub(r'\[.*?\]', '', text)
            text = re.sub(r'C{2}HCl{3}O. H{2}O!', '', text)
            text = re.sub(r'\{.*?\}', '', text) # erase {.jpg}

            # {sic} mispeallign original so maintain original
           

            # history, grammar, &c., went on for
                # Christian{sic} had s
            #odern Morpheus--C{2}HCl{3}O. H{2}O!
            txt_all += text + " "

            
    print(f"Total books: {len(files)}")
    print(f"Total characters: {len(txt_all)}")
    return txt_all

def inspect_pattern(text, pattern, name, max_examples=10):
    print(f"\nPattern: {name}")
    matches = list(re.finditer(pattern, text))
    
    print(f"Total matches: {len(matches)}")
    
    for i, m in enumerate(matches[:max_examples]):
        start, end = m.span()
        context = text[max(0, start-20):min(len(text), end+20)]
        
        print(f"Match {i+1}: '{m.group()}'")
        print(f"Context: ...{context}...")
        print("-"*40)


if __name__ == "__main__":
    corpus = load_and_clean_txt('data/archive')
    #print(corpus[0:2000])
    #inspect_pattern(corpus, r'[^a-zA-Z0-9\s\.\,\!\?\;\:\'\-\"\(\)]', "weird characters")
    with open('data/corpus_clean.txt', 'w', encoding='utf-8') as f:
        f.write(corpus)
    print('done')

    