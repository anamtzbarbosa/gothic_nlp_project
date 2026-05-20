import os
import re

def load_and_clean_txt(folder_data):
    if isinstance(folder_data, str):
        folder_data = [folder_data]

    txt_all = ""
    total_files = 0

    for folder in folder_data:
        files = [f for f in os.listdir(folder) if f.endswith('.txt')]
        total_files += len(files)
        for file in files:
            ruta = os.path.join(folder, file)
            with open(ruta, 'r', encoding='utf-8-sig') as f:
                text = f.read()

            text = re.sub(r'(?i)chapter [0-9ivxlc]+[\.\-]*[^\n]*', '', text)
            # Carmilla-style: standalone Roman numeral on its own line (e.g. I., XIV.)
            text = re.sub(r'(?m)^\s*[IVXLC]+\.\s*$', '', text)
            text = re.sub(r'([* ])+([*])', '', text)
            text = re.sub(r'\s+', ' ', text)

            # Normalize fancy quotes and dashes to ASCII equivalents
            text = re.sub(u'[\u201c\u201d\u201e\u00ab\u00bb]', '"', text)
            text = re.sub(u'[ -]+\u2018', ' "', text)
            text = re.sub(u'\u2019 ', '" ', text)
            text = re.sub(u'[\u2018\u2019]', "'", text)
            text = re.sub(r'--', ' ', text)
            text = re.sub(u'[\u2014\u2013]', '-', text)
            text = re.sub(r'- -', '-', text)
            text = re.sub(r'_', '', text)

            # References and [Illustration]
            text = re.sub(r'[\]+[0-9]+[\]]+', '', text)
            text = re.sub(r'\[.*?\]', '', text)
            text = re.sub(r'C{2}HCl{3}O. H{2}O!', '', text)
            text = re.sub(r'\{.*?\}', '', text)

            txt_all += text + " "

    print(f"Total books: {total_files}")
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
    folders = ['data/archive', 'data/new books', 'data/new_books_2']
    corpus = load_and_clean_txt(folders)
    with open('data/corpus_clean.txt', 'w', encoding='utf-8') as f:
        f.write(corpus)
    print('done')
