"""

Creates a txt file from a Wikipedia Dump

Inspired from 
https://github.com/panyang/Wikipedia_Word2vec/blob/master/v1/process_wiki.py

"""

import sys
from gensim.corpora import WikiCorpus


def corpus_make(f_in, txt_out):
    """
    """
    output = open(txt_out, 'w')
    corpus = WikiCorpus(f_in)

    i = 0
    for text in corpus.get_texts():
        output.write(bytes(' '.join(text), 'utf-8').decode('utf-8') + '\n')
        i = i + 1
        if (i % 1000 == 0):
            print("Processed " + str(i) + "articles.")

    output.close()
    print("Processing Done.")


if __name__ == '__main__':

    if len(sys.argv) != 3:
        print("python wikiCorpus.py wiki...xml.bz2 wiki_tr.txt")
        sys.exit(1)
    f_in, txt_out = sys.argv[1:3]
    corpus_make(f_in, txt_out)
