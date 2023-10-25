from typing import Tuple, Optional, Sequence
from gensim.models import Phrases
import pyLDAvis.gensim_models
from pyLDAvis._prepare import PreparedData
from wordcloud import WordCloud, STOPWORDS
from nltk import RegexpTokenizer, WordNetLemmatizer
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from datetime import datetime

import seaborn as styler
styler.set_theme(style="whitegrid")

def refine_texts(texts: Sequence[str]) -> Sequence[Sequence[str]]:
    txt_tokenizer = RegexpTokenizer(r'\w+')
    normalizer = WordNetLemmatizer()
    processed_texts = []
    for txt in texts:
        txt = txt.lower()
        txt = txt_tokenizer.tokenize(txt)
        txt = [term for term in txt if not term.isdigit()]
        txt = [term for term in txt if len(term) > 3]
        txt = [normalizer.lemmatize(term) for term in txt]
        txt = [term for term in txt if term not in STOPWORDS]
        processed_texts.append(txt)
    return processed_texts

def enhance_ngrams(texts: Sequence[Sequence[str]]) -> Sequence[Sequence[str]]:
    dual = Phrases(texts, min_count=5)
    triple = Phrases(dual[texts])
    texts_with_ngrams = []
    for txt in texts:
        temp_ngrams = []
        temp_ngrams.extend(txt)
        combined = []
        for term in dual[txt]:
            if '_' in term:
                combined.append(term)
        for term in triple[dual[txt]]:
            if '_' in term:
                combined.append(term)
        temp_ngrams.extend(combined)
        texts_with_ngrams.append(temp_ngrams)
    return texts_with_ngrams

def execute_lda(texts: Sequence[str]) -> PreparedData:
    refined_texts = refine_texts(texts)
    texts_ngrams = enhance_ngrams(refined_texts)

    glossary = Dictionary(texts_ngrams)
    print('Distinct words in initial texts:', len(glossary))

    glossary.filter_extremes(no_below=2, no_above=0.8)
    print('Distinct words post filtering:', len(glossary))
    text_map = [glossary.doc2bow(txt) for txt in texts_ngrams]
    print('Distinct terms: %d' % len(glossary))
    print('Total texts: %d' % len(text_map))

    topics_count = 10
    batch_size = 500
    rounds = 20
    loops = 100
    assess_period = 1

    preview = glossary[0]
    term2id = glossary.id2token

    kickoff = datetime.now()
    lda_instance = LdaModel(
        corpus=text_map,
        id2word=term2id,
        chunksize=batch_size,
        alpha='auto', eta='auto',
        iterations=loops, num_topics=topics_count,
        passes=rounds, eval_every=assess_period
    )
    print(f"Duration for LDA: {datetime.now() - kickoff}")
    pyLDAvis.enable_notebook()

    return pyLDAvis.gensim_models.prepare(lda_instance, text_map, glossary)
