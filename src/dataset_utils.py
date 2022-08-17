from nltk import tokenize
import numpy as np
from sentence_transformers import util
import re


def tokenize_dataset(row, tokenizer):
    claim = row['claim']
    main_text = row['main_text']
    tokenized_input = tokenizer(claim, main_text, return_tensors='np', padding='max_length', truncation='only_second')
    row['input_ids'] = tokenized_input['input_ids']
    row['token_type_ids'] = tokenized_input['token_type_ids']
    row['attention_mask'] = tokenized_input['attention_mask']
    return row

def get_websites_list(row):
    sources = row['sources'].split(', ')
    sources_list = []
    for source in sources:
        cur_source = re.sub(r"^(https://|http://)", "", source)
        cur_source = re.sub(r"^www.", "", cur_source)
        cur_source = cur_source.split('/')[0]
        if cur_source.isspace():
            cur_source = "NO_SOURCES"
        sources_list.append(cur_source)
    row['source_sites'] = sources_list
    return row

def get_sources_vector(row, useful_sources):
    vector = np.zeros(len(useful_sources))
    for source in row['source_sites']:
        if source in useful_sources:
            vector[useful_sources[source]] += 1
    row['sources_vector'] = vector
    return row


def get_top_k_sentences(row, sentence_model, k):
    claim_emb = sentence_model.encode(row['claim'])  # Claim is usually a single sentence
    sentences = tokenize.sent_tokenize(row['main_text'])

    sentences_emb = sentence_model.encode(sentences)

    similarities = util.dot_score(claim_emb, sentences_emb)
    sorted_sims_with_indices = sorted(enumerate(similarities[0]), key=lambda x: x[1], reverse=True)

    top_k_sentences = []
    for idx, sim in sorted_sims_with_indices[:k]:
        top_k_sentences.append(sentences[idx])
    row['main_text'] = ' '.join(top_k_sentences)
    return row

def get_subjects_vector(row, useful_subjects):
    vector = np.zeros(len(useful_subjects))
    for subj in row['subjects'].split(','):
        if subj.strip() in useful_subjects:
            vector[useful_subjects[subj.strip()]] += 1
    row['subjects_vector'] = vector
    return row