from datasets import load_dataset, load_from_disk
from transformers import BertTokenizer
from functools import reduce
from collections import Counter
from sentence_transformers import SentenceTransformer
import os

from dataset_utils import tokenize_dataset, get_websites_list, get_sources_vector, get_top_k_sentences, get_subjects_vector

def prepare_dataset(config):
    if os.path.exists(config.save_path + "_train"):
        print('Dataset already exists, loading from disk')
        train_dataset = load_from_disk(config.save_path + "_train")
        val_dataset = load_from_disk(config.save_path + "_val")
        test_dataset = load_from_disk(config.save_path + "_test")
        return train_dataset, val_dataset, test_dataset

    train_dataset = load_dataset(config.dataset_name, split="train")
    val_dataset = load_dataset(config.dataset_name, split="validation")
    test_dataset = load_dataset(config.dataset_name, split="test")

    train_dataset = train_dataset.filter(lambda x: x['label'] != -1)
    val_dataset = val_dataset.filter(lambda x: x['label'] != -1)
    test_dataset = test_dataset.filter(lambda x: x['label'] != -1)

    if config.add_metadata_to_input:
        # get most common sources, encode sources for each row of dataset and add it as a new column

        train_dataset = train_dataset.map(get_websites_list, batched=False)
        val_dataset = val_dataset.map(get_websites_list, batched=False)
        test_dataset = test_dataset.map(get_websites_list, batched=False)

        all_sources = reduce(lambda x, y: x + y, train_dataset['source_sites'])
        counter = Counter(all_sources)
        useful_sources = {elem[0]: i for i, elem in
                          enumerate(counter.most_common(config.metadata_params.n_most_common_sources))}

        train_dataset = train_dataset.map(get_sources_vector, batched=False, fn_kwargs={'useful_sources': useful_sources})
        val_dataset = val_dataset.map(get_sources_vector, batched=False, fn_kwargs={'useful_sources': useful_sources})
        test_dataset = test_dataset.map(get_sources_vector, batched=False, fn_kwargs={'useful_sources': useful_sources})

        train_dataset = train_dataset.remove_columns(['source_sites'])
        val_dataset = val_dataset.remove_columns(['source_sites'])
        test_dataset = test_dataset.remove_columns(['source_sites'])


        all_subjects = reduce(lambda x, y: x + y.split(','), train_dataset['subjects'], [])
        # Sometimes the subjects have space after comma, sometimes thy don't
        all_subjects = list(map(lambda x: x.strip(), all_subjects))
        subj_counter = Counter(all_subjects)
        useful_subjects = {elem[0]: i for i, elem in
                           enumerate(subj_counter.most_common(config.metadata_params.n_most_common_subjects))}

        train_dataset = train_dataset.map(get_subjects_vector, batched=False, fn_kwargs={'useful_subjects': useful_subjects})
        val_dataset = val_dataset.map(get_subjects_vector, batched=False, fn_kwargs={'useful_subjects': useful_subjects})
        test_dataset = test_dataset.map(get_subjects_vector, batched=False, fn_kwargs={'useful_subjects': useful_subjects})


    if config.use_sentence_transformer:
        # Select top k sentences from main_text with largest similarity to claim

        sentence_model = SentenceTransformer(config.sentence_transformer_params.sentence_transformer_name)
        top_k_sentences = config.sentence_transformer_params.top_k_sentences

        train_dataset = train_dataset.map(get_top_k_sentences, batched=False,
                                          fn_kwargs={'sentence_model': sentence_model, 'k':top_k_sentences})
        val_dataset = val_dataset.map(get_top_k_sentences, batched=False,
                                      fn_kwargs={'sentence_model': sentence_model, 'k':top_k_sentences})
        test_dataset = test_dataset.map(get_top_k_sentences, batched=False,
                                        fn_kwargs={'sentence_model': sentence_model, 'k':top_k_sentences})


    tokenizer = BertTokenizer.from_pretrained(config.tokenizer_name)

    train_dataset = train_dataset.map(tokenize_dataset, batched=True, fn_kwargs={'tokenizer': tokenizer})
    val_dataset = val_dataset.map(tokenize_dataset, batched=True, fn_kwargs={'tokenizer': tokenizer})
    test_dataset = test_dataset.map(tokenize_dataset, batched=True, fn_kwargs={'tokenizer': tokenizer})

    train_dataset = train_dataset.remove_columns(
        ['claim_id', 'date_published', 'explanation', 'fact_checkers'])
    val_dataset = val_dataset.remove_columns(
        ['claim_id', 'date_published', 'explanation', 'fact_checkers'])
    test_dataset = test_dataset.remove_columns(
        ['claim_id', 'date_published', 'explanation', 'fact_checkers'])

    train_dataset.save_to_disk(config.save_path + "_train")
    val_dataset.save_to_disk(config.save_path + "_val")
    test_dataset.save_to_disk(config.save_path + "_test")
    print('Dataset processed and saved to ', config.save_path)

    return train_dataset, val_dataset, test_dataset