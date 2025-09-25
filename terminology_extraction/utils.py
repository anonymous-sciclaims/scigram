import os
import re
from typing import List, Dict, Any, Tuple

import numpy as np
import srsly
from flair.data import Sentence
from flair.models import SequenceTagger
from nltk import WordNetLemmatizer, sent_tokenize
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, BatchEncoding

NGRAM_TYPES = {1: "unigrams", 2: "bigrams", 3: "trigrams"}


def load_tqa_with_splits(root_path: str, json_files: List[str]) -> List[Dict]:

    # Aggregate the three files (one for each split) from tqa in just one,
    # adding a new field with the split info

    final_dataset = []
    for json_file in json_files:
        json_path = os.path.join(root_path, json_file)
        dataset = srsly.read_json(json_path)
        for lesson in tqdm(dataset):
            lesson["split"] = json_file.split("/")[0]
        final_dataset = final_dataset + dataset
    return final_dataset


def generate_terminology_wise_topics(
    dataset: Dict[str, Any],
    chunker: SequenceTagger,
    tagger: SequenceTagger,
    model: AutoModel,
    tokenizer: AutoTokenizer,
    lemmatizer: WordNetLemmatizer,
    bnc_freqs: Dict[str, Any],
    tqa_freqs: Dict[str, Any],
    weirdness_index_threshold: float,
    stopwords: List[str],
) -> List[Dict]:

    # Return a list of topics with their terminology
    #
    # The terminology for a topic only has terms that are also present
    # in the terminology from the question/answers of the lesson.
    #
    # The terms are filtered using their embeddings and their relative
    # position to the centroid of the topic.
    #
    # Terms are selected if they are closer to the centroid than the distance
    # threshold.
    #
    # After the term selection, a new centroid and threshold are calculated, to
    # represent the semantic field of the topic.
    #
    # Embeddings are removed to reduce the size of the file when saved.

    out_topics = []
    pbar = tqdm(dataset)
    for lesson in pbar:
        topics = lesson["topics"]
        lesson_id = lesson["globalID"]
        lessonname = lesson["lessonName"]
        split = lesson["split"]
        question_types = ["nonDiagramQuestions", "diagramQuestions"]
        qa_text = get_qa_text(lesson=lesson, question_types=question_types)
        qa_terminology = get_terminology(
            text=qa_text,
            chunker=chunker,
            tagger=tagger,
            model=model,
            tokenizer=tokenizer,
            lemmatizer=lemmatizer,
            bnc_freqs=bnc_freqs,
            tqa_freqs=tqa_freqs,
            weirdness_index_threshold=weirdness_index_threshold,
            stopwords=stopwords,
        )
        for topic_id in topics.keys():
            topic = topics[topic_id]
            topicname = topic["topicName"]
            content = topic["content"]["text"]
            text = topicname + ". " + content
            terminology = get_terminology(
                text=text,
                chunker=chunker,
                tagger=tagger,
                model=model,
                tokenizer=tokenizer,
                lemmatizer=lemmatizer,
                bnc_freqs=bnc_freqs,
                tqa_freqs=tqa_freqs,
                weirdness_index_threshold=weirdness_index_threshold,
                stopwords=stopwords,
            )
            terminology = {
                lemma: lemma_info
                for lemma, lemma_info in terminology.items()
                if lemma in qa_terminology
            }
            filtered_terminology = filter_terminology(terminology=terminology)
            centroid, threshold, distances = get_embedding_info(
                terminology=filtered_terminology
            )
            terminology_without_embeddings = remove_embeddings(
                terminology=filtered_terminology
            )
            out_topic = {
                "lessonId": lesson_id,
                "lessonName": lessonname,
                "topicId": topic_id,
                "topicName": topicname,
                "text": text,
                "terms": terminology_without_embeddings,
                "centroid": centroid.astype(float).tolist(),
                "threshold": threshold.astype(float),
                "split": split,
            }
            out_topics.append(out_topic)
    return out_topics


def get_qa_text(lesson: Dict[str, Any], question_types: List[str]) -> str:

    # Return a string block with all the questions and answers from a
    # lesson, separated by a dot.

    qa_text = ""
    for question_type in question_types:
        if lesson["questions"][question_type]:
            for question_id in lesson["questions"][question_type]:
                qa_dict = lesson["questions"][question_type][question_id]
                question = qa_dict["beingAsked"]["processedText"]
                qa_text = add_str_with_dot(text=qa_text, new_text=question)
                for choice in qa_dict["answerChoices"]:
                    answer = qa_dict["answerChoices"][choice]["processedText"]
                    qa_text = add_str_with_dot(text=qa_text, new_text=answer)
    return qa_text


def get_terminology(
    text: str,
    chunker: SequenceTagger,
    tagger: SequenceTagger,
    model: AutoModel,
    tokenizer: AutoTokenizer,
    lemmatizer: WordNetLemmatizer,
    bnc_freqs: Dict[str, Any],
    tqa_freqs: Dict[str, Any],
    weirdness_index_threshold: float,
    stopwords: List[str],
) -> Dict[str, Any]:

    # Return the terminology without filters (dictionary of lemmas with
    # their sentences, the forms as they appear in the sentence and their
    # embedding) of a given text.
    #
    # 1. Extract noun phrases which have a better weirdness index than the
    # threshold.
    # 2. Lemmatize the noun phrases to avoid repeated terms.
    # 3. Extract the embedding of each appearance of the lemma in the text.
    # 4. Generate terminology without filters

    sentences = sent_tokenize(text)
    terminology = {}
    for sentence in sentences:
        noun_phrases = get_noun_phrases(
            sentence=sentence,
            chunker=chunker,
            tagger=tagger,
            stopwords=stopwords,
            lemmatizer=lemmatizer,
        )
        for noun_phrase in noun_phrases:
            weirdness_index = get_weirdness_index(
                term=noun_phrase.lower(), bnc_freqs=bnc_freqs, tqa_freqs=tqa_freqs
            )
            if weirdness_index >= weirdness_index_threshold:
                lemma = " ".join(
                    [lemmatizer.lemmatize(x) for x in noun_phrase.split()]
                ).lower()
                embedding = get_embedding(
                    term=noun_phrase,
                    sentence=sentence,
                    model=model,
                    tokenizer=tokenizer,
                )
                if lemma in terminology:
                    terminology[lemma]["sentences"].append(sentence)
                    terminology[lemma]["term_forms"].append(noun_phrase)
                    terminology[lemma]["embeddings"].append(embedding)
                else:
                    terminology[lemma] = {
                        "sentences": [sentence],
                        "term_forms": [noun_phrase],
                        "embeddings": [embedding],
                    }
    return terminology


def get_noun_phrases(
    sentence: str,
    chunker: SequenceTagger,
    tagger: SequenceTagger,
    stopwords: List[str],
    lemmatizer: WordNetLemmatizer,
) -> List[str]:

    # Return the list of noun phrases of a given sentence.
    #
    # 1. Select chunks that are noun phrases
    # 2. Tag each word of the chunk
    # 3. Shorten the candidates to delete stopwords, digits or determinants
    # from them
    # 4. Discard those with more than three words or that don't appear in the
    # sentence (they are malformed).

    valid_tags = [
        "NN",
        "NNS",
        "NNP",
        "NNPS",
        "JJ",
        "JJR",
        "JJS",
        "VB",
        "VBD",
        "VBG",
        "VBN",
    ]
    chunking = Sentence(sentence)
    chunker.predict(chunking)
    noun_phrases = []
    candidates = [
        chunk.text
        for chunk in chunking.get_spans("np")
        if chunk.labels[0].value == "NP"
    ]
    for candidate in candidates:
        tagging = Sentence(candidate)
        tagger.predict(tagging)
        words = tagging.get_spans("pos")
        shortened_candidate = shorten_chunk(
            words=words,
            lemmatizer=lemmatizer,
            stopwords=stopwords,
            valid_tags=valid_tags,
        )
        if shortened_candidate and len(shortened_candidate) <= 3:
            noun_phrase = " ".join(shortened_candidate)
            if noun_phrase in sentence:
                noun_phrases.append(noun_phrase)
    return noun_phrases


def shorten_chunk(
    words: List[Dict],
    lemmatizer: WordNetLemmatizer,
    stopwords: List[str],
    valid_tags: List[str],
) -> List[Dict]:

    # The shortened chunk starts with the first valid term with a valid tag
    # (see valid tag list above), and it ends when an invalid term appears or
    # the original chunk ends

    start_index = None
    end_index = None
    for index, word in enumerate(words):
        if valid_term(term=lemmatizer.lemmatize(word.text), stopwords=stopwords):
            if start_index is None:
                if word.labels[0].value in valid_tags:
                    end_index = len(words)
                    start_index = index
            else:
                if end_index == len(words):
                    end_index = index + 1
    shortened_chunk = [
        word.text
        for word in words[start_index:end_index]
        if start_index is not None and end_index is not None
    ]
    return shortened_chunk


def valid_term(term: str, stopwords: List[str]) -> bool:
    return not term.isdigit() and term not in stopwords


def get_weirdness_index(
    term: str, bnc_freqs: Dict[str, Any], tqa_freqs: Dict[str, Any]
) -> float:

    # Return the weirdness index of a term given its frequency in a general
    # corpus (BNC) and a target corpus (TQA).
    #
    # Term frequencies are separated by unigrams, bigrams and trigrams.
    #
    # If the term is not in the general corpus, weirdness index is infinite
    # (set as 30000 to ease its use)
    #
    # If the term is not in the target corpus, weirdness index is 0.
    #
    # If the term is in both corpora, weirdness index is calculated with the
    # formula:
    # (frequency in tqa / sum of frequencies of tqa) /
    # (frequency in bnc / sum of frequencies of bnc)

    gram_types = ["unigrams", "bigrams", "trigrams"]
    weirdness_index = 0
    gram_type = gram_types[len(term.split()) - 1]
    if term in tqa_freqs[gram_type]:
        tqa_freq = tqa_freqs[gram_type][term]
        tqa_total = sum(tqa_freqs[gram_type].values())
        if term in bnc_freqs[gram_type]:
            bnc_freq = bnc_freqs[gram_type][term]
            bnc_total = sum(bnc_freqs[gram_type].values())
            weirdness_index = (tqa_freq / tqa_total) / (bnc_freq / bnc_total)
        else:
            weirdness_index = 30000
    return weirdness_index


def filter_terminology(terminology: Dict[str, Any]) -> Dict[str, Any]:

    # Calculate centroid, threshold and the distances of each lemma to the
    # centroid and returns the terminology with those lemmas which are below
    # the threshold

    centroid, threshold, distances = get_embedding_info(terminology=terminology)
    filtered_terminology = {
        lemma: terminology[lemma]
        for lemma, dis in zip(terminology.keys(), distances)
        if dis <= threshold
    }
    return filtered_terminology


def get_embedding_info(
    terminology: Dict[str, Any]
) -> Tuple[np.array, float, List[float]]:

    # Calculate the centroid, threshold and distances from a given terminology.
    #
    # Embeddings of a single lemma within a topic are combined using the mean.
    #
    # The centroid is the mean of all the embeddings from the topic
    #
    # Distances are the euclidean distance of each embedding of the topic to
    # the centroid
    #
    # Threshold is the average distance of all the embeddings plus standard
    # deviation

    embeddings = np.array(
        [np.mean(x["embeddings"], axis=0) for x in terminology.values()]
    )
    centroid = np.mean(embeddings, axis=0)
    distances = [get_euclidean_distance(emb, centroid) for emb in embeddings]
    threshold = np.mean(distances) + np.std(distances)
    return centroid, threshold, distances


def remove_embeddings(terminology: Dict[str, Any]) -> Dict[str, Any]:
    final_dict = {
        lemma: {
            "sentences": lemma_info["sentences"],
            "term_forms": lemma_info["term_forms"],
        }
        for lemma, lemma_info in terminology.items()
    }
    return final_dict


def get_embedding(
    term: str, sentence: str, model: AutoModel, tokenizer: AutoTokenizer
) -> np.array:
    inputs = tokenizer(sentence, return_tensors="pt")
    term_positions = get_term_positions(inputs=inputs, term=term, sentence=sentence)
    outputs = model(**inputs)
    last_hidden_states = outputs.last_hidden_state
    arrays = [last_hidden_states[0][pos].detach().numpy() for pos in term_positions]
    final_array = np.mean(arrays, axis=0)
    return final_array


def add_str_with_dot(text: str, new_text: str) -> str:
    text = text + new_text
    if not text.lstrip().endswith("."):
        text = text + ". "
    else:
        text = text + " "
    return text


def get_euclidean_distance(a: np.array, b: np.array) -> np.float64:
    return np.linalg.norm(a - b)


def get_term_positions(inputs: BatchEncoding, term: str, sentence: str) -> List[int]:
    span = None
    for match in re.finditer(term, sentence):
        span = match.span()
    span_start = inputs.char_to_token(0, span[0])
    span_end = inputs.char_to_token(0, span[1] - 1) + 1
    return [x for x in range(span_start, span_end)]


def get_clean_term(ngram: List[str]) -> str:
    term = " ".join([word.strip() for word in ngram])
    term = re.sub(r"[^\w\s-]", "", term)
    return term


def clean_ngram(ngram: List[str], lemmatizer: WordNetLemmatizer) -> Tuple[str, str]:
    term = get_clean_term(ngram=ngram)
    lemma = " ".join([lemmatizer.lemmatize(word) for word in term.split()]).lower()
    return term, lemma