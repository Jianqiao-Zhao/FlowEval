import numpy as np

from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction


def consensus_intent_score_v1(K, L, train_intent_features, train_intent, test_intent_features, test_intent):
    import faiss

    # Build a single index of all candidate vectors, populated on the-fly below.
    candidate_index = faiss.IndexFlatIP(train_intent_features.shape[-1])
    # Add training set features
    train_intent_features = np.ascontiguousarray(train_intent_features[:, L, :])
    faiss.normalize_L2(train_intent_features)
    candidate_index.add(train_intent_features)
    # Search test set features
    queries = np.ascontiguousarray(test_intent_features[:, L, :])
    faiss.normalize_L2(queries)  # actually we don't need this, as the returned scores are not important
    scores, predictions = candidate_index.search(queries, K)

    predicted_scores = []
    for idx, neighbor_ids in enumerate(predictions):
        hyp = test_intent[idx]
        refs = []
        predicted_score = []
        for neighbor in neighbor_ids:
            ref = train_intent[neighbor]
            refs.append(ref)
            predicted_score.append(
                sentence_bleu([ref], hyp, smoothing_function=SmoothingFunction().method4))  # percentage
        predicted_scores.append(max(predicted_score))
    return predicted_scores


def consensus_intent_score_v2(K, L, train_features, train_intent, train_utterance, test_features, test_intent, test_utterance):
    # scale intent score by tf-idf score (topic match), and add utterance BLEU score
    from nltk.translate.bleu_score import sentence_bleu
    from nltk.translate.bleu_score import SmoothingFunction
    from sklearn.feature_extraction.text import TfidfTransformer
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from nltk import word_tokenize

    count_vect = CountVectorizer()
    tfidf_transformer = TfidfTransformer()
    train_tfidf = tfidf_transformer.fit_transform(count_vect.fit_transform(train_utterance))
    test_tfidf = tfidf_transformer.transform(count_vect.transform(test_utterance))
    tfidf_scores = cosine_similarity(test_tfidf, train_tfidf)  # cosine similarity

    train_features = train_features.astype(np.double)
    test_features = test_features.astype(np.double)

    train_features = np.ascontiguousarray(train_features[:, L, :])
    train_features = train_features / np.linalg.norm(train_features, axis=-1, keepdims=True)
    queries = np.ascontiguousarray(test_features[:, L, :])
    queries = queries / np.linalg.norm(queries, axis=-1, keepdims=True)
    scores = np.matmul(queries, train_features.T)  # cosine similarity

    scores = (tfidf_scores + 1.) * (scores + 1.)  # rescale scores (work best)
    predictions = np.argsort(-scores, axis=-1)[:, :K]
    scores = np.take_along_axis(scores, predictions, axis=-1)

    predicted_scores = []
    for idx, neighbor_ids in enumerate(predictions):
        hyp = test_intent[idx]
        refs = []
        predicted_score = []
        for neighbor in neighbor_ids:
            ref = train_intent[neighbor]
            refs.append(ref)
            predicted_score.append(
                sentence_bleu([ref], hyp, smoothing_function=SmoothingFunction().method4) + sentence_bleu(
                    [word_tokenize(train_utterance[neighbor])], word_tokenize(test_utterance[idx]),
                    smoothing_function=SmoothingFunction().method4))  # percentage
        predicted_scores.append(max(predicted_score))
    return predicted_scores


def consensus_pretrained_model_score(k, test_utterance, test_utterance_features, train_utterance, train_utterance_features):
    # a good but slow score using utterance only
    from bert_score import BERTScorer

    # should be obtained by calling utility.extract_utterance_features
    train_utterance_features = train_utterance_features.astype(np.double)
    test_utterance_features = test_utterance_features.astype(np.double)

    train_utterance_features = train_utterance_features / np.linalg.norm(train_utterance_features, axis=-1, keepdims=True)
    queries = test_utterance_features / np.linalg.norm(test_utterance_features, axis=-1, keepdims=True)
    scores = np.matmul(queries, train_utterance_features.T)  # cosine similarity
    predictions = np.argsort(-scores, axis=-1)[:, :k]
    scores = np.take_along_axis(scores, predictions, axis=-1)

    scorer = BERTScorer(idf=True, idf_sents=train_utterance, lang='en', use_fast_tokenizer=True)
    predicted_scores = []
    for idx, neighbor_ids in enumerate(predictions):
        hyp = test_utterance[idx]
        refs = []
        for neighbor in neighbor_ids:
            ref = train_utterance[neighbor]
            refs.append(ref)
        predicted_score = scorer.score(
            [hyp] * len(refs),
            refs,
        )[2]  # P, R, F
        predicted_scores.append(predicted_score.max().cpu().detach().tolist())
    return predicted_scores
