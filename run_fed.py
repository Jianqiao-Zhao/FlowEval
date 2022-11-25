from score.utility import extract_intent_features, extract_utterance_features, load_intent, load_utterance, calculate_correlation
from score.score import *

import os
import json
import torch
import numpy as np


if __name__ == "__main__":

    train_path = "training_set"
    test_path = "test_sets/FED"
    model_path = "IntentBERT/all/models_all"
    segment = [0, 41, 81, 125]

    pooler_type = 'max_pooling'
    # `recompute` flag can be turned off to save time if you are sure that the results to be loaded are correct;
    # This flag is turned on to ensure correctness at any moment
    recompute = True

    # load data & extract features
    train_intent = load_intent(train_path)
    train_utterance = load_utterance(train_path)
    train_intent_features = extract_intent_features(train_path, model_path, pooler_type=pooler_type, overwrite=recompute)
    train_utterance_features = extract_utterance_features(train_path, overwrite=recompute)

    test_intent = load_intent(test_path)
    test_utterance = load_utterance(test_path)
    test_intent_features = extract_intent_features(test_path, model_path, pooler_type=pooler_type, overwrite=recompute)
    test_utterance_features = extract_utterance_features(test_path, overwrite=recompute)

    human_bot_reference_score = {
        "overall": [
            v for v in json.load(open(os.path.join(test_path, "dialog_id_to_score.json"))).values()
        ],
    }

    # compute scores
    human_bot_candidate_score = {}

    path = os.path.join(test_path, f"{consensus_intent_score_v1.__name__}_score.npy")
    if os.path.exists(path) and not recompute:
        predicted_v1_scores = np.load(path)
    else:
        predicted_v1_scores = consensus_intent_score_v1(10, 0, train_intent_features, train_intent, test_intent_features, test_intent)
        np.save(path, predicted_v1_scores)
    human_bot_candidate_score[f"{consensus_intent_score_v1.__name__}"] = predicted_v1_scores

    path = os.path.join(test_path, f"{consensus_intent_score_v2.__name__}_score.npy")
    if os.path.exists(path) and not recompute:
        predicted_v2_scores = np.load(path)
    else:
        predicted_v2_scores = consensus_intent_score_v2(40, 1, train_intent_features, train_intent, train_utterance, test_intent_features, test_intent, test_utterance)
        np.save(path, predicted_v2_scores)
    human_bot_candidate_score[f"{consensus_intent_score_v2.__name__}"] = predicted_v2_scores

    path = os.path.join(test_path, f"{consensus_pretrained_model_score.__name__}_score.npy")
    if os.path.exists(path) and not recompute:
        predicted_utterance_scores = np.load(path)
    else:
        predicted_utterance_scores = consensus_pretrained_model_score(10, test_utterance, test_utterance_features, train_utterance, train_utterance_features)
        np.save(path, predicted_utterance_scores)
    human_bot_candidate_score[f"{consensus_pretrained_model_score.__name__}"] = predicted_utterance_scores

    path = os.path.join(test_path, f"combined_score.npy")
    if os.path.exists(path) and not recompute:
        predicted_combined_scores = np.load(path)
    else:
        predicted_combined_scores = (0.1 * torch.tensor(predicted_v2_scores) + 0.9 * torch.tensor(predicted_utterance_scores)).tolist()
        np.save(path, predicted_combined_scores)
    human_bot_candidate_score["combined_score"] = predicted_combined_scores

    calculate_correlation(human_bot_reference_score, human_bot_candidate_score, segment)

