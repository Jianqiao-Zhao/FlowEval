import numpy as np
from scipy.stats import pearsonr, spearmanr, kendalltau
import os
from statistics import mean
import pickle
import json
import random
from scipy.special import softmax


def standardize(vector):
    vector = np.array(vector)
    vector_standardized = (vector - vector.mean()) / vector.std()
    return vector_standardized


def range_normalize(vector):
    vector = np.array(vector)
    vector_normalized = (vector - vector.min()) / (vector.max() - vector.min())
    return vector_normalized


def softmax_normalize(vector):
    return softmax(vector)


def get_fed_overall(fed_score_raw):
    return np.array([np.mean([score["coherent"], score["error recovery"], score["consistent"], score["diverse"],
                              score["depth"], score["likeable"], score["understand"], score["flexible"],
                              score["informative"], score["inquisitive"]]) for score in fed_score_raw])


def combine_two_score(score1, score2, normalization=standardize, weights=(1, 1)):
    assert len(score1) == len(score2)
    if normalization is None:
        score = weights[0] * score1 + weights[1] * score2
        return score
    score1 = normalization(score1)
    score2 = normalization(score2)
    score = weights[0] * score1 + weights[1] * score2
    return score


def combine_three_score(score1, score2, score3, normalization=standardize, weights=(1, 1, 1)):
    assert len(score1) == len(score2) and len(score1) == len(score3)
    if normalization is None:
        score = weights[0] * score1 + weights[1] * score2 + weights[2] * score3
        return score
    score1 = normalization(score1)
    score2 = normalization(score2)
    score3 = normalization(score3)
    score = weights[0] * score1 + weights[1] * score2 + weights[2] * score3
    return score



def calculate_correlation(folder, segment=[0, 91, 187, 278]):
    # For Controllable Dialogue, the segment should be [0, 91, 187, 278] (Three systems with 91, 96, and 91 dialogues.)
    # For FED, the segment should be [0, 41, 81, 125] (Three systems with 41, 40, and 44 dialogues.)
    # For DSTC9, the segment should be [[200 * i for i in range(12)]] (Every 200 dialogues come from the same systems.)
    splits = generate_split_for_controllable()

    assert os.path.isfile(f"{folder}/reference_score.npy")
    reference_score = np.load(f"{folder}/reference_score.npy", allow_pickle=True)
    reference_score = [float(score) for score in reference_score]

    fed_score_raw = np.load(f"{folder}/fed_score.npy", allow_pickle=True)
    fed_score = get_fed_overall(fed_score_raw)

    flow_score = np.load(f"{folder}/flow_score.npy")

    with open(f"{folder}/eval_empathetic.score" if "DSTC9" not in folder else f"{folder}/eval_dailydialog.score", "rb") as f:
        dyna_score = pickle.load(f)

    if "human" in folder:
        # bert_score = np.load(f"{folder}/bert_score.npy")
        bert_score = get_test_score(np.load(f"{folder}/bert_score.npy"), splits)
        bleu_sm3_score = get_test_score(np.load(f"{folder}/bleu_sm3_score.npy"), splits)

    v1_score = np.load(f"{folder}/consensus_intent_score_v1_score.npy")
    v2_score = np.load(f"{folder}/consensus_intent_score_v2_score.npy")
    v3_score = np.load(f"{folder}/consensus_intent_score_v3_score.npy")

    utt_bert = np.load(f"{folder}/consensus_pretrained_model_score_score.npy")
    # utt_flow = np.load(f"{folder}/consensus_pretrained_model_score_flow_score.npy")

    if "human" in folder:
        reference_score = get_test_score(reference_score, splits)
        fed_score = get_test_score(fed_score, splits)
        # flow_score = get_test_score(flow_score, splits)
        dyna_score = get_test_score(dyna_score, splits)
        v1_score = get_test_score(v1_score, splits)
        v2_score = get_test_score(v2_score, splits)
        v3_score = get_test_score(v3_score, splits)
        utt_bert = get_test_score(utt_bert, splits)

    # test_score = combine_two_score(v2_score, utt_bert, None, [0.1, 0.9])
    test_score = combine_two_score(utt_bert, dyna_score, None, [1, 1])
    # test_score = combine_three_score(v2_score, utt_bert, flow_score, range_normalize, [0.05, 0.45, -0.5])

    calculate_score(test_score, reference_score, segment=segment)


def test_v2_chatbot_correlation(folder):
    assert os.path.isfile(f"{folder}/reference_score.npy") and \
           os.path.isfile(f"{folder}/consensus_intent_score_v2_score.npy")
    reference_score = np.load(f"{folder}/reference_score.npy", allow_pickle=True)
    reference_score = [float(score) for score in reference_score]
    v2_score = np.load(f"{folder}/consensus_intent_score_v2_score.npy")

    segment = [200 * i for i in range(12)]

    test_score = v2_score

    chatbot_reference_score = []
    chatbot_pearson = []
    chatbot_spearman = []
    chatbot_kendall = []
    for i in range(len(segment) - 1):
        chatbot_reference_score.append(mean(reference_score[segment[i]: segment[i + 1]]))
        chatbot_pearson.append(
            pearsonr(reference_score[segment[i]: segment[i + 1]], test_score[segment[i]: segment[i + 1]]))
        chatbot_spearman.append(
            spearmanr(reference_score[segment[i]: segment[i + 1]], test_score[segment[i]: segment[i + 1]]))
        chatbot_kendall.append(
            kendalltau(reference_score[segment[i]: segment[i + 1]], test_score[segment[i]: segment[i + 1]]))

    print(chatbot_reference_score)
    print(chatbot_pearson)
    print(chatbot_spearman)
    print(chatbot_kendall)


def calculate_score(score1, score2, segment=[0, 91, 187, 278]):
    assert len(score1) == len(score2)

    pearson_result = pearsonr(score1, score2)
    spearman_result = spearmanr(score1, score2)
    kendall_result = kendalltau(score1, score2)
    print("******************Dialogue-Level************************************")
    print(f"Pearson correlation is {pearson_result[0]}, p-value is {pearson_result[1]}")
    print(f"Spearman correlation is {spearman_result[0]}, p-value is {spearman_result[1]}")
    print(f"Kendall correlation is {kendall_result[0]}, p-value is {kendall_result[1]}")

    # For System Comparison ONLY
    # chatbot_score1 = []
    # chatbot_score2 = []
    # for i in range(len(segment) - 1):
    #     chatbot_score1.append(mean(score1[segment[i]: segment[i + 1]]))
    #     chatbot_score2.append(mean(score2[segment[i]: segment[i + 1]]))
    # pearson_result = pearsonr(chatbot_score2, chatbot_score1)
    # spearman_result = spearmanr(chatbot_score2, chatbot_score1)
    # kendall_result = kendalltau(chatbot_score2, chatbot_score1)
    # print("******************Chatbot-Level************************************")
    # print(f"Pearson correlation is {pearson_result[0]}, p-value is {pearson_result[1]}")
    # print(f"Spearman correlation is {spearman_result[0]}, p-value is {spearman_result[1]}")
    # print(f"Kendall correlation is {kendall_result[0]}, p-value is {kendall_result[1]}")


def calculate_inter_correlation(folder, segment=[0, 91, 187, 278]):
    assert os.path.isfile(f"{folder}/reference_score.npy")
    reference_score = np.load(f"{folder}/reference_score.npy", allow_pickle=True)
    reference_score = [float(score) for score in reference_score]

    if "human" in folder:
        bert_score = np.load(f"{folder}/bert_score.npy")
        bleu_sm3_score = np.load(f"{folder}/bleu_sm3_score.npy")

    fed_score_raw = np.load(f"{folder}/fed_score.npy", allow_pickle=True)
    fed_score = get_fed_overall(fed_score_raw)

    flow_score = -np.load(f"{folder}/flow_score.npy")

    with open(f"{folder}/eval_empathetic.score", "rb") as f:
        dyna_score = pickle.load(f)

    v1_score = np.load(f"{folder}/consensus_intent_score_v1_score.npy")
    v2_score = np.load(f"{folder}/consensus_intent_score_v2_score.npy")

    utt_bert = np.load(f"{folder}/consensus_pretrained_model_score_score.npy")

    calculate_score(bert_score, bleu_sm3_score, segment=segment)


def generate_split_for_controllable():
    if os.path.isfile("controllable_dialogue_split.json"):
        with open("controllable_dialogue_split.json") as f:
            return json.load(f)

    valid = []
    all = list(range(278))
    valid.extend(random.sample(list(range(91)), 9))
    valid.extend(random.sample(list(range(91, 187)), 10))
    valid.extend(random.sample(list(range(187, 278)), 9))
    test = [i for i in all if i not in valid]
    with open("controllable_dialogue_split.json", "w") as f:
        json.dump({"valid": valid, "test": test}, f, indent=4)
    return {"valid": valid, "test": test}


def get_test_score(score, splits):
    test = splits["test"]
    return np.array([score[i] for i in test])


if __name__ == "__main__":

    # calculate_correlation("DSTC9/", segment=[200 * i for i in range(12)])
    calculate_correlation("FED/", segment=[0, 41, 81, 125])
    # calculate_correlation("human-to-bot/")
    # test_v2_chatbot_correlation("DSTC9/")
    # calculate_inter_correlation("human-to-bot/")
    pass
