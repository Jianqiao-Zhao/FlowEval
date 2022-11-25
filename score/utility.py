import os
import json
import torch
import transformers
from tqdm import tqdm
import numpy as np

from torch.utils.data import DataLoader

from model import MyDataset


def _extract_intent_features(data_loader, model, pooler_type):
    def _move(data):
        # move to GPU
        for k in data.keys():
            data[k] = data[k].to(model.device)

    all_pooled_outputs = []
    for batch in tqdm(data_loader):
        with torch.no_grad():
            _move(batch)
            outputs = model(**batch, output_hidden_states=True)
            sequence_outputs = torch.stack(outputs.hidden_states, dim=2).cpu()  # batch * seq_len * layer * hidden_size
            if pooler_type == 'cls':
                pooled_outputs = sequence_outputs[:, 0, :, :].clone()
            elif pooler_type == 'mean_pooling':
                # remove padding (and <s>, </s>, <cls>?)
                lengths = batch['attention_mask'].sum(dim=1, keepdim=True).unsqueeze(
                    -1).cpu()  # 1 for unmask, 0 for mask
                pooled_outputs = torch.sum(sequence_outputs * batch['attention_mask'].unsqueeze(-1).unsqueeze(-1).cpu(),
                                           dim=1) / lengths
            elif pooler_type == 'max_pooling':
                # remove padding (and <s>, </s>, <cls>?)
                mask = 1 - batch['attention_mask'].unsqueeze(-1).unsqueeze(-1).cpu()
                pooled_outputs = \
                    sequence_outputs.masked_fill(mask.bool(), value=torch.tensor(float('-inf'))).max(dim=1)[0]
            else:
                raise ValueError(f"Unknown pooler_type {pooler_type}")
            all_pooled_outputs.append(pooled_outputs)
            torch.cuda.empty_cache()
    return torch.cat(all_pooled_outputs, dim=0)


def extract_intent_features(data_path, model_path, data_mode="all", vocab_path='vocab.json', max_length=512,
                            batch_size=64, pooler_type='max_pooling', mask_prob=0., overwrite=False):
    if os.path.isfile(f"{data_path}/{pooler_type}_feature.npy") and not overwrite:
        return np.load(f"{data_path}/{pooler_type}_feature.npy")
    model = transformers.RobertaForMaskedLM.from_pretrained(model_path, local_files_only=True)
    if torch.cuda.is_available():
        model.cuda()
    dataset = MyDataset(data_path, mode=data_mode, vocab_path=vocab_path, max_length=max_length, mask_prob=mask_prob)
    loader = DataLoader(dataset, batch_size=batch_size)
    features = _extract_intent_features(loader, model, pooler_type=pooler_type)
    features = features.cpu().detach().numpy()
    np.save(f"{data_path}/{pooler_type}_feature.npy", features)
    return features


def load_intent(data_path, mode="all"):
    raw_data = json.load(open(os.path.join(data_path, 'dialog_id_to_intent.json')))
    if mode != "all":
        split_allocation = json.load(open(os.path.join(data_path, "split_allocation.json")))
        related_dialog = split_allocation[mode]
    else:
        related_dialog = list(raw_data.keys())
    return [raw_data[dialog_id] for dialog_id in related_dialog]


def load_utterance(data_path, mode="all", delimiter="\n"):
    raw_data = json.load(open(os.path.join(data_path, 'dialog_id_to_utterance.json')))
    if mode != "all":
        split_allocation = json.load(open(os.path.join(data_path, "split_allocation.json")))
        related_dialog = split_allocation[mode]
    else:
        related_dialog = list(raw_data.keys())
    prefix = delimiter.strip() + " " if delimiter.strip() != "" else ""  # for DialoGPT (" <|endoftext|> ")
    return [prefix + delimiter.join(raw_data[dialog_id]) for dialog_id in related_dialog]


def extract_utterance_features(data_path, mode="all", model_name="roberta-large", delimiter="\n", overwrite=False):
    if os.path.isfile(f"{data_path}/{model_name}_feature.npy") and not overwrite:
        return np.load(f"{data_path}/{model_name}_feature.npy")
    from transformers import AutoTokenizer, AutoModel

    dialog = load_utterance(data_path, mode, delimiter)

    def _move(data):
        # move to GPU
        for k in data.keys():
            data[k] = data[k].to(model.device)

    with torch.no_grad():
        model = AutoModel.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        if torch.cuda.is_available():
            model.cuda()

        features = []
        for utterance in tqdm(dialog):
            batch = tokenizer([utterance], truncation=True, return_tensors="pt")
            _move(batch)
            outputs = model(**batch, output_hidden_states=True)
            try:
                pooler_output = outputs.pooler_output
            except AttributeError:
                # the last representation by default
                pooler_output = outputs.last_hidden_state[:, -1, :]
            features.append(pooler_output.cpu().detach())
        features = torch.cat(features, dim=0).numpy()

    os.makedirs(os.path.dirname(f"{data_path}/{model_name}_feature.npy"), exist_ok=True)
    np.save(f"{data_path}/{model_name}_feature.npy", features)
    return features


def calculate_correlation(candidate_scores: dict, reference_scores: dict, segment: list) -> None:
    from scipy.stats import pearsonr, spearmanr, kendalltau
    from statistics import mean

    for reference_aspect in reference_scores.keys():
        for candidate_aspect in candidate_scores.keys():
            print(f"{reference_aspect} VS {candidate_aspect}:")
            reference_score = reference_scores[reference_aspect]
            predicted_score = candidate_scores[candidate_aspect]
            pearson_result = pearsonr(reference_score, predicted_score)
            spearman_result = spearmanr(reference_score, predicted_score)
            kendall_result = kendalltau(reference_score, predicted_score)
            print("******************Dialogue-Level************************************")
            print(f"Pearson correlation is {pearson_result[0]}, p-value is {pearson_result[1]}")
            print(f"Spearman correlation is {spearman_result[0]}, p-value is {spearman_result[1]}")
            print(f"Kendall correlation is {kendall_result[0]}, p-value is {kendall_result[1]}")

            chatbot_predicted_scores = []
            chatbot_reference_scores = []
            for i in range(len(segment) - 1):
                chatbot_predicted_scores.append(mean(predicted_score[segment[i]: segment[i + 1]]))
                chatbot_reference_scores.append(mean(reference_score[segment[i]: segment[i + 1]]))
            pearson_result = pearsonr(chatbot_reference_scores, chatbot_predicted_scores)
            spearman_result = spearmanr(chatbot_reference_scores, chatbot_predicted_scores)
            kendall_result = kendalltau(chatbot_reference_scores, chatbot_predicted_scores)
            print("******************Chatbot-Level************************************")
            print(f"Pearson correlation is {pearson_result[0]}, p-value is {pearson_result[1]}")
            print(f"Spearman correlation is {spearman_result[0]}, p-value is {spearman_result[1]}")
            print(f"Kendall correlation is {kendall_result[0]}, p-value is {kendall_result[1]}")
