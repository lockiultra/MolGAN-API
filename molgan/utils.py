from fast_jtnn.jtnn_vae import JTNNVAE
from fast_jtnn.vocab import Vocab
from DiseasePipeline import DiseasePipeline
import tensorflow.keras as keras
from MPNN import MessagePassing, TransformerEncoderReadout
import pandas as pd
import torch

def get_vocab() -> Vocab:
    with open('./data/vocab.txt', 'r') as f:
        vocab_list = [smiles for smiles in f.read().split('\n')]
    return Vocab(vocab_list)

def get_config() -> dict:
    config = {
        'vocab': get_vocab(),
        'hidden_size': 450,
        'latent_size': 56,
        'depthT': 3,
        'depthG': 3
    }
    return config

def get_model() -> JTNNVAE:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = get_config()
    model = JTNNVAE(**config)
    model.load_state_dict(torch.load('./models/pretrained/jtvae.pth', map_location=device))
    return model

def get_disease_pipeline() -> dict:
    dp = DiseasePipeline()
    for disease in dp.diseases:
        dp.models[disease] = keras.models.load_model(f'./Models/model_{disease}.h5', compile=False, custom_objects={'MessagePassing': MessagePassing, 'TransformerEncoderReadout': TransformerEncoderReadout})
    dp.is_trained = True
    return dp

def predict_to_json(predict_dict: dict) -> dict:
    return {disease: float(prob[0]) for disease, prob in predict_dict.items()}