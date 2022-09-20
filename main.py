#!/usr/bin/env python
# coding: utf-8

import os
import torch
from torch import nn
import argparse
from datasets import load_from_disk, load_metric, load_dataset
from transformers import AutoTokenizer, BartConfig
from architectures import Autoencoder, CBartForConditionalGeneration

# Set the parameters

parser = argparse.ArgumentParser()
    
parser.add_argument('--checkpoint_dir', type=str, help="The cBART model directory.", required=True)
parser.add_argument('--exp_name', type=str, help="The experiment name.", required=True)
parser.add_argument('--batch_size', type=int, help="The batch_size.", required=True)
parser.add_argument('--first', type=int, help="The AE's first projection.", required=True)
parser.add_argument('--second', type=int, help="The AE's second projection.", required=True)
parser.add_argument('--third', type=int, help="The AE's third projection.", required=True)
parser.add_argument('--test', action='store_true', help="A test run with just one sample.")

args = parser.parse_args()

bart_checkpoint    = 'facebook/bart-base'
encoder_max_length = 1024
log_directory      = os.path.join("./results/", args.exp_name)

os.makedirs(log_directory, exist_ok=True)
rouge = load_metric("rouge")


# The AutoEncoder

ae = Autoencoder(bart_encoder_emb_size=768,
                 first_proj=args.first,
                 second_proj=args.second,
                 third_proj=args.third,
                 max_len=encoder_max_length)


# Initialize compressedBART

cbart_config = BartConfig.from_pretrained(bart_checkpoint)
cbart_config.enc_d_model = cbart_config.d_model
cbart_config.d_model = args.third

CBart_model = CBartForConditionalGeneration.from_pretrained(args.checkpoint_dir,
                                                            config=cbart_config,
                                                            ae=ae,
                                                            ignore_mismatched_sizes=True)


# Load the weights and put it on GPU

ae_checkpoint = os.path.join( args.checkpoint_dir, "ae-checkpoint.pth" )
if torch.cuda.is_available():
    ae_checkpoint = torch.load(ae_checkpoint)

else:
    ae_checkpoint = torch.load(ae_checkpoint, map_location=torch.device('cpu'))

CBart_model.model.encoder.ae.load_state_dict(ae_checkpoint['model'])
del ae_checkpoint

CBart_model.eval()

if torch.cuda.is_available():
    CBart_model.to("cuda")


# Load the Tokenizer

tokenizer = AutoTokenizer.from_pretrained(bart_checkpoint, cache_dir="./hf-cache/bart-base")


# Load the Data

test_dataset = load_from_disk( '../../hf-cache/cnn_dailymail/{}'.format('test') )
# test_dataset = load_dataset("ccdv/cnn_dailymail", "3.0.0", cache_dir="./hf-cache/cnn_dailymail",
#                              split="test", ignore_verifications=True)

if args.test:
    test_dataset = test_dataset.select(range(1))


def generate_summary(batch):
    inputs_dict = tokenizer(batch["article"], padding="max_length", max_length=encoder_max_length, return_tensors="pt", truncation=True)
    input_ids = inputs_dict.input_ids
    attention_mask = inputs_dict.attention_mask
    
    if torch.cuda.is_available():
        input_ids = input_ids.to("cuda")
        attention_mask = attention_mask.to("cuda")

    predicted_abstract_ids = CBart_model.generate(input_ids, attention_mask=attention_mask, use_cache=True)
    batch["predicted_highlights"] = tokenizer.batch_decode(predicted_abstract_ids, skip_special_tokens=True)
    
    return batch


# Decoding Beam Search

CBart_model.config.max_length = 144
CBart_model.config.min_length = 55
CBart_model.config.no_repeat_ngram_size = 3
CBart_model.config.early_stopping = True
CBart_model.config.length_penalty = 2.0
CBart_model.config.num_beams = 4
    
result = test_dataset.map(generate_summary, batched=True, batch_size=args.batch_size)


# Save the results

result.to_csv( os.path.join(log_directory, 'beam.csv') )

beam_scores = rouge.compute(predictions=result["predicted_highlights"], references=result["highlights"],
                            rouge_types=["rouge1", "rouge2", "rouge3", "rougeLsum"])
    
for item, score in beam_scores.items():
    print(item)
    print("{:.3f}".format(score.mid.fmeasure))