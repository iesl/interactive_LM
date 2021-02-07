#! /usr/bin/env python3
# coding=utf-8

# Copyright (c) 2019 Uber Technologies, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Example command with bag of words:
python examples/run_pplm.py -B space --cond_text "The president" --length 100 --gamma 1.5 --num_iterations 3 --num_samples 10 --stepsize 0.01 --window_length 5 --kl_scale 0.01 --gm_scale 0.95

Example command with discriminator:
python examples/run_pplm.py -D sentiment --class_label 3 --cond_text "The lake" --length 10 --gamma 1.0 --num_iterations 30 --num_samples 10 --stepsize 0.01 --kl_scale 0.01 --gm_scale 0.95
"""

import argparse
import json
from operator import add
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from tqdm import trange

from pplm_classification_head import ClassificationHead

import sys
sys.path.insert(0, sys.path[0]+'/..')
from gpt2_model.tokenization_gpt2 import GPT2Tokenizer
from gpt2_model.file_utils import cached_path
#from transformers.modeling_gpt2 import GPT2LMHeadModel
from gpt2_model.modeling_gpt2_condition import GPT2LMHeadModel
from gpt2_model.configuration_gpt2 import GPT2Config

import os

PPLM_BOW = 1
PPLM_DISCRIM = 2
PPLM_BOW_DISCRIM = 3
SMALL_CONST = 1e-15
BIG_CONST = 1e10

# BAG_OF_WORDS_ARCHIVE_MAP = {
#     "legal": "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/bow/legal.txt",
#     "military": "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/bow/military.txt",
#     "politics": "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/bow/politics.txt",
#     "religion": "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/bow/religion.txt",
#     "science": "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/bow/science.txt",
#     "space": "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/bow/space.txt",
#     "technology": "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/bow/technology.txt",
#     "1": "http://m.uploadedit.com/bbtc/1578681958480.txt"
# }

DISCRIMINATOR_MODELS_PARAMS = {
    "clickbait": {
        "url": "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/discriminators/clickbait_classifier_head.pt",
        "class_size": 2,
        "embed_size": 1024,
        "class_vocab": {"non_clickbait": 0, "clickbait": 1},
        "default_class": 1,
        "pretrained_model": "gpt2-medium",
    },
    "sentiment": {
        "url": "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/discriminators/SST_classifier_head.pt",
        "class_size": 5,
        "embed_size": 1024,
        "class_vocab": {"very_positive": 2, "very_negative": 3},
        "default_class": 3,
        "pretrained_model": "gpt2-medium",
    },
}


def to_var(x, requires_grad=False, volatile=False, device="cuda"):
    if torch.cuda.is_available() and device == "cuda":
        x = x.cuda()
    elif device != "cuda":
        x = x.to(device)
    return Variable(x, requires_grad=requires_grad, volatile=volatile)


def top_k_filter(logits, k, probs=False):
    """
    Masks everything but the k top entries as -infinity (1e10).
    Used to mask logits such that e^-infinity -> 0 won't contribute to the
    sum of the denominator.
    """
    if k == 0:
        return logits
    else:
        values = torch.topk(logits, k)[0]
        batch_mins = values[:, -1].view(-1, 1).expand_as(logits)
        if probs:
            return torch.where(logits < batch_mins, torch.ones_like(logits) * 0.0, logits)
        return torch.where(logits < batch_mins, torch.ones_like(logits) * -BIG_CONST, logits)


def perturb_past(
    past,
    model,
    last,
    unpert_past=None,
    unpert_logits=None,
    accumulated_hidden=None,
    grad_norms=None,
    stepsize=0.01,
    one_hot_bows_vectors=None,
    classifier=None,
    class_label=None,
    loss_type=0,
    num_iterations=3,
    horizon_length=1,
    window_length=0,
    decay=False,
    gamma=1.5,
    kl_scale=0.01,
    device="cuda",
):
    with torch.enable_grad():
        # Generate inital perturbed past
        grad_accumulator = [(np.zeros(p.shape).astype("float32")) for p in past]


        if accumulated_hidden is None:
            accumulated_hidden = 0

        if decay:
            decay_mask = torch.arange(0.0, 1.0 + SMALL_CONST, 1.0 / (window_length))[1:]
        else:
            decay_mask = 1.0

        # TODO fix this comment (SUMANTH)
        # Generate a mask is gradient perturbated is based on a past window
        _, _, _, curr_length, _ = past[0].shape

        if curr_length > window_length and window_length > 0:
            ones_key_val_shape = tuple(past[0].shape[:-2]) + tuple([window_length]) + tuple(past[0].shape[-1:])

            zeros_key_val_shape = (
                tuple(past[0].shape[:-2]) + tuple([curr_length - window_length]) + tuple(past[0].shape[-1:])
            )

            ones_mask = torch.ones(ones_key_val_shape)
            ones_mask = decay_mask * ones_mask.permute(0, 1, 2, 4, 3)
            ones_mask = ones_mask.permute(0, 1, 2, 4, 3)

            window_mask = torch.cat((ones_mask, torch.zeros(zeros_key_val_shape)), dim=-2).to(device)
        else:
            window_mask = torch.ones_like(past[0]).to(device)

        # accumulate perturbations for num_iterations
        loss_per_iter = []
        new_accumulated_hidden = None
        for i in range(num_iterations):
            # print("Iteration ", i + 1)
            curr_perturbation = [
                to_var(torch.from_numpy(p_), requires_grad=True, device=device) for p_ in grad_accumulator
            ]

            # Compute hidden using perturbed past
            perturbed_past = list(map(add, past, curr_perturbation))
            _, _, _, curr_length, _ = curr_perturbation[0].shape
            all_logits, _, all_hidden = model(last, past=perturbed_past)
            hidden = all_hidden[-1]
            new_accumulated_hidden = accumulated_hidden + torch.sum(hidden, dim=1).detach()
            # TODO: Check the layer-norm consistency of this with trained discriminator (Sumanth)
            logits = all_logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)

            loss = 0.0
            loss_list = []
            if loss_type == PPLM_BOW or loss_type == PPLM_BOW_DISCRIM:
                for one_hot_bow in one_hot_bows_vectors:
                    bow_logits = torch.mm(probs, torch.t(one_hot_bow))
                    bow_loss = -torch.log(torch.sum(bow_logits))
                    loss += bow_loss
                    loss_list.append(bow_loss)
                # print(" pplm_bow_loss:", loss.data.cpu().numpy())

            if loss_type == 2 or loss_type == 3:
                ce_loss = torch.nn.CrossEntropyLoss()
                # TODO why we need to do this assignment and not just using unpert_past? (Sumanth)
                curr_unpert_past = unpert_past
                curr_probs = torch.unsqueeze(probs, dim=1)
                wte = model.resize_token_embeddings()
                for _ in range(horizon_length):
                    inputs_embeds = torch.matmul(curr_probs, wte.weight.data)
                    _, curr_unpert_past, curr_all_hidden = model(past=curr_unpert_past, inputs_embeds=inputs_embeds)
                    curr_hidden = curr_all_hidden[-1]
                    new_accumulated_hidden = new_accumulated_hidden + torch.sum(curr_hidden, dim=1)

                prediction = classifier(new_accumulated_hidden / (curr_length + 1 + horizon_length))

                label = torch.tensor(prediction.shape[0] * [class_label], device=device, dtype=torch.long)
                discrim_loss = ce_loss(prediction, label)
                # print(" pplm_discrim_loss:", discrim_loss.data.cpu().numpy())
                loss += discrim_loss
                loss_list.append(discrim_loss)

            kl_loss = 0.0
            if kl_scale > 0.0:
                unpert_probs = F.softmax(unpert_logits[:, -1, :], dim=-1)
                unpert_probs = unpert_probs + SMALL_CONST * (unpert_probs <= SMALL_CONST).float().to(device).detach()
                correction = SMALL_CONST * (probs <= SMALL_CONST).float().to(device).detach()
                corrected_probs = probs + correction.detach()
                kl_loss = kl_scale * ((corrected_probs * (corrected_probs / unpert_probs).log()).sum())
                # print(" kl_loss", kl_loss.data.cpu().numpy())
                loss += kl_loss

            loss_per_iter.append(loss.data.cpu().numpy())
            # print(" pplm_loss", (loss - kl_loss).data.cpu().numpy())

            # compute gradients
            loss.backward()

            # calculate gradient norms
            if grad_norms is not None and loss_type == PPLM_BOW:
                grad_norms = [
                    torch.max(grad_norms[index], torch.norm(p_.grad * window_mask))
                    for index, p_ in enumerate(curr_perturbation)
                ]
            else:
                grad_norms = [
                    (torch.norm(p_.grad * window_mask) + SMALL_CONST) for index, p_ in enumerate(curr_perturbation)
                ]

            # normalize gradients
            grad = [
                -stepsize * (p_.grad * window_mask / grad_norms[index] ** gamma).data.cpu().numpy()
                for index, p_ in enumerate(curr_perturbation)
            ]

            # accumulate gradient
            grad_accumulator = list(map(add, grad, grad_accumulator))

            # reset gradients, just to make sure
            for p_ in curr_perturbation:
                p_.grad.data.zero_()

            # removing past from the graph
            new_past = []
            for p_ in past:
                new_past.append(p_.detach())
            past = new_past

        # apply the accumulated perturbations to the past
        grad_accumulator = [to_var(torch.from_numpy(p_), requires_grad=True, device=device) for p_ in grad_accumulator]
        pert_past = list(map(add, past, grad_accumulator))

        return pert_past, new_accumulated_hidden, grad_norms, loss_per_iter


def get_classifier(
    name: Optional[str], class_label: Union[str, int], device: str
) -> Tuple[Optional[ClassificationHead], Optional[int]]:
    if name is None:
        return None, None

    params = DISCRIMINATOR_MODELS_PARAMS[name]
    classifier = ClassificationHead(class_size=params["class_size"], embed_size=params["embed_size"]).to(device)
    if "url" in params:
        resolved_archive_file = cached_path(params["url"])
    elif "path" in params:
        resolved_archive_file = params["path"]
    else:
        raise ValueError("Either url or path have to be specified " "in the discriminator model parameters")
    classifier.load_state_dict(torch.load(resolved_archive_file, map_location=device))
    classifier.eval()

    if isinstance(class_label, str):
        if class_label in params["class_vocab"]:
            label_id = params["class_vocab"][class_label]
        else:
            label_id = params["default_class"]
            print("class_label {} not in class_vocab".format(class_label))
            print("available values are: {}".format(params["class_vocab"]))
            print("using default class {}".format(label_id))

    elif isinstance(class_label, int):
        if class_label in set(params["class_vocab"].values()):
            label_id = class_label
        else:
            label_id = params["default_class"]
            print("class_label {} not in class_vocab".format(class_label))
            print("available values are: {}".format(params["class_vocab"]))
            print("using default class {}".format(label_id))

    else:
        label_id = params["default_class"]

    return classifier, label_id


def get_bag_of_words_indices(bag_of_words: List[str], tokenizer) -> List[List[List[int]]]:
    bow_indices = []
    
    bow_indices.append([tokenizer.encode(word.strip(), add_prefix_space=True, add_special_tokens=False) for word in bag_of_words])

    return bow_indices


def build_bows_one_hot_vectors(bow_indices, tokenizer, device="cuda"):
    if bow_indices is None:
        return None

    one_hot_bows_vectors = []

    for single_bow in bow_indices:
        single_bow = list(filter(lambda x: len(x) <= 1, single_bow))
        single_bow = torch.tensor(single_bow).to(device)
        num_words = single_bow.shape[0]
        #if num_words == 0:
        #    single_bow = [[318]]
        #    single_bow = torch.tensor(single_bow).to(device)
        #    num_words = 1
        one_hot_bow = torch.zeros(num_words, tokenizer.vocab_size).to(device)
        one_hot_bow.scatter_(1, single_bow, 1)
        one_hot_bows_vectors.append(one_hot_bow)
    return one_hot_bows_vectors


def full_text_generation(
    model,
    tokenizer,
    context=None,
    num_samples=1,
    device="cuda",
    bag_of_words=None,
    discrim=None,
    class_label=None,
    length=100,
    stepsize=0.02,
    temperature=1.0,
    top_k=10,
    sample=False,
    num_iterations=3,
    grad_length=10000,
    horizon_length=1,
    window_length=0,
    decay=False,
    gamma=1.5,
    gm_scale=0.9,
    kl_scale=0.01,
    **kwargs
):
    classifier, class_id = get_classifier(discrim, class_label, device)

    bow_indices = []
    if bag_of_words:
        bow_indices = get_bag_of_words_indices(bag_of_words, tokenizer)


    if bag_of_words and classifier:
        # print("Both PPLM-BoW and PPLM-Discrim are on. This is not optimized.")
        loss_type = PPLM_BOW_DISCRIM

    elif bag_of_words:
        loss_type = PPLM_BOW
        # print("Using PPLM-BoW")

    elif classifier is not None:
        loss_type = PPLM_DISCRIM
        # print("Using PPLM-Discrim")

    else:
        raise Exception("Specify either a bag of words or a discriminator")

    if device == "cuda":
        torch.cuda.empty_cache()

    pert_gen_tok_texts = []
    discrim_losses = []
    losses_in_time = []
    # generated_texts = []


    context_t = torch.tensor(context, device=device, dtype=torch.long)


    pert_gen_tok_texts, discrim_losses, losses_in_time, generated_texts, perplexity = generate_text_pplm(
        model=model,
        tokenizer=tokenizer,
        context=context_t.unsqueeze(0).expand(num_samples,context_t.shape[0]),
        device=device,
        perturb=True,
        bow_indices=bow_indices,
        classifier=classifier,
        class_label=class_id,
        loss_type=loss_type,
        length=length,
        stepsize=stepsize,
        temperature=temperature,
        top_k=top_k,
        sample=sample,
        num_iterations=num_iterations,
        grad_length=grad_length,
        horizon_length=horizon_length,
        window_length=window_length,
        decay=decay,
        gamma=gamma,
        gm_scale=gm_scale,
        kl_scale=kl_scale,
    )
    return pert_gen_tok_texts, discrim_losses, losses_in_time, generated_texts, perplexity


def generate_text_pplm(
    model,
    tokenizer,
    context=None,
    past=None,
    device="cuda",
    perturb=True,
    bow_indices=None,
    classifier=None,
    class_label=None,
    loss_type=0,
    length=100,
    stepsize=0.02,
    temperature=1.0,
    top_k=10,
    sample=False,
    num_iterations=3,
    grad_length=10000,
    horizon_length=1,
    window_length=0,
    decay=False,
    gamma=1.5,
    gm_scale=0.9,
    kl_scale=0.01,
):
    output_so_far = context

    sample_num = output_so_far.shape[0]
    # if context:
    #     context_t = torch.tensor(context, device=device, dtype=torch.long)
    #     while len(context_t.shape) < 2:
    #         context_t = context_t.unsqueeze(0)
    #     output_so_far = context_t

    # collect one hot vectors for bags of words
    one_hot_bows_vectors = build_bows_one_hot_vectors(bow_indices, tokenizer, device)

    grad_norms = None
    last = None
    unpert_discrim_loss = 0
    loss_in_time = []
    generated = None
    generated_prob = torch.ones(sample_num)


    for i in trange(length, ascii=True):

        # Get past/probs for current output, except for last word
        # Note that GPT takes 2 inputs: past + current_token

        # run model forward to obtain unperturbed
        if past is None and output_so_far is not None:
            last = output_so_far[:, -1:]
            if output_so_far.shape[1] > 1:
                _, past, _ = model(output_so_far[:, :-1])

        unpert_logits, unpert_past, unpert_all_hidden = model(output_so_far)
        unpert_last_hidden = unpert_all_hidden[-1]

        # check if we are abowe grad max length
        if i >= grad_length:
            current_stepsize = stepsize * 0
        else:
            current_stepsize = stepsize

        # modify the past if necessary
        if not perturb or num_iterations == 0:
            pert_past = past

        else:
            accumulated_hidden = unpert_last_hidden[:, :-1, :]
            accumulated_hidden = torch.sum(accumulated_hidden, dim=1)

            if past is not None:
                pert_past, _, grad_norms, loss_this_iter = perturb_past(
                    past,
                    model,
                    last,
                    unpert_past=unpert_past,
                    unpert_logits=unpert_logits,
                    accumulated_hidden=accumulated_hidden,
                    grad_norms=grad_norms,
                    stepsize=current_stepsize,
                    one_hot_bows_vectors=one_hot_bows_vectors,
                    classifier=classifier,
                    class_label=class_label,
                    loss_type=loss_type,
                    num_iterations=num_iterations,
                    horizon_length=horizon_length,
                    window_length=window_length,
                    decay=decay,
                    gamma=gamma,
                    kl_scale=kl_scale,
                    device=device,
                )
                loss_in_time.append(loss_this_iter)
            else:
                pert_past = past

        pert_logits, past, pert_all_hidden = model(last, past=pert_past)
        pert_logits = pert_logits[:, -1, :] / temperature  # + SMALL_CONST
        pert_probs = F.softmax(pert_logits, dim=-1)

        if classifier is not None:
            ce_loss = torch.nn.CrossEntropyLoss()
            prediction = classifier(torch.mean(unpert_last_hidden, dim=1))
            label = torch.tensor([class_label], device=device, dtype=torch.long)
            unpert_discrim_loss = ce_loss(prediction, label)
            # print("unperturbed discrim loss", unpert_discrim_loss.data.cpu().numpy())
        else:
            unpert_discrim_loss = 0

        # Fuse the modified model and original model
        if perturb:

            unpert_probs = F.softmax(unpert_logits[:, -1, :], dim=-1)

            pert_probs = (pert_probs ** gm_scale) * (unpert_probs ** (1 - gm_scale))  # + SMALL_CONST
            
            pert_probs = top_k_filter(pert_probs, k=top_k, probs=True)  # + SMALL_CONST

            # rescale
            if torch.sum(pert_probs) <= 1:
                pert_probs = pert_probs / torch.sum(pert_probs)

            
            

        else:
            pert_logits = top_k_filter(pert_logits, k=top_k)  # + SMALL_CONST
            pert_probs = F.softmax(pert_logits, dim=-1)

        # sample or greedy
        if sample:
            last = torch.multinomial(pert_probs, num_samples=1)

        else:
            _, last = torch.topk(pert_probs, k=1, dim=-1)

        for j in range(sample_num):
            last_prob = pert_probs[j][last[j]].tolist()[0]
            generated_prob[j] *= last_prob


        # update context/output_so_far appending the new token
        output_so_far = last if output_so_far is None else torch.cat((output_so_far, last), dim=1)

        generated = last if generated is None else torch.cat((generated, last), dim=1)

        
    perplexity = 1/generated_prob
    power = 1/length
    perplexity = perplexity.pow(power)

    return output_so_far, unpert_discrim_loss, loss_in_time, generated, perplexity


def set_generic_model_params(discrim_weights, discrim_meta):
    if discrim_weights is None:
        raise ValueError("When using a generic discriminator, " "discrim_weights need to be specified")
    if discrim_meta is None:
        raise ValueError("When using a generic discriminator, " "discrim_meta need to be specified")

    with open(discrim_meta, "r") as discrim_meta_file:
        meta = json.load(discrim_meta_file)
    meta["path"] = discrim_weights
    DISCRIMINATOR_MODELS_PARAMS["generic"] = meta



def run_pplm_example(
    pretrained_model="gpt2-large",
    cond_text="",
    uncond=False,
    num_samples=1,
    bag_of_words=None,
    discrim=None,
    discrim_weights=None,
    discrim_meta=None,
    class_label=-1,
    length=100,
    stepsize=0.02,
    temperature=1.0,
    top_k=10,
    sample=False,
    num_iterations=3,
    grad_length=10000,
    horizon_length=1,
    window_length=0,
    decay=False,
    gamma=1.5,
    gm_scale=0.9,
    kl_scale=0.01,
    seed=0,
    no_cuda=False,
    colorama=False,
):
    # set Random seed
    torch.manual_seed(seed)
    np.random.seed(seed)


    # setup the device
    device = "cuda" if torch.cuda.is_available() and not no_cuda else "cpu"

    if discrim == "generic":
        set_generic_model_params(discrim_weights, discrim_meta)

    print("pretrained_model set to discriminator's = {}".format(pretrained_model))

    if discrim is not None:
        pretrained_model = DISCRIMINATOR_MODELS_PARAMS[discrim]["pretrained_model"]
        print("discrim = {}, pretrained_model set " "to discriminator's = {}".format(discrim, pretrained_model))

    # load pretrained model
    model = GPT2LMHeadModel.from_pretrained(pretrained_model, output_hidden_states=True)
    model.to(device)
    model.eval()

    # load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model)

    # Freeze GPT-2 weights
    for param in model.parameters():
        param.requires_grad = False

    


class pplm():
    def __init__(
        self,
        seed,
        pretrained_model,
        checkpoint,
        device,
        discrim=None,
        discrim_weights=None,
        discrim_meta=None,
        class_label=-1,
        ):

        
        # load data
        self.device = device
        self.class_label = class_label
        self.discrim = discrim
        self.discrim_weights = discrim_weights
        self.discrim_meta = discrim_meta


        # set Random seed
        torch.manual_seed(seed)
        np.random.seed(seed)


        # self.device = device
        # # load model
        encoder_state_dict = torch.load(os.path.join(checkpoint, 'encoder.pt'), map_location=device)
        gpt2_config = GPT2Config.from_pretrained(pretrained_model, output_hidden_states=True)
        gpt2_config.word_emb_dim = 300
        self.model = GPT2LMHeadModel.from_pretrained(pretrained_model, state_dict = encoder_state_dict, config = gpt2_config)
        
        #self.model = GPT2LMHeadModel.from_pretrained(pretrained_model, output_hidden_states=True)
        self.model.to(self.device)
        self.model.eval()

        # # load tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model)

        # # Freeze GPT-2 weights
        for param in self.model.parameters():
            param.requires_grad = False
        print("pplm model: ",self.device)

        


    def run_pplm_example(self,
        cond_text="",
        uncond=False,
        num_samples=1,
        bag_of_words=None,
        length=100,
        stepsize=0.02,
        temperature=1.0,
        top_k=10,
        sample=False,
        num_iterations=1,
        grad_length=10000,
        horizon_length=1,
        window_length=0,
        decay=False,
        gamma=1.5,
        gm_scale=0.9,
        kl_scale=0.01,
        colorama=False,
        ):
        # figure out conditioning text
        if uncond:
            tokenized_cond_text = self.tokenizer.encode([self.tokenizer.bos_token])
        else:
            raw_text = cond_text
            while not raw_text:
                print("Did you forget to add `--cond_text`? ")
                raw_text = input("Model prompt >>> ")
            tokenized_cond_text = self.tokenizer.encode(self.tokenizer.bos_token + raw_text)



        # full_text_generation returns:
        # pert_gen_tok_texts, discrim_losses, losses_in_time, generated_text
        pert_gen_tok_texts, _, _, generated_texts, perplexity= full_text_generation(
            model=self.model,
            tokenizer=self.tokenizer,
            context=tokenized_cond_text,
            device=self.device,
            num_samples=num_samples,
            bag_of_words=bag_of_words,
            discrim=self.discrim,
            class_label=self.class_label,
            length=length,
            stepsize=stepsize,
            temperature=temperature,
            top_k=top_k,
            sample=sample,
            num_iterations=num_iterations,
            grad_length=grad_length,
            horizon_length=horizon_length,
            window_length=window_length,
            decay=decay,
            gamma=gamma,
            gm_scale=gm_scale,
            kl_scale=kl_scale,
        )

        # # untokenize unperturbed text
        # pert_gen_text = self.tokenizer.decode(generated_texts[0].tolist()[0])

        # bow_word_ids = set()
        # if bag_of_words and colorama:
        #     bow_indices = get_bag_of_words_indices(bag_of_words, self.tokenizer)
        #     for single_bow_list in bow_indices:
        #         # filtering all words in the list composed of more than 1 token
        #         filtered = list(filter(lambda x: len(x) <= 1, single_bow_list))
        #         # w[0] because we are sure w has only 1 item because previous fitler
        #         bow_word_ids.update(w[0] for w in filtered)

        # # # iterate through the perturbed texts
        texts = []
        for i, generated_text in enumerate(generated_texts.tolist()):
            gen_text = self.tokenizer.decode(generated_text)
            texts.append(gen_text)
        #     try:
        #         # untokenize unperturbed text
        #         if colorama:
        #             import colorama

        #             pert_gen_text = ""
        #             for word_id in generated_text.tolist()[0]:
        #                 if word_id in bow_word_ids:
        #                     pert_gen_text += "{}{}{}".format(
        #                         colorama.Fore.RED, self.tokenizer.decode([word_id]), colorama.Style.RESET_ALL
        #                     )
        #                 else:
        #                     pert_gen_text += self.tokenizer.decode([word_id])
        #         else:
        #             pert_gen_text = self.tokenizer.decode(generated_text.tolist()[0])
        #     except Exception as exc:
        #         print("Ignoring error while generating perturbed text:", exc)

        return texts, perplexity
        

        
def main():
    device_pplm = torch.device("cuda")
    pplm_model = pplm(seed = 0, pretrained_model = 'gpt2-medium', device = device_pplm)
    context = "Hello, my name is "
    bag_of_words1 = ['Barrichello', 'Rosberg', 'Kovalainen', 'Coulthard', 'Fisichella']
    bag_of_words2 = ['these', 'are', 'easy', 'words', 'ohmygodthatisnoteasy']
    bag_of_words3 = ["is"]
    gen_text, _ = pplm_model.run_pplm_example(context, False, 3, bag_of_words1, 5, 0.05, 1.0, 5, True, 1, 10000, 1, 0, False, 1.5, 0.9, 0.01, True)
    print(gen_text)

  
if __name__== "__main__":
    main()

