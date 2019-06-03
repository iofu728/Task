# -*- coding: utf-8 -*- 
# Author: Shengqiang Zhang
# Time  : 2019/5/18 0:44

from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from nltk.translate import bleu_score
from tqdm import tqdm

from model import Dataset
from model import Model
from model import generate_nmt_batches


def set_seed_everywhere(seed, cuda):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed_all(seed)


def handle_dirs(dirpath):
    if not Path(dirpath).exists():
        Path(dirpath).mkdir(parents=True)


def make_train_states(args):
    return {
        "early_stopping": False,
        "early_stopping_step": 0,
        "early_stopping_best_val": 1e8,
        "learning_rate": args.learning_rate,
        "epoch_index": 0,
        "train_loss": [],
        "train_acc": [],
        "dev_loss": [],
        "dev_acc": [],
        "test_loss": -1,
        "test_acc": -1,
        "model_filename": args.model_state_file
    }


def update_train_state(args, model, train_state):
    """
    Handle the training states update.
    Components: Early stopping and model checkpoint saving.
    :param args:
    :param model:
    :param train_state:
    :return: a new train state
    """

    # save at least one model
    if train_state["epoch_index"] == 0:
        torch.save(model.state_dict(), train_state["model_filename"])
        train_state["early_stopping"] = False

    # Save model if performance improved
    if train_state["epoch_index"] >= 1:
        loss_tml, loss_t = train_state["dev_loss"][-2:]
        if loss_t >= loss_tml:
            train_state["early_stopping_step"] += 1
        else:
            if loss_t < train_state["early_stopping_best_val"]:
                torch.save(model.state_dict(), train_state["model_filename"])
                train_state["early_stopping_best_val"] = loss_t
            train_state["early_stopping_step"] = 0

        train_state["early_stopping"] = \
            train_state["early_stopping_step"] >= args.early_stopping_criteria

    return train_state


def normalize_sizes(y_pred, y_true):
    """
    Normalize tensor sizes
    :param y_pred: model predictions, if a 3-dim tensor, reshapes to a matrix
    :param y_true: target predictions, if a matrix, reshapes to a vector
    :return:
    """
    if len(y_pred.size()) == 3:
        y_pred = y_pred.contiguous().view(-1, y_pred.size(2))
    if len(y_true.size()) == 2:
        y_true = y_true.contiguous().view(-1)

    return y_pred, y_true


def compute_accuracy(y_pred, y_true, mask_index):
    """

    :param y_pred:
    :param y_true:
    :param mask_index:
    :return:
    """
    y_pred, y_true = normalize_sizes(y_pred, y_true)
    _, y_pred_indices = y_pred.max(dim=1)

    correct_indices = torch.eq(y_pred_indices, y_true).float()
    valid_indices = torch.ne(y_true, mask_index).float()

    n_correct = (correct_indices * valid_indices).sum().item()
    n_valid = valid_indices.sum().item()

    return n_correct / n_valid * 100


def sequence_loss(y_pred, y_true, mask_index):
    y_pred, y_true = normalize_sizes(y_pred, y_true)
    return F.cross_entropy(y_pred, y_true, ignore_index=mask_index)


args = Namespace(
    dataset_csv="data/out_mspars.csv",
    vectorizer_file="vectorizer.json",
    glove_file="dataset/glove.6b.100d.txt",
    model_state_file="model.pth",
    save_dir="model_storage_no_unk/",
    reload_from_files=True,
    expand_filepaths_to_save_dir=True,
    cuda=False,
    seed=1337,
    learning_rate=5e-4,
    batch_size=64,
    num_epochs=1,
    early_stopping_criteria=5,
    source_embedding_size=64,
    target_embedding_size=64,
    encoding_size=64,
    catch_keyboard_interrupt=True
)

if args.expand_filepaths_to_save_dir:
    args.vectorizer_file = Path(args.save_dir, args.vectorizer_file)
    args.model_state_file = Path(args.save_dir, args.model_state_file)

    print("Expanded filepaths:")
    print("\t{}".format(args.vectorizer_file))
    print("\t{}".format(args.model_state_file))

if torch.cuda.is_available():
    args.cuda = True
else:
    args.cuda = False

args.device = torch.device("cuda" if args.cuda else "cpu")

print("Using CUDA: {}".format(args.cuda))

set_seed_everywhere(args.seed, args.cuda)

handle_dirs(args.save_dir)

if args.reload_from_files and Path(args.vectorizer_file).exists():
    dataset = Dataset.load_dataset_and_load_vectorizer(args.dataset_csv, args.vectorizer_file)
else:
    dataset = Dataset.load_dataset_and_make_vectorizer(args.dataset_csv)
    dataset.save_vectorizer(args.vectorizer_file)

vectorizer = dataset.get_vectorizer()

model = Model(source_vocab_size=len(vectorizer.source_vocab),
              source_embedding_size=args.source_embedding_size,
              target_vocab_size=len(vectorizer.target_vocab),
              target_embedding_size=args.target_embedding_size,
              encoding_size=args.encoding_size,
              target_bos_index=vectorizer.target_vocab.begin_seq_index,
              )

if args.reload_from_files and Path(args.model_state_file).exists():
    model.load_state_dict(torch.load(args.model_state_file))
    print("Reloaded model")
else:
    print("New model")

model = model.to(args.device)

optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                 mode="min",
                                                 factor=0.5,
                                                 patience=1)
mask_index = vectorizer.target_vocab.mask_index
train_state = make_train_states(args)

epoch_bar = tqdm(desc='training routine',
                 total=args.num_epochs,
                 position=0)

dataset.set_split('train')
train_bar = tqdm(desc='split=train',
                 total=dataset.get_num_batches(args.batch_size),
                 position=1,
                 leave=True)
dataset.set_split('dev')
dev_bar = tqdm(desc='split=dev',
               total=dataset.get_num_batches(args.batch_size),
               position=1,
               leave=True)

for epoch_index in range(args.num_epochs):
    sample_probability = (20 + epoch_index) / args.num_epochs

    train_state["epoch_index"] = epoch_index

    # Iterate over train dataset
    dataset.set_split("train")
    train_batch_generator = generate_nmt_batches(dataset=dataset,
                                                 batch_size=args.batch_size,
                                                 shuffle=True,
                                                 device=args.device)
    running_loss = 0.0
    running_acc = 0.0
    model.train()

    for batch_index, batch_dict in enumerate(train_batch_generator):
        optimizer.zero_grad()
        y_pred = model(batch_dict["x_source"],
                       batch_dict["x_source_length"],
                       batch_dict["x_target"],
                       sample_probability)
        loss = sequence_loss(y_pred, batch_dict["y_target"], mask_index)
        loss.backward()
        optimizer.step()

        running_loss += (loss.item() - running_loss) / (batch_index + 1)
        acc_t = compute_accuracy(y_pred, batch_dict["y_target"], mask_index)
        running_acc += (acc_t - running_acc) / (batch_index + 1)

        # update bar
        train_bar.set_postfix(loss=running_loss, acc=running_acc,
                              epoch=epoch_index)
        train_bar.update()

    train_state["train_loss"].append(running_loss)
    train_state["train_acc"].append(running_acc)

    # Iterate over dev dataset
    dataset.set_split("dev")
    dev_batch_generator = generate_nmt_batches(dataset=dataset,
                                               batch_size=args.batch_size,
                                               device=args.device)
    running_loss = 0.
    running_acc = 0.
    model.eval()

    for batch_index, batch_dict in enumerate(dev_batch_generator):
        y_pred = model(batch_dict["x_source"],
                       batch_dict["x_source_length"],
                       batch_dict["x_target"],
                       sample_probability)
        loss = sequence_loss(y_pred, batch_dict["y_target"], mask_index)

        running_loss += (loss.item() - running_loss) / (batch_index + 1)
        acc_t = compute_accuracy(y_pred, batch_dict["y_target"], mask_index)
        running_acc += (acc_t - running_acc) / (batch_index + 1)
        # Update bar
        dev_bar.set_postfix(loss=running_loss, acc=running_acc,
                            epoch=epoch_index)
        dev_bar.update()

    train_state["dev_loss"].append(running_loss)
    train_state["dev_acc"].append(running_acc)

    train_state = update_train_state(args, model, train_state)
    scheduler.step(train_state["dev_loss"][-1])

    if train_state["early_stopping"]:
        break

    train_bar.n = 0
    dev_bar.n = 0
    epoch_bar.set_postfix(best_val=train_state['early_stopping_best_val'])
    epoch_bar.update()


def sentence_from_indices(indices, vocab, strict=True, return_string=True):
    ignore_indices = {vocab.mask_index, vocab.begin_seq_index, vocab.end_seq_index}

    out = []
    for index in indices:
        if index == vocab.begin_seq_index and strict:
            continue
        elif index == vocab.end_seq_index and strict:
            break
        else:
            out.append(vocab.lookup_index(index))
    if return_string:
        return " ".join(out)
    else:
        return out


chencherry = bleu_score.SmoothingFunction()


class NMTSampler:
    def __init__(self, vectorizer, model):
        self.vectorizer = vectorizer
        self.model = model

    def apply_to_batch(self, batch_dict):
        self._last_batch = batch_dict
        y_pred = self.model(x_source=batch_dict['x_source'],
                            x_source_lengths=batch_dict['x_source_length'],
                            target_sequence=batch_dict['x_target'])
        self._last_batch['y_pred'] = y_pred

        # attention_batched = np.stack(self.model.decoder._cache_p_attn).transpose(1, 0, 2)
        # self._last_batch['attention'] = attention_batched

    def _get_source_sentence(self, index, return_string=True):
        indices = self._last_batch['x_source'][index].cpu().detach().numpy()
        vocab = self.vectorizer.source_vocab
        return sentence_from_indices(indices, vocab, return_string=return_string)

    def _get_reference_sentence(self, index, return_string=True):
        indices = self._last_batch['y_target'][index].cpu().detach().numpy()
        vocab = self.vectorizer.target_vocab
        return sentence_from_indices(indices, vocab, return_string=return_string)

    def _get_sampled_sentence(self, index, return_string=True):
        _, all_indices = torch.max(self._last_batch['y_pred'], dim=2)
        sentence_indices = all_indices[index].cpu().detach().numpy()
        vocab = self.vectorizer.target_vocab
        return sentence_from_indices(sentence_indices, vocab, return_string=return_string)

    def get_ith_item(self, index, return_string=True):
        output = {"source": self._get_source_sentence(index, return_string=return_string),
                  "reference": self._get_reference_sentence(index, return_string=return_string),
                  "sampled": self._get_sampled_sentence(index, return_string=return_string)}

        reference = output['reference']
        hypothesis = output['sampled']

        if not return_string:
            reference = " ".join(reference)
            hypothesis = " ".join(hypothesis)

        output['bleu-4'] = bleu_score.sentence_bleu(references=[reference],
                                                    hypothesis=hypothesis,
                                                    smoothing_function=chencherry.method1)

        return output


model = model.eval().to(args.device)

sampler = NMTSampler(vectorizer, model)

dataset.set_split('test')
batch_generator = generate_nmt_batches(dataset,
                                       batch_size=args.batch_size,
                                       shuffle=False,
                                       device=args.device)

test_results = []
for batch_dict in batch_generator:
    sampler.apply_to_batch(batch_dict)
    for i in range(args.batch_size):
        test_results.append(sampler.get_ith_item(i, False))

with open("results.txt", "w") as f:
    for item in test_results:
        f.write(str(item))
        f.write("\n")
