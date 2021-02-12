"""
In this module a SciBERT model is trained and finetuned. The transformation is followed by a
classification layer. Then the model is stored and evaluated.
"""
import os
import string
import random
import transformers
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
import torch
import evaluation
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler


titles = []
with open("Dataset_Titles.txt") as titles_file:
    for line in titles_file:
        titles.append(line.replace("\n", ""))

torch.cuda.empty_cache()
#open data, split it into dataset name and query text, erase dataset name from query text and store
#related dataset and query as tuple in data_list (for citations encoding 'ISO-8859-1' needs to be
#specified)
with open("Abstracts_New_Database.txt") as f:
#with open("Citation_New_Database.txt", encoding='ISO-8859-1') as f:
    data_list = []
    for i, line in enumerate(f):
        #for abstracts use the following
        dataset = str(str(line).split("\t")[2]).replace("\n", "").strip()
        #for citations use the following
        #dataset = str(str(line).split("\t")[0]).replace("\n", "").strip()
        text = str(line).split("\t")[1]
        text = text.replace(dataset, "")
        for title in titles:
            text = text.replace(title, "")
        tokens = word_tokenize(text)
        if len(tokens) > 512:
            senteces = text.split(".")
            for x in senteces:
                tokens_x = word_tokenize(x)
                if len(tokens_x) > 3 and len(tokens_x) <= 512:
                    x = x.translate(str.maketrans("", "", string.punctuation))
                    data_tuple = (dataset, x)
                    data_list.append(data_tuple)
        else:
            text = text.translate(str.maketrans("", "", string.punctuation))
            data_tuple = (dataset, text)
            data_list.append(data_tuple)

datasets, queries = zip(*data_list)
le = LabelEncoder()
le.fit(datasets)
LabelEncoder()
labels = list(le.classes_)
dataset_labels = le.transform(datasets)

print("Data is read in, stored in a list and labels are encoded")
input_ids = []
tokenizer = transformers.BertTokenizer.from_pretrained('allenai/scibert_scivocab_uncased',
                                                       do_lower_case=True)
for sent in queries:
    encoded_sent = tokenizer.encode(sent, add_special_tokens=True, max_length=512)
    input_ids.append(encoded_sent)

print("Sentences are encoded")

#truncate and pad senteces
MAX_LEN = 512
input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", value=0, truncating="post",
                          padding="post")

#create attention masks
attention_masks = []
for sent in input_ids:
    att_mask = [int(token_id > 0) for token_id in sent]
    attention_masks.append(att_mask)

#split data in training and test dataset
train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(input_ids,
                                                                                    dataset_labels,
                                                                                    test_size=0.2)
train_masks, validation_masks, _, _ = train_test_split(attention_masks, dataset_labels,
                                                       test_size=0.2)

#convert inputs and labels to torch tensors
train_inputs = torch.tensor(train_inputs)
validation_inputs = torch.tensor(validation_inputs)
train_labels = torch.tensor(train_labels)
validation_labels = torch.tensor(validation_labels)
train_masks = torch.tensor(train_masks)
validation_masks = torch.tensor(validation_masks)

#create iterator for the dataset: create DataLoader for training and validation set
batch_size = 4
train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
validation_sampler = SequentialSampler(validation_data)
validation_dataloader = DataLoader(validation_data, sampler=validation_sampler,
                                   batch_size=batch_size)

#build BERT model with additional layer on top for classification
NUM = len(le.classes_)
model = BertForSequenceClassification.from_pretrained("allenai/scibert_scivocab_uncased",
                                                      num_labels=NUM,
                                                      output_attentions=False)
model.cuda()
print("Model has been build successfully")
params = list(model.named_parameters())

optimizer = AdamW(model.parameters(), lr=1e-5, eps=1e-8)
epochs = 4
total_steps = epochs*len(train_dataloader)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0,
                                            num_training_steps=total_steps)

def flat_accuracy(pred, lab):
    """Calculates the accuracy of the classification."""
    pred_flat = np.argmax(pred, axis=1).flatten()
    labels_flat = lab.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

#set seed value for randomization
seed_val = 42
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

#Training of the model
loss_values = []
for epoch_i in range(0, epochs):
    print("Training in epoch ", epoch_i+1)
    total_loss = 0
    model.train()
    for step, batch in enumerate(train_dataloader):
        b_input_ids = batch[0].to('cuda')
        b_input_mask = batch[1].to('cuda')
        b_labels = batch[2].to('cuda')
        outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask,
                        labels=b_labels)
        loss = outputs[0]
        total_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
    avg_train_loss = total_loss / len(train_dataloader)
    loss_values.append(avg_train_loss)
    print("Average training loss: {0:.2f}".format(avg_train_loss))
    print("Validation starting")
    model.eval()
    eval_loss = 0
    eval_accuracy = 0
    nb_eval_steps = 0
    for batch in validation_dataloader:
        batch = tuple(t.to('cuda') for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        with torch.no_grad():
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
            logits = outputs[0]
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            tmp_eval_accuracy = flat_accuracy(logits, label_ids)
            eval_accuracy += tmp_eval_accuracy
            nb_eval_steps += 1
    print("Accuracy: {0:.2f}".format(eval_accuracy/nb_eval_steps))

print("Training completed!")

os.makedirs("finetuning_bert")
model.save_pretrained("finetuning_bert")
tokenizer.save_pretrained("finetuning_bert")
#Evaulation
predictions = []
true_labels = []
model.eval()
for batch in validation_dataloader:
    batch = tuple(t.to('cuda') for t in batch)
    b_input_ids, b_input_mask, b_labels = batch
    with torch.no_grad():
        outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
        logits = outputs[0]
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        predictions.append(logits)
        true_labels.append(label_ids)
flat_predictions = [item for sublist in predictions for item in sublist]
flat_predictions = np.argmax(flat_predictions, axis=1).flatten()
flat_true_labels = [item for sublist in true_labels for item in sublist]
bert_evaluation_scores, bert_cm = evaluation.multilabel_evaluation(
    flat_true_labels, flat_predictions, "SciBERT Embeddings Finetuning")
documentation_file_modelopt = open("classifier_optimization_finetuning_scibert.txt", "w+")
documentation_file_modelopt.write(bert_evaluation_scores)
documentation_file_modelopt.close()
