import torch, json
from tqdm import tqdm
import numpy as np
from pytorch_transformers import BertForMultipleChoice, BertTokenizer, RobertaConfig, RobertaTokenizer
from roberta_mc import RobertaForMultipleChoice
from sklearn.metrics import accuracy_score
from train_model import collate_data

## for probabilities
import token_probability as tp
from transformers import BertTokenizer, BertForMaskedLM
from transformers import RobertaTokenizer, RobertaForMaskedLM

def test(model, val_data, device):
    model.eval()
    preds = []
    labels = []
    for batch in tqdm(val_data):
        batch = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            outputs = model(batch[0], token_type_ids=batch[1], attention_mask=batch[2])
        logits = outputs[0].detach().cpu().numpy()
        gold = batch[3].to('cpu').numpy()
        preds.extend(np.argmax(logits, axis=1).tolist())
        labels.extend(gold.tolist())
    
    print('Accuracy: ', accuracy_score(labels, preds))

    return preds

def test_pretrained(model, tokenizer, val_data):
    preds = []
    labels = []
    print(f'evaluating on {len(val_data)} test instances')
    for i in val_data:
        sentence = i['sentence']
        option1 = i['option1']
        option2 = i['option2']
        labels.append(i['answer'])
    
        prop1 = tp.get_candidate_prob(model, tokenizer, sentence, option1)
        prop2 = tp.get_candidate_prob(model, tokenizer, sentence, option2)
        pred = '0'
        if prop1 > prop2:
            pred = '1'
        elif prop2 > prop1:
            pred = '2'
        preds.append(pred)

    print('Accuracy: ', accuracy_score(labels, preds))

    return preds


def get_batch_ints(instances, batch_size):
    
    batch_ints_total = []
    n_b = int(len(instances)/batch_size)
    total = len(instances)
    batch_ints = []
    start = 0
    for n in range(n_b):
        end = start+batch_size
        batch_ints.append((start, end))
        start = end 
    for i, (start, end) in enumerate(batch_ints):
        batch = []
        if i == len(batch_ints)-1:
            for n in range(start, total):
                batch.append(n)
        else:
            for n in range(start, end):
                batch.append(n)
        batch_ints_total.append(batch)
    return batch_ints_total


def main(dataset_path, output_filepath, model_path, model_type='pt'):
    """
    DOCSTRING HERE
    """
    if 'BLACKBOX' in dataset_path:
        dataset_path = dataset_path
    else:
        dataset_path+'/test.jsonl'
    with open(dataset_path, "r") as infile:
        test_examples = [json.loads(line.strip('\n')) for line in infile.readlines()]
        
      

    if 'roberta' in model_path:
        tokenizer = RobertaTokenizer.from_pretrained(model_path)
        pt_model = RobertaForMaskedLM.from_pretrained(model_path)
    elif 'bert' in model_path:
        tokenizer = BertTokenizer.from_pretrained(model_path)
        pt_model = BertForMaskedLM.from_pretrained(model_path)
        # create batches of 500
    batch_size = 500
    batch_ints = get_batch_ints(test_examples, batch_size)
    for n, batch in enumerate(batch_ints):
        print(f'Testing batch {n}')
        print(f'Batches left: {len(batch_ints)-n}')
        test_examples_batch = [test_examples[i] for i in batch]
        predictions = test_pretrained(pt_model, tokenizer, test_examples_batch)
        for example, prediction in zip(test_examples_batch, predictions):
            example[f'prediction'] = prediction
        if n == 0:
            mode = 'w'
        else:
            mode = 'a'
        with open(output_filepath, mode) as outfile:
            for example in test_examples_batch:
                outfile.write(json.dumps(example) + '\n')
                        
                


if __name__ == "__main__":
# Sanne
#     dataset_path = '../output/diagnostic_dataset_2A'
#     output_filepath = '../output/bert-ft-wino-testpreds2A.jsonl'
#     model_path = '../models/bert-ft-wino'
#     batch_size = 64
#     max_seq_length = 50

    dataset_path =  '../data/datasets/DATASET_BLACKBOX_SUBMISSION/challenge_dataset_1.1.jsonl'
    output_filepath = '../output/pt-bert-large-uncased-wino-blackbox.jsonl'
    model_path = 'bert-large-uncased'
  
    
    main(dataset_path, output_filepath, model_path, batch_size, max_seq_length, model_type='pt')