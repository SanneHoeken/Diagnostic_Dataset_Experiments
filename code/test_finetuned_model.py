import torch, json
from tqdm import tqdm
import numpy as np
from pytorch_transformers import BertForMultipleChoice, BertTokenizer, RobertaConfig, RobertaTokenizer
from roberta_mc import RobertaForMultipleChoice
from sklearn.metrics import accuracy_score
from finetune_model import collate_data

def test(model, test_data, device):
    model.eval()
    preds = []
    labels = []
    for batch in tqdm(test_data):
        batch = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            outputs = model(batch[0], token_type_ids=batch[1], attention_mask=batch[2])
        logits = outputs[0].detach().cpu().numpy()
        gold = batch[3].to('cpu').numpy()
        preds.extend(np.argmax(logits, axis=1).tolist())
        labels.extend(gold.tolist())
    
    print('Accuracy: ', accuracy_score(labels, preds))

    return preds


def main(dataset_path, output_filepath, model_path, batch_size, max_seq_length):
    """
    DOCSTRING HERE
    """

    with open(dataset_path+'/test.jsonl', "r") as infile:
        test_examples = [json.loads(line.strip('\n')) for line in infile.readlines()]

    if 'bert' in model_path:
        model = BertForMultipleChoice.from_pretrained(model_path)
        tokenizer = BertTokenizer.from_pretrained(model_path)
    elif 'roberta' in model_path:
        config = RobertaConfig.from_pretrained(model_path, num_labels=1)
        model = RobertaForMultipleChoice.from_pretrained(model_path, config=config)
        tokenizer = RobertaTokenizer.from_pretrained(model_path)

    test_data = collate_data(test_examples, tokenizer, max_seq_length, batch_size, is_test=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    preds = test(model, test_data, device) 
    predictions = ['1' if p == 0 else '2' for p in preds]

    for example, prediction in zip(test_examples, predictions):
        example[f'prediction'] = prediction

    # write output to jsonl-file
    with open(output_filepath, 'w') as outfile:
        for example in test_examples:
            outfile.write(json.dumps(example) + '\n')


if __name__ == "__main__":

    dataset_path = '../data/datasets/diagnostic_dataset_1'
    output_filepath = '../output/bert-ft-2A-testpreds1.jsonl'
    model_path = '../models/bert-ft-2A'
    batch_size = 64
    max_seq_length = 50
    
    main(dataset_path, output_filepath, model_path, batch_size, max_seq_length)