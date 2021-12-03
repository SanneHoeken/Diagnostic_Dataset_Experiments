import torch, json
from tqdm import tqdm
import numpy as np
from pytorch_transformers import BertForMultipleChoice, BertTokenizer, RobertaConfig, RobertaTokenizer
from roberta_mc import RobertaForMultipleChoice
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score


def collate_data(examples, tokenizer, max_seq_length, batch_size, is_test=False):
    """
    Encodes every multiple choice option for every example as:
    [CLS] [encoded context] [SEP] [encoded option] [SEP] [padding]
    """
    input_ids = []
    input_masks = []
    segment_ids = []
    label_ids = []

    for example in examples:
        
        sentence = example['sentence']
        idx = sentence.index('_')
        context_tokens = tokenizer.tokenize(sentence[:idx])
        sep = tokenizer.sep_token
        pad = tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0]
        cls_token = tokenizer.cls_token
        
        option_input_ids = []
        option_input_masks = []
        option_segment_ids = []

        for option in [example['option1'], example['option2']]:
            
            option_tokens = tokenizer.tokenize(option + sentence[idx+1:])
            tokens = [cls_token] + context_tokens + [sep] + option_tokens + [sep]
            padding_length = max_seq_length - len(tokens)
            
            option_input_ids.append(tokenizer.convert_tokens_to_ids(tokens) + [pad] * padding_length)
            option_input_masks.append([1] * len(tokens) + [0] * padding_length)
            option_segment_ids.append([0] * (len(context_tokens) + 2) + [1] * (len(option_tokens) + 1) + [0] * padding_length)
            
        input_ids.append(option_input_ids)
        input_masks.append(option_input_masks)
        segment_ids.append(option_segment_ids)
        label_ids.append(0 if example['answer'] == '1' else 1)

    input_ids = torch.tensor(input_ids, dtype=torch.long)
    input_masks = torch.tensor(input_masks, dtype=torch.long)
    segment_ids = torch.tensor(segment_ids, dtype=torch.long)
    label_ids = torch.tensor(label_ids, dtype=torch.long)

    dataset = TensorDataset(input_ids, input_masks, segment_ids, label_ids)
    
    if is_test:
        sampler = SequentialSampler(dataset)
        dataloader = DataLoader(dataset, batch_size=batch_size)
    else:
        sampler = RandomSampler(dataset)
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)
    
    return dataloader


def train_model(model, model_path, optimizer, scheduler, train_data, val_data, epochs, device):
    
    highest_acc = None

    for epoch in range(epochs):
        print(f"Epoch {epoch+1} out of {epochs}...")

        model, avg_train_loss = train(model, optimizer, scheduler, train_data, device)
        print(f"Average training loss: {avg_train_loss}")

        if epochs < 2:
            model.save_pretrained(model_path)
        
        else:
            accuracy = evaluate(model, val_data, device)
            print(f"Accuracy: {accuracy}")
            
            if highest_acc == None or accuracy > highest_acc:
                highest_acc = accuracy
                model.save_pretrained(model_path)



def train(model, optimizer, scheduler, train_data, device):
    model.train()
    total_loss = 0

    print("Train model...")
    for batch in tqdm(train_data):
        batch = tuple(t.to(device) for t in batch)
        model.zero_grad()
        outputs = model(batch[0], token_type_ids=batch[1], attention_mask=batch[2], labels=batch[3])
        loss = outputs[0]
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        scheduler.step()

    avg_train_loss = total_loss / len(train_data)
    return model, avg_train_loss
            
        
def evaluate(model, val_data, device):
    model.eval()
    preds = []
    labels = []
    print("Evaluate model...")
    for batch in tqdm(val_data):
        batch = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            outputs = model(batch[0], token_type_ids=batch[1], attention_mask=batch[2])
        logits = outputs[0].detach().cpu().numpy()
        gold = batch[3].to('cpu').numpy()
        preds.extend(np.argmax(logits, axis=1).tolist())
        labels.extend(gold.tolist())
    accuracy = accuracy_score(labels, preds)

    return accuracy


def main(dataset_filepath, 
        model_path, 
        pretrained_model, 
        batch_size, 
        max_seq_length, 
        learning_rate, 
        epochs):
    """
    DOCSTRING HERE
    """
    
    if pretrained_model.startswith('bert'):
        model = BertForMultipleChoice.from_pretrained(pretrained_model)
        tokenizer = BertTokenizer.from_pretrained(pretrained_model)
    elif pretrained_model.startswith('roberta'):
        config = RobertaConfig.from_pretrained(pretrained_model, num_labels=1)
        model = RobertaForMultipleChoice.from_pretrained(pretrained_model, config=config)
        tokenizer = RobertaTokenizer.from_pretrained(pretrained_model)

    tokenizer.save_vocabulary(model_path)

    with open(dataset_filepath+'/train.jsonl', "r") as infile:
        train_examples = [json.loads(line.strip('\n')) for line in infile.readlines()]
    train_data = collate_data(train_examples, tokenizer, max_seq_length, batch_size)
    val_data = None

    if epochs > 1:
        with open(dataset_filepath+'/val.jsonl', "r") as infile:
            val_examples = [json.loads(line.strip('\n')) for line in infile.readlines()]
        val_data = collate_data(val_examples, tokenizer, max_seq_length, batch_size, is_test=True)
    
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(train_data) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_model(model, model_path, optimizer, scheduler, train_data, val_data, epochs, device)


if __name__ == "__main__":

    dataset_filepath = '../data/datasets/diagnostic_dataset_1'
    model_path = '../models/bert-ft-1'
    pretrained_model = 'bert-base-uncased'
    batch_size = 64
    max_seq_length = 50
    learning_rate = 5e-5
    epochs = 1
    
    main(dataset_filepath, 
        model_path, 
        pretrained_model, 
        batch_size, 
        max_seq_length, 
        learning_rate, 
        epochs)