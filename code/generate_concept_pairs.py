import glob, json
from os import path
import gensim.downloader as api
from sklearn.model_selection import train_test_split


def get_similarity(concept1, concept2, embedding_model):
    """
    Returns cosine similarity between two word vectors
    Returns 0 if words are not in the embedding model
    """
    if concept1 in embedding_model.vocab and concept2 in embedding_model.vocab:
        return float(embedding_model.similarity(concept1, concept2))
    
    return 0


def main(semantic_property_dir, output_file):

    positive_concepts = set()
    negative_concepts = set()

    concepts_dic = dict()
    
    # iterate over property json-files in semantic property dataset
    for filepath in glob.iglob(f'{semantic_property_dir}/*.json'):
        
        # open file and load into dictionary
        with open(filepath, 'r') as infile:
            prop_dic = json.load(infile)
  
        prop = path.basename(filepath).replace('.json', '')
        concepts_dic[prop] = {'pconcepts': [], 'nconcepts': []}
        
        # iterate over all combinations of a positive and negative example of the property
        for concept, annotations in prop_dic.items():
            if annotations['ml_label'] in ["all", "all-some", "some", "few-some"]:
                positive_concepts.add(concept)
                concepts_dic[prop]['pconcepts'].append(concept)
            elif annotations['ml_label'] == "few":
                negative_concepts.add(concept)
                concepts_dic[prop]['nconcepts'].append(concept)
        
    train_positive_concepts, test_positive_concepts = train_test_split(list(positive_concepts), test_size=0.2)
    train_negative_concepts, test_negative_concepts = train_test_split(list(negative_concepts), test_size=0.2)

    # load google news embedding model
    embedding_model = api.load('word2vec-google-news-300')

    concept_pairs = dict()   

    for prop in concepts_dic:
        concept_pairs[prop] = {'train': [], 'test': []}

        for pos in concepts_dic[prop]['pconcepts']:
            for neg in concepts_dic[prop]['pconcepts']:
                if pos in train_positive_concepts and neg in train_negative_concepts:
                    sim = get_similarity(pos, neg, embedding_model)
                    concept_pairs[prop]['train'].append((pos, neg, sim))
                elif pos in test_positive_concepts and neg in test_negative_concepts:
                    sim = get_similarity(pos, neg, embedding_model)
                    concept_pairs[prop]['test'].append((pos, neg, sim))
                
        
    # write output to jsonl-file
    with open(output_file, 'w') as outfile:
        json.dump(concept_pairs, outfile)


if __name__ == "__main__":
    semantic_property_dir = '../data/semantic_property_data'
    output_file = '../data/concept_pairs.json'
    main(semantic_property_dir, output_file)