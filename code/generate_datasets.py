import json, random

def get_prop_phrase(prop):
    """
    Returns a grammatically correct phrase that 
    attributes the given property to a subject 
    (that has yet to be completed)
    """
    prop_phrase = f" is {prop.replace('_', ' ')}"    
    if prop in ['fly', 'lay_eggs', 'roll', 'swim']:
        prop_phrase = f" can {prop.replace('_', ' ')}"
    elif prop in ['wheels', 'wings']:
        prop_phrase = f" has {prop}"
    
    return prop_phrase


def generate_example(templates, properties, prop_dic, split, sim_lower, sim_upper):

    # choose random template
    template_id = random.randint(0, len(templates)-1)
    template = templates[template_id]

    # choose random property
    prop = properties[random.randint(0, len(properties)-1)]
    prop_phrase = get_prop_phrase(prop)

    # choose random concept pair
    concept_pair = prop_dic[prop][split][random.randint(0, len(prop_dic[prop][split])-1)]
    if sim_lower:
        while not concept_pair[2] >= 0.5:
            concept_pair = prop_dic[prop][split][random.randint(0, len(prop_dic[prop][split])-1)]
    elif sim_upper:
        while not concept_pair[2] < 0.5:
            concept_pair = prop_dic[prop][split][random.randint(0, len(prop_dic[prop][split])-1)]

    concept1 = concept_pair[0]
    concept2 = concept_pair[1]
    
    # generate an example
    example = dict()

    # fill gaps in sentence
    example['sentence'] = template.replace('[POS]', concept1).replace('[NEG]', concept2).replace('[PROP]', prop_phrase)
    
    # assign candidate options and answer randomly
    example['option1'] = random.choice([concept1, concept2])
    example['option2'] = concept2 if example['option1'] == concept1 else concept1
    example['answer'] = '1' if example['option1'] == concept1 else '2'
    
    # add information 
    example.update({'property': prop, 'pconcept': concept1, 'nconcept': concept2, 'template': template})
    
    return example


def create_dataset(output_filepath, concept_pairs_file, templates_file, dataset_sizes, 
                    sim_lower = None, sim_upper = None, exclude_props = []):
    """
    Creates a Challenge Dataset that includes examples for a masked word prediction task
    where common sense knowledge of the semantic properties of the candidate concepts should
    be required to solve the task. A Semantic Property Dataset is used to create this dataset. 
    Examples for this Challenge Dataset are generated using templates in such a way that the two 
    candidates in the example are a pair of which one is a positive concept with respect to a 
    property in the property dataset, and the other is a negative concept.
    """

    # open and read file with templates
    with open(templates_file, 'r') as infile:
        templates = [line.replace('\n', '') for line in infile.readlines()]

    # open concept pairs file and load into dictionary
    with open(concept_pairs_file, 'r') as infile:
        concept_pairs = json.load(infile)
    props = list(concept_pairs.keys())
    properties = [p for p in props if p not in exclude_props]
    
    for split, size in dataset_sizes.items():
        output = []
        for _ in range(size):
            example = generate_example(templates, properties, concept_pairs, split, sim_lower, sim_upper)            
            output.append(example)
            
        # write output to jsonl-file
        with open(output_filepath+f'/{split}.jsonl', 'w') as outfile:
            for example in output:
                outfile.write(json.dumps(example) + '\n')


if __name__ == "__main__":

    output_filepath = '../data/datasets/diagnostic_dataset_4B'
    concept_pairs_file = '../data/concept_pairs.json'
    templates_file = '../data/templates/templates1.txt'
    dataset_sizes = {'train': 24000, 'test':6000}
    
    create_dataset(output_filepath, concept_pairs_file, templates_file, dataset_sizes, 
    exclude_props=['black', 'female', 'dangerous', 'lay_eggs', 'made_of_wood', 'warm', 
    'fly', 'roll', 'cold', 'red', 'yellow', 'round', 'wheels', 'juicy'])