import os
from tqdm import tqdm
from geoclidean_env_euclid import generate_concept

def make_negative_examples(base_path, dataset_section, target_concept = None, num_examples=1):
    concepts_path = os.path.join(base_path, dataset_section)

    for index, concept in tqdm(enumerate(os.listdir(concepts_path))):
        if target_concept is not None and concept != target_concept:
            continue
        concept_path = os.path.join(concepts_path, concept)
        if os.path.isdir(concept_path):
            out_close_def = open(os.path.join(concept_path, 'close_concept.txt'), 'r').read().split("\n")[:-1]
            out_close_def = [line[1:-2] for line in out_close_def]
            
            out_far_def = open(os.path.join(concept_path, 'far_concept.txt'), 'r').read().split("\n")[:-1]
            out_far_def = [line[1:-2] for line in out_far_def]
            for i in range(num_examples):
                generate_concept(out_close_def, path=os.path.join(concept_path,"train\\neg_close_"+str(i)+".png"))
                generate_concept(out_far_def, path=os.path.join(concept_path,"train\\neg_far_"+str(i)+".png"))

if __name__ == "__main__":
    base_path = "../../data/geoclidean"
    make_negative_examples(base_path, "elements")
    make_negative_examples(base_path, "constraints")
