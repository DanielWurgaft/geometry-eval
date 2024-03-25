## run geoclidean evaluation

# import functions
from utils import create_concept_dictionary, eval_n_generations_llm
from constants import *
import pandas as pd

def run(model_eval_dir, eval_mode, answer_format, negative_examples, test=False, test_dict=None, model="gpt-4-1106-vision-preview",
        path_inferred_definitions_constraints=None, path_inferred_definitions_elements=None, use_custom_definitions=False, control=False):
    if test:
        # test eval with one datapoint
        out_dir = os.path.join(model_eval_dir, "test")
        eval_n_generations_llm(model=model, out_dir=out_dir, eval_mode=eval_mode, answer_format=answer_format, negative_examples=negative_examples,
                       concept_dict = test_dict, n = 1, max_tokens=1000, temperature=0)
    else:
        if eval_mode not in questions.keys() or answer_format not in answer_formats.keys():
            raise ValueError("Invalid eval_mode or answer_format")
        
        # make out dir paths for elements and constraints
        out_dir_elements = os.path.join(model_eval_dir, "elements")
        out_dir_constraints = os.path.join(model_eval_dir, "constraints")

        # make concept dictionaries for evaluation
        if use_custom_definitions:
            custom_definitions_elements = list(pd.read_csv(path_inferred_definitions_elements)["definition"])
            elements_dict = create_concept_dictionary(ROOT_DIR + "data\\geoclidean", "elements", custom_definitions=custom_definitions_elements)                
        
            custom_definitions_constraints = list(pd.read_csv(path_inferred_definitions_constraints)["definition"])
            constraints_dict = create_concept_dictionary(ROOT_DIR + "data\\geoclidean", "constraints", custom_definitions=custom_definitions_constraints)
        else:
            elements_dict = create_concept_dictionary(ROOT_DIR + "data\\geoclidean", "elements")
            constraints_dict = create_concept_dictionary(ROOT_DIR + "data\\geoclidean", "constraints")
        if control:
            if negative_examples:
                for concept in elements_dict.keys():
                    elements_dict[concept].test = elements_dict[concept].train_neg
                for concept in constraints_dict.keys():
                    constraints_dict[concept].test = constraints_dict[concept].train_neg
            else:
                for concept in elements_dict.keys():
                    elements_dict[concept].test = elements_dict[concept].train
                for concept in constraints_dict.keys():
                    constraints_dict[concept].test = constraints_dict[concept].train
        
        # run eval
        if eval_mode == "inferDef":
             eval_n_generations_llm(model=model, out_dir=out_dir_elements, eval_mode=eval_mode, answer_format=answer_format, negative_examples=negative_examples,
                       concept_dict = elements_dict, n = 1, max_tokens=1000, temperature=0)
             eval_n_generations_llm(model=model, out_dir=out_dir_constraints, eval_mode=eval_mode, answer_format=answer_format, negative_examples=negative_examples,
                       concept_dict = constraints_dict, n = 1, max_tokens=1000, temperature=0)
        elif answer_format == "binary":
            if "claude" not in model:
                n = 5
                temperature = 1
            else:
                n =1
                temperature = 0
            eval_n_generations_llm(model=model, out_dir=out_dir_elements, eval_mode=eval_mode, answer_format=answer_format, negative_examples=negative_examples,
                        concept_dict = elements_dict, n = n, max_tokens=5, temperature=temperature, use_custom_definitions=use_custom_definitions, control=control)
            eval_n_generations_llm(model=model, out_dir=out_dir_constraints, eval_mode=eval_mode, answer_format=answer_format, negative_examples=negative_examples,
                        concept_dict = constraints_dict, n = n, max_tokens=5, temperature=temperature, use_custom_definitions=use_custom_definitions, control=control)         
        else:
             eval_n_generations_llm(model=model, out_dir=out_dir_elements, eval_mode=eval_mode, answer_format=answer_format, negative_examples=negative_examples,
                       concept_dict = elements_dict, n = 5, max_tokens=1000, temperature=1, use_custom_definitions=use_custom_definitions, control=control)
             eval_n_generations_llm(model=model, out_dir=out_dir_constraints, eval_mode=eval_mode, answer_format=answer_format, negative_examples=negative_examples,
                       concept_dict = constraints_dict, n = 5, max_tokens=1000, temperature=1, use_custom_definitions=use_custom_definitions, control=control)

if __name__ == "__main__":
    model_eval_dir = ROOT_DIR + "data\\model_eval\\"
    
    test_dict = {"test_concept": create_concept_dictionary(ROOT_DIR + "data\\geoclidean", "elements")["concept_ang_bisector"]}
    test_dict["test_concept"].test = test_dict["test_concept"].test[:1]+test_dict["test_concept"].test[-1:]
    # run test
    # run(model="claude-3-opus-20240229",
    #     model_eval_dir=model_eval_dir, 
    #     eval_mode="classify_fromImg", 
    #     answer_format="binary", 
    #     test=True, test_dict=test_dict, negative_examples=False)

    # run standard eval
    ## w/o negative examples
    ### binary
    run(model="claude-3-opus-20240229", model_eval_dir=model_eval_dir, 
    eval_mode="classify_fromImg", 
    answer_format="binary", negative_examples=False)
    # # ### cot
    # run(model_eval_dir=model_eval_dir, 
    # eval_mode="classify_fromImg", 
    # answer_format="cot", negative_examples=False)
    # # ## w negative examples
    # # ### binary
    # run(model_eval_dir=model_eval_dir, 
    # eval_mode="classify_fromImg", 
    # answer_format="binary", negative_examples=True)
    # # # ### cot
    # run(model_eval_dir=model_eval_dir, 
    # eval_mode="classify_fromImg", 
    # answer_format="cot", negative_examples=True)

    # run control
    ## binary
    ### w/o negative examples
    # run(model_eval_dir=model_eval_dir, 
    # eval_mode="classify_fromImg", 
    # answer_format="binary", negative_examples=False, control=True)
    # # ### w negative examples
    # run(model_eval_dir=model_eval_dir, 
    # eval_mode="classify_fromImg", 
    # answer_format="binary", negative_examples=True, control=True)


    
    # # infer definitions
    # ## w/o negative examples
    # run(model_eval_dir=model_eval_dir,
    # eval_mode="inferDef",
    # answer_format="inferDef_direct", negative_examples=False)
    # ## w negative examples
    # run(model_eval_dir=model_eval_dir,
    # eval_mode="inferDef",
    # answer_format="inferDef", negative_examples=True)

    # classify from definition
    ## inferred definition (using definition inferred from pos+neg examples for both)
    # path_inferred_definitions_elements = model_eval_dir+"\\elements\\choices_inferDef_negEx.csv"
    # path_inferred_definitions_constraints = model_eval_dir+"\\constraints\\choices_inferDef_negEx.csv"
    # ### w negative examples
    # #### binary
    # run(model_eval_dir=model_eval_dir,
    # eval_mode="classify_fromImg&Def", use_custom_definitions=True,
    # answer_format="binary", negative_examples=True,
    # path_inferred_definitions_elements=path_inferred_definitions_elements,
    # path_inferred_definitions_constraints=path_inferred_definitions_constraints)
    # #### cot
    # run(model_eval_dir=model_eval_dir,
    # eval_mode="classify_fromImg&Def", use_custom_definitions=True,
    # answer_format="cot", negative_examples=True,
    # path_inferred_definitions_elements=path_inferred_definitions_elements,
    # path_inferred_definitions_constraints=path_inferred_definitions_constraints)
    # ### w/o negative examples
    # ### only definition
    # #### binary
    # run(model_eval_dir=model_eval_dir,
    # eval_mode="classify_fromDef", use_custom_definitions=True,
    # answer_format="binary", negative_examples=False,
    # path_inferred_definitions_elements=path_inferred_definitions_elements,
    # path_inferred_definitions_constraints=path_inferred_definitions_constraints)
    # #### cot
    # run(model_eval_dir=model_eval_dir,
    # eval_mode="classify_fromDef", use_custom_definitions=True,
    # answer_format="cot", negative_examples=False,
    # path_inferred_definitions_elements=path_inferred_definitions_elements,
    # path_inferred_definitions_constraints=path_inferred_definitions_constraints)

    # correct definition 
    ## w negative examples
    ### binary
    # run(model_eval_dir=model_eval_dir,
    # eval_mode="classify_fromImg&Def",
    # answer_format="binary", negative_examples=True,
    # use_custom_definitions=False)
    # #### cot 
    # run(model_eval_dir=model_eval_dir,
    # eval_mode="classify_fromImg&Def",
    # answer_format="cot", negative_examples=True,
    # use_custom_definitions=False)
    ### w/o negative examples
    ### only definition
    #### binary
    # run(model_eval_dir=model_eval_dir,
    # eval_mode="classify_fromDef",
    # answer_format="binary", negative_examples=False,
    # use_custom_definitions=False)
    #### cot
    # run(model_eval_dir=model_eval_dir,
    # eval_mode="classify_fromDef",
    # answer_format="cot", negative_examples=False,
    # use_custom_definitions=False)
    

    # run standard eval
    ## w/o negative examples
    ### binary
    # run(model_eval_dir=model_eval_dir, 
    # eval_mode="classify_fromImg", 
    # answer_format="binary", negative_examples=False)
    # # ### cot
    # run(model_eval_dir=model_eval_dir, 
    # eval_mode="classify_fromImg", 
    # answer_format="cot", negative_examples=False)
    # # ## w negative examples
    # # ### binary
    # run(model_eval_dir=model_eval_dir, 
    # eval_mode="classify_fromImg", 
    # answer_format="binary", negative_examples=True)
    # # # ### cot
    # run(model_eval_dir=model_eval_dir, 
    # eval_mode="classify_fromImg", 
    # answer_format="cot", negative_examples=True)