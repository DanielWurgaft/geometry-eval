# Functions for evaluating geometric concept learning
import os
from openai import RateLimitError, BadRequestError
from tenacity import retry 
from tenacity import wait_fixed, retry_if_exception_type, before_sleep_log
import pandas as pd
import os
import types
from tqdm import tqdm
import base64
import re
import random
import logging
import copy
import csv
from constants import *



# define functions
def create_concept_dictionary(base_path, dataset_section, custom_definitions = None):
    concept_dict = {}
    concepts_path = os.path.join(base_path, dataset_section)

    for index, concept in enumerate(os.listdir(concepts_path)):
        concept_path = os.path.join(concepts_path, concept)
        if os.path.isdir(concept_path):
            namespace = types.SimpleNamespace(train = [], test = [])

            if custom_definitions:
                namespace.definition = custom_definitions[index]
            else:
                namespace.definition = open(os.path.join(concept_path, 'definition.txt'), 'r').readlines()[0].strip()

            namespace.definition_true = open(os.path.join(concept_path, 'concept.txt'), 'r').read()

            # Adding full file paths for train images
            train_path = os.path.join(concept_path, 'train')
            if os.path.exists(train_path):
                namespace.train = [os.path.join(train_path, file) for file in os.listdir(train_path) if "neg" not in file]
                namespace.train_neg = [os.path.join(train_path, file) for file in os.listdir(train_path) if "neg" in file]
            # Adding full file paths for test images
            test_path = os.path.join(concept_path, 'test')
            if os.path.exists(test_path):
                namespace.test = [os.path.join(test_path, file) for file in os.listdir(test_path)]

            concept_dict[concept] = namespace

    return concept_dict


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def assign_trial_variables(prompt_content, name, placeholder = placeholder_name, pseudowords = pseudowords, concept_definition = None):
    deep_copied_dict = copy.deepcopy(prompt_content)
    deep_copied_dict["text"] = deep_copied_dict["text"].replace(placeholder, name)
    if concept_definition:
        # find if any name in pseudowords list is in the concept definition
        for pseudoword in pseudowords:
            if pseudoword in concept_definition:
                concept_definition = concept_definition.replace(pseudoword, name)
        deep_copied_dict["text"] = deep_copied_dict["text"].replace("concept_definition", concept_definition)
    return deep_copied_dict


def image_list_to_prompt_content(image_list, detail="high", model="gpt-4-vision-preview"):
    prompt_content = []
    for img in image_list:
        if "claude" in model:
            image_media_type = "image/png"
            base64_image = encode_image(img)
            prompt_content.append({
                    "type": "image",
                    "source": {
                    "type": "base64",
                    "media_type": image_media_type,
                    "data": base64_image,
                    },
                })
        else:
            base64_image = encode_image(img)
            prompt_content.append({
                    "type": "image_url",
                    "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}",
                    "detail": detail
                    }
                })
    return prompt_content


def unravel_choices(choices, eval_mode, answer_format, model):
    if "claude" in model:
        return unravel_choices_anthropic(choices, eval_mode, answer_format)
    else:
        return unravel_choices_oai(choices, eval_mode, answer_format)

def unravel_choices_oai(choices, eval_mode, answer_format):
    choices_df = pd.json_normalize([choice.__dict__ for choice in choices])
    messages = choices_df["message"]
    choices_df.drop("message", axis=1, inplace=True)
    if "inferDef" not in eval_mode:
        choices_df["final_answer"] = [extract_final_answer(message.content, pre_answer, answer_format=answer_format) for message in messages]
    else:
        choices_df["definition"] = [extract_final_answer(message.content, pre_answer_definition, return_coded=False, answer_format=answer_format) for message in messages]
    choices_df["message_content"] = [message.content for message in messages]
    choices_df["function_call"] = [message.tool_calls for message in messages]
    choices_df["tool_calls"] = [message.function_call for message in messages]
    choices_df["role"] = [message.role for message in messages]
    return choices_df

def unravel_choices_anthropic(choices, eval_mode, answer_format):
    choices_df = pd.json_normalize([choice[0].__dict__ for choice in choices])
    messages = choices_df["text"]
    if "inferDef" not in eval_mode:
        choices_df["final_answer"] = [extract_final_answer(message, pre_answer, answer_format=answer_format) for message in messages]
    else:
        choices_df["definition"] = [extract_final_answer(message, pre_answer_definition, return_coded=False, answer_format=answer_format) for message in messages]
    return choices_df

def extract_final_answer(response, pre_answer, return_coded=True, answer_format="cot"):
    if pre_answer in response:
        if not return_coded:
            return response[response.rfind(pre_answer) + len(pre_answer):]
        else:
            final_answer = re.sub(r'[^A-Za-z0-9]+', '', response.lower()[response.rfind(pre_answer) + len(pre_answer):])
            if final_answer == "yes":
                return 1
            elif final_answer == "no":
                return 0 
            else:
                return -1
    elif answer_format == "binary":
        final_answer = re.sub(r'[^A-Za-z0-9]+', '', response.lower())
        if final_answer == "yes":
            return 1
        elif final_answer == "no":
            return 0
    else:
        return -1
        

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)    
    

@retry(retry=(retry_if_exception_type(RateLimitError) | retry_if_exception_type(BadRequestError)), wait=wait_fixed(60), before_sleep=before_sleep_log(logger, logging.INFO))
def api_call_with_retry(prompt, api_mode="chat", model = "gpt-4-0613", n=1, system_prompt=None, temperature=1,
                        logprobs=None, max_tokens=1, pre_prompt_messages=None,
                        top_logprobs=None):
    if "claude" not in model:
        client = oai_client
    else:
        client = anthropic_client 
    messages = []    
    if system_prompt is not None:
        if client == oai_client:
            messages.append({"role": "system", "content": system_prompt})
    if pre_prompt_messages:
        pre_prompt_messages["content"] += prompt
        messages += [pre_prompt_messages]
    else:
        messages.append({"role": "user", "content": prompt})
    if client == oai_client:
        return client.chat.completions.create(
            model=model,
            messages=messages,
            n=n,
            temperature=temperature,
            max_tokens = max_tokens,
            #logprobs = logprobs,
            #top_logprobs = 2,
        ).__dict__
    else:
        return [dict(client.messages.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens = max_tokens,
        )) for _ in range(n)]



def eval_n_generations_llm(out_dir,  concept_dict, eval_mode = "classify_fromImg", answer_format = "cot", negative_examples = False, system_prompt=None, n=10, model="gpt-4-vision-preview", temperature=1,
                                      api_mode = "chat", max_tokens = 1, logprobs = None,
                                        top_logprobs = None, use_custom_definitions = False, control=False):
    # determine experiment condition
    if eval_mode == "inferDef":
        condition = answer_format 
    else:
        condition = eval_mode+"_"+answer_format
    if negative_examples:
        condition += "_negEx"
    if use_custom_definitions:
        condition += "_inferredDef"
    if control:
        condition += "_control"
    condition += "_"+model

    with open(os.path.join(out_dir, "responses_"+condition+".csv"), 'w', newline='', encoding="utf-8") as csv_responses:
        writer_responses = csv.writer(csv_responses)
        with open(os.path.join(out_dir, "choices_"+condition+".csv"), 'w', newline='', encoding="utf-8") as csv_choices:
            writer_choices = csv.writer(csv_choices)
            for concept in tqdm(concept_dict):
                for test_img in tqdm(concept_dict[concept].test,leave=False):
                    pseudoword = pseudowords[random.randint(0, len(pseudowords)-1)]
                
                    # determine prompt content
                    concept_definition = None
                    if "inferDef" in eval_mode:
                        if negative_examples:
                            instruction = instructions['inferDef']['pos and neg']
                            question = questions['inferDef']['pos and neg']
                        else:
                            instruction = instructions['inferDef']['pos']
                            question = questions['inferDef']['pos']
                        prompt = f"""{question}\n{answer_formats[answer_format]}\n{encourage_to_answer}"""
                    else:
                        if negative_examples:
                            instruction = instructions[eval_mode]['pos and neg']
                        elif "classify_fromDef" in eval_mode:
                            instruction = instructions[eval_mode]
                        else:
                            instruction = instructions[eval_mode]['pos']
                        if "Def" in eval_mode:
                            concept_definition = concept_dict[concept].definition
                            prompt = f"""{definition}\n{questions[eval_mode]}\n{answer_formats[answer_format]}\n{encourage_to_answer}"""
                        else:
                            prompt = f"""{questions[eval_mode]}\n{answer_formats[answer_format]}\n{encourage_to_answer}"""
                    prompt_dict = {"type": "text", "text": prompt}
                    prompt_content = [assign_trial_variables(prompt_dict, name=pseudoword, concept_definition=concept_definition)] + image_list_to_prompt_content([test_img], model=model)
                    
                    # determine pre_prompt messages
                    if eval_mode != "classify_fromDef":
                        pre_prompt_messages = {"role": "user", "content": [{"type": "text", "text": instruction}] + [assign_trial_variables(initial_prompt_content, name=pseudoword)] +\
                        image_list_to_prompt_content(concept_dict[concept].train, model=model)}
                        if negative_examples:
                            pre_prompt_messages["content"] += [assign_trial_variables(initial_prompt_content_negative_ex, name=pseudoword)] +\
                                                                image_list_to_prompt_content(concept_dict[concept].train_neg, model=model)
                    else:
                        pre_prompt_messages = None
                    
                    response =  api_call_with_retry(prompt=prompt_content, pre_prompt_messages=pre_prompt_messages, api_mode=api_mode, model=model, n=n,
                                    system_prompt=system_prompt, temperature=temperature,
                                    logprobs=logprobs, max_tokens=max_tokens)
                    # convert response to json
                    response_row = pd.json_normalize(response)
                    # write response_row to csv file
                    if os.stat(os.path.join(out_dir, "responses_"+condition+".csv")).st_size == 0:
                        writer_responses.writerow(response_row.columns)

                    for index, row in response_row.iterrows():
                        writer_responses.writerow(row)

                    #convert choices to json 
                    if "claude" in model:
                        response_row.rename(columns={"content":"choices"}, inplace=True)
                        choices_rows = unravel_choices(response_row["choices"], eval_mode, answer_format, model)
                    else:
                        choices_rows = unravel_choices(response["choices"], eval_mode, answer_format, model)

                    # add concept, test target, and prompt columns to choices
                    choices_rows["eval_mode"] = [eval_mode]*len(choices_rows)
                    choices_rows["answer_format"] = [answer_format]*len(choices_rows)
                    choices_rows["negative_examples"] = [int(negative_examples)]*len(choices_rows)
                    choices_rows["inferred_definition_used"] = [int(use_custom_definitions)]*len(choices_rows)
                    choices_rows["control"] = [int(control)]*len(choices_rows)
                    choices_rows["concept"] = [concept]*len(choices_rows) 
                    choices_rows["model"] = [model]*len(choices_rows)
                    choices_rows["pre_prompt_messages"] = [pre_prompt_messages]*len(choices_rows)
                    choices_rows["prompt"] = [prompt_content]*len(choices_rows)
                    if eval_mode != "inferDef":
                        choices_rows["test_image"] = [test_img]*len(choices_rows)
                        choices_rows["test_cond"] = ["far" if "far_" in test else "close" if "close_" in test else "in" for test in choices_rows["test_image"]]
                        choices_rows["solution"] = [0 if "out_" or "neg_" in test else 1 for test in choices_rows["test_image"]] #ERROR: TEST WHY!!!!!!!!
                        choices_rows["correct"] = [1 if choices_rows["solution"][i] == choices_rows["final_answer"][i] else 0 for i in range(len(choices_rows["solution"]))]

                    # write choices_rows to csv file
                    # Write each row
                    if os.stat(os.path.join(out_dir, "choices_"+condition+".csv")).st_size == 0:
                        writer_choices.writerow(choices_rows.columns)
                    for index, row in choices_rows.iterrows():
                        writer_choices.writerow(row)

                    if eval_mode == "inferDef":
                    # run evaluation for each concept only once if the task is to induce abstraction
                        break