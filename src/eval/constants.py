# Constants for geometric concept learning experiments
from openai import OpenAI
from  anthropic import Anthropic
import os
from dotenv import load_dotenv
load_dotenv()

ROOT_DIR = "..\\..\\"
oai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


pre_answer = "Final answer:"
pre_answer_definition = "Definition:"

pseudowords = ['zim', 'frolp', 'bliv', 'jeex',
              'muxl', 'wunf', 'yulf', 'dax', 
              'zup', 'fep','jeg', 'vop']

placeholder_name = 'pseudoword'

instructions_infer_from_definition = "###Instructions###: In the following trial, you will view the definition of a concept and a test image. Your task is to determine whether the test image belongs to the concept represented by the definition."
instructions_infer_from_images = "###Instructions###: In the following trial, you will view a series of example images for a concept and a test image. Your task is to determine whether the test image belongs to the concept represented by the example images."
instructions_infer_from_images_w_neg = "###Instructions###: In the following trial, you will view a series of example images for a concept, example images which do not belong to the concept, and a test image. Your task is to determine whether the test image belongs to the concept represented by the example images."
instructions_infer_from_images_and_definition = "###Instructions###: In the following trial, you will view a series of example images for a concept and the definition of the concept, as well as a test image. Your task is to determine whether the test image belongs to the concept represented by the example images and the definition."
instructions_infer_from_images_and_definition_w_neg = "###Instructions###: In the following trial, you will view a series of example images for a concept, example images which do not belong to the concept, the definition of the concept, and a test image. Your task is to determine whether the test image belongs to the concept represented by the example images and the definition."
instructions_infer_definition_from_images = "###Instructions###: In the following trial, you will view a series of example images for a concept. Your task is to provide a definition of the concept that is consistent with all images of the concept shown."
instructions_infer_definition_from_images_w_neg = "###Instructions###: In the following trial, you will view a series of example images for a concept, as well as example images which do not belong to the concept. Your task is to provide a definition of the concept that is consistent with all images of the concept shown and excludes all images that do not belong to the concept."
instructions = {"classify_fromImg": {"pos":instructions_infer_from_images, "pos and neg": instructions_infer_from_images_w_neg},
                "classify_fromImg&Def": {"pos":instructions_infer_from_images_and_definition, "pos and neg": instructions_infer_from_images_and_definition_w_neg},
                "classify_fromDef":  instructions_infer_from_definition,
                "inferDef": {"pos":instructions_infer_definition_from_images, "pos and neg":instructions_infer_definition_from_images_w_neg}}

Question_infer_from_images = '###Question###: Based on the images provided, is the following image a "pseudoword"?'
Question_infer_from_definition = '###Question###: Based on the definition provided, is the following image a "pseudoword"?'
Question_infer_from_images_and_definition = '###Question###: Based on the definition and images provided, is the following image a "pseudoword"?'
Question_infer_definition_from_images = '###Question###: Based on the images provided, what is the definition of a "pseudoword"? give a detailed definition of a "pseudoword" that is consistent with all images of a "pseudoword" shown.'
Question_infer_definition_from_images_w_neg = '###Question###: Based on the images provided, what is the definition of a "pseudoword"? give a detailed definition of a "pseudoword" that is consistent with all images of a "pseudoword" shown and excludes all images not labeled as a "pseudoword".'
questions = {"classify_fromImg": Question_infer_from_images,
             "classify_fromImg&Def": Question_infer_from_images_and_definition,
             "classify_fromDef": Question_infer_from_definition,
             "inferDef": {"pos":Question_infer_definition_from_images, "pos and neg":Question_infer_definition_from_images}}

answer_format_binary = '###Answer Format###: Format your answer as: [only Yes/No].'
answer_format_cot = '###Answer Format###: Format your answer as: "Thinking: [step-by-step thinking]. '+pre_answer+' [only Yes/No]."'
answer_format_definition = '###Answer Format###: Format your answer as: "'+pre_answer_definition+' [definition of a "pseudoword"]."'
answer_format_definition_cot = '###Answer Format###: Format your answer as: "Thinking: [step-by-step thinking].'+pre_answer_definition+' [definition of a "pseudoword"]."'
answer_format_mcq = '###Answer Format###: Format your answer as: "[only A/B]."'
answer_formats = {"binary": answer_format_binary,
                  "cot": answer_format_cot,
                  "inferDef_direct": answer_format_definition,
                  "inferDef_cot": answer_format_definition_cot,
                  "mcq": answer_format_mcq}

encourage_to_answer = '###Guidance###: This question is intended to be answered based on visual inspection alone, without any markings or additional measurements, and the information provided is sufficient for arriving at a correct final answer. Therefore, even if you believe that you cannot definitively conclude the correct answer, you must provide your best guess based on visually inspecting the images provided.'

definition = '###Definition###: The definition of a "pseudoword" is: concept_definition'

initial_prompt_content = {
                "type": "text",
                "text": '###Examples###: Each of the following five images shown is a "pseudoword".'
              }

initial_prompt_content_negative_ex = {
                "type": "text",
                "text": '###Examples###: Each of the following two images shown is NOT a "pseudoword".'
              }