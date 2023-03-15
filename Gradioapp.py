import numpy as np
import pandas as pd
import torch
import gradio as gr
import random
# model functinos

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def set_seed(seed):
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

set_seed(42)

tokenizer = AutoTokenizer.from_pretrained("ramsrigouthamg/t5_boolean_questions")
model = AutoModelForSeq2SeqLM.from_pretrained("ramsrigouthamg/t5_boolean_questions")
# tokenizer = pickle.load(open('tokenizer.pkl', 'rb'))
# model = pickle.load(open('model.pkl', 'rb'))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print(device)


def generate (inp_ids,attn_mask):
   output = model.generate(input_ids=inp_ids,
                                 attention_mask=attn_mask,
                                 max_length=256,
                                 num_beams=10,
                                 num_return_sequences=3,
                                 no_repeat_ngram_size=2,
                                 early_stopping=True
                                 )
   # print(output)
   Questions = [tokenizer.decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=True) for out in
               output]
   return [Question.strip().capitalize() for Question in Questions]



truefalse = ['yes', 'no']
def generate_question_from_passage(passage, truefalse):
   text = "truefalse: %s passage: %s </s>" % (passage, truefalse)
   max_len = 256

   encoding = tokenizer.encode_plus(text, return_tensors="pt")
   input_ids, attention_masks = encoding["input_ids"].to(device), encoding["attention_mask"].to(device)

   print ("\n\nContext: ",passage)
   output = generate(input_ids,attention_masks)
   print ("\n Questions generated were ::\n")
   # output = ' '.join([str(e) for e in output])
   return output



# gradio ui

tmp = []
def Question_generation(passage):
   ans = truefalse[random.randint(0, 1)]
   question = generate_question_from_passage(passage, ans)
   # print(len(question))
   if ans == 'yes':
      ans = 'True'
   else:
      ans = 'False'
   tmp.clear()
   tmp.append(ans)
   return question[random.randint(0, 2)]

def check(tf):
   if tf == tmp[0]:
      return "Correct!!"
   else:
      return "Wrong!!"

with gr.Blocks() as demo:
   passage = gr.Textbox(label='Passage')
   submit_btn = gr.Button('Submit')
   question = gr.Textbox(label = 'Question')
   truefalse_rd = gr.Radio(['True', 'False'], label="Choose one Option!")
   submit_btn.click(Question_generation, passage, question)
   check_ans_btn = gr.Button(label='Check?')
   label = gr.Label()
   check_ans_btn.click(check, truefalse_rd, label)



demo.launch(server_port = 8080)
