import streamlit as st
import numpy as np
import pandas as pd
import torch
st.title("Question Generation App")

# model functinos

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def set_seed(seed):
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

set_seed(98)

# tokenizer = AutoTokenizer.from_pretrained("ramsrigouthamg/t5_boolean_questions")
# model = AutoModelForSeq2SeqLM.from_pretrained("ramsrigouthamg/t5_boolean_questions")
tokenizer = pickle.load(open('tokenizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print ("device ",device)
model = model.to(device)
# device


def generate (inp_ids,attn_mask):
  output = model.generate(input_ids=inp_ids,
                                 attention_mask=attn_mask,
                                 max_length=256,
                               num_beams=10,
                               num_return_sequences=3,
                               no_repeat_ngram_size=2,
                               early_stopping=True
                               )
  Questions = [tokenizer.decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=True) for out in
               output]
  return [Question.strip().capitalize() for Question in Questions]

# passage = []
# truefalse = []
# passage.append("Starlink, of SpaceX, is a satellite constellation project being developed by Elon Musk and team to give satellite Internet go-to access for people in any part of the world. The plan is to comprise thousands of mass-delivered little satellites in low Earth circle, orbit, working in mix with ground handheld devices, for instance, our iPhones. Elon Musk speaks about it as a grand Idea that could change the way we view and access the world around us.")
# truefalse.append("no")


# passage.append("About 400 years ago, a battle was unfolding about the nature of the Universe. For millennia, astronomers had accurately described the orbits of the planets using a geocentric model, where the Earth was stationary and all the other objects orbited around it.")
# truefalse.append("no")


# passage.append('''Months earlier, Coca-Cola had begun “Project Kansas.” It sounds like a nuclear experiment but it was just a testing project for the new flavor. In individual surveys, they’d found that more than 75% of respondents loved the taste, 15% were indifferent, and 10% had a strong aversion to the taste to the point that they were angry.''')
# truefalse.append("no")


# passage.append("The US has passed the peak on new coronavirus cases, President Donald Trump said and predicted that some states would reopen this month. The US has over 637000 confirmed Covid-19 cases and over 30826 deaths, the highest for any country in the world.")
# truefalse.append("yes")
# passage.append('''Mohandas Karamchand Gandhi was an Indian lawyer, anti-colonial nationalist and political ethicist who employed nonviolent resistance to lead the successful campaign for India's independence from British rule, and to later inspire movements for civil rights and freedom across the world.''')
# truefalse.append("no")

def generate_question_from_passage(passage, truefalse = 'yes'):
    text = "truefalse: %s passage: %s </s>" % (passage, truefalse)


    max_len = 256

    encoding = tokenizer.encode_plus(text, return_tensors="pt")
    input_ids, attention_masks = encoding["input_ids"].to(device), encoding["attention_mask"].to(device)



    print ("\n\nContext: ",passage)
    output = generate(input_ids,attention_masks)
    print ("\n Questions generated were ::\n")
    # output = ' '.join([str(e) for e in output])
    return output



# text_input = st.text_input(
#         "Enter some text 👇",
#         label_visibility=st.session_state.visibility,
#         disabled=st.session_state.disabled,
#         placeholder=st.session_state.placeholder,
#     )
text = st.text_input("Enter Text below", '''Mohandas Karamchand Gandhi was an Indian lawyer, anti-colonial nationalist and political ethicist who employed nonviolent resistance to lead the successful campaign for India's independence from British rule, and to later inspire movements for civil rights and freedom across the world''')
# run function from this text_input
out = []
if text != '' :
   st.write('Generating Questions!!')
   out = generate_question_from_passage(text)
# display out

for x in out:
    st.write(x)
