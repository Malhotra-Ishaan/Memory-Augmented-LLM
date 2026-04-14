import streamlit as st
from google import genai
import os
import json
import ast
import re
from sentence_transformers import SentenceTransformer,util
from transformers import AutoTokenizer,AutoModelForSequenceClassification
import torch
import faiss
import time
import numpy as np

st.set_page_config(page_title="AI Memory Chat",layout="wide")

os.environ["GEMINI_API_KEY"]=""

client=genai.Client()

model=SentenceTransformer('all-mpnet-base-v2')
model_name="cross-encoder/nli-deberta-v3-base"
tokenizer=AutoTokenizer.from_pretrained(model_name)
nli_model=AutoModelForSequenceClassification.from_pretrained(model_name)
labels=["contradiction","neutral","entailment"]

file_path_history="data1.json"
file_path_memory="memory1.json"

def read_history():
    if not os.path.exists(file_path_history) or os.path.getsize(file_path_history)==0:
        return []
    with open(file_path_history,"r") as f:
        return json.load(f)

def save_history(data):
    with open(file_path_history,"w") as f:
        json.dump(data,f,indent=4)

def read_memory():
    if not os.path.exists(file_path_memory) or os.path.getsize(file_path_memory)==0:
        return {"preferences":[],"facts":[]}
    with open(file_path_memory,"r") as f:
        return json.load(f)

def save_memomry(data):
    with open(file_path_memory,"w") as f:
        json.dump(data,f,indent=4)

TOP_K=5

def build_faiss_index(memory_list):
    if len(memory_list)==0:
        return None
    embeddings=np.array([item["embedding"] for item in memory_list]).astype("float32")
    faiss.normalize_L2(embeddings)
    dim=embeddings.shape[1]
    index=faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index

def retrieve_memory(userinput,memory_list,model,top_k=TOP_K):
    if len(memory_list)==0:
        return []
    query_embedding=model.encode([userinput]).astype("float32")
    faiss.normalize_L2(query_embedding)
    index=build_faiss_index(memory_list)
    scores,indices=index.search(query_embedding,top_k)
    results=[]
    for i,idx in enumerate(indices[0]):
        if idx==-1:
            continue
        sim_score=scores[0][i]
        importance=memory_list[idx]["importance_score"]
        final_score=0.7*sim_score+0.3*importance
        results.append((final_score,memory_list[idx]["text"]))
    results=sorted(results,key=lambda x:x[0],reverse=True)
    return [x[1] for x in results]

def get_response(userinput):
    messages=read_history()
    messages.append({"role":"user","parts":[{"text":str(userinput)}]})
    all_memory=read_memory()
    preferences=retrieve_memory(userinput,all_memory["preferences"],model)
    facts=retrieve_memory(userinput,all_memory["facts"],model)
    response=client.models.generate_content(
        model="gemma-3-1b-it",
        contents=f"""Use the following user preferences and facts ONLY if relevant.

Preferences:
{preferences}

Facts:
{facts}

User input:
{userinput}

Rules:
- Follow preferences if applicable
- Use facts if relevant
- Do NOT force irrelevant memory
"""
    )
    model_txt=response.text if response.text else "..."
    messages.append({"role":"model","parts":[{"text":model_txt}]})
    save_history(messages)
    return model_txt

def memory_extractor(userinput):
    pref_fact=client.models.generate_content(
        model="gemma-4-26b-a4b-it",
        contents=f"""Extract user preferences and facts.

STRICT RULES:
1. DO NOT assume anything
2. Extract only explicitly stated info
3. Return complete sentences
4. Output JSON only

USER INPUT:
{userinput}

FORMAT:
{{"preferences":[],"facts":[]}}
"""
    )
    data_memory_str=pref_fact.text
    data_memory_str=data_memory_str.strip("`").replace("json\n","").strip()
    data_memory_json=ast.literal_eval(data_memory_str)

    def data_clean(sentence):
        sentence=sentence.lower().strip()
        sentence=re.sub(r"[^\w\s]","",sentence)
        return sentence

    def compute_importance(freq,last_seen,freq_w=0.15,rec_w=0.85):
        current_time=time.time()
        age_hours=(current_time-last_seen)/3600
        recency=1/(1+age_hours)
        return freq_w*freq+rec_w*recency

    def get_nli_label(sent1,sent2):
        inputs=tokenizer(sent1,sent2,return_tensors="pt",truncation=True)
        with torch.no_grad():
            outputs=nli_model(**inputs)
        probs=torch.softmax(outputs.logits,dim=1)
        label_id=torch.argmax(probs).item()
        return labels[label_id],probs[0][label_id].item()

    def check_duplication(cleaned_data,all_data):
        if isinstance(cleaned_data,str):
            cleaned_data=[cleaned_data]
        if len(all_data)==0:
            new_entries=[]
            for s in cleaned_data:
                emb=model.encode([s])[0]
                now=time.time()
                new_entries.append({"text":s,"embedding":emb.tolist(),"frequency":1,"last_seen":now,"importance_score":compute_importance(1,now)})
            return cleaned_data,new_entries

        unique_sentences=[]
        to_remove=set()
        old_texts=[item["text"] for item in all_data]
        old_embeddings=[item["embedding"] for item in all_data]
        new_embeddings=model.encode(cleaned_data)
        similarity=util.cos_sim(new_embeddings,old_embeddings)

        for i in range(len(cleaned_data)):
            max_idx=similarity[i].argmax().item()
            max_score=similarity[i][max_idx].item()
            best_match=old_texts[max_idx]

            if max_score<0.65:
                unique_sentences.append(cleaned_data[i])
                continue

            label,_=get_nli_label(cleaned_data[i],best_match)

            if label=="contradiction":
                to_remove.add(max_idx)
                unique_sentences.append(cleaned_data[i])
            elif label=="entailment":
                now=time.time()
                all_data[max_idx]["frequency"]+=1
                all_data[max_idx]["last_seen"]=now
                all_data[max_idx]["importance_score"]=compute_importance(all_data[max_idx]["frequency"],all_data[max_idx]["last_seen"])
            else:
                unique_sentences.append(cleaned_data[i])

        updated_all_data=[item for idx,item in enumerate(all_data) if idx not in to_remove]

        for sentence in unique_sentences:
            emb=model.encode([sentence])[0]
            now=time.time()
            updated_all_data.append({"text":sentence,"embedding":emb.tolist(),"frequency":1,"last_seen":now,"importance_score":compute_importance(1,now)})

        return unique_sentences,updated_all_data

    loaded_memory_data=read_memory()

    for d in data_memory_json["preferences"]:
        cleaned_data=data_clean(str(d))
        _,updated_memory=check_duplication(cleaned_data,loaded_memory_data["preferences"])
        loaded_memory_data["preferences"]=updated_memory

    for d in data_memory_json["facts"]:
        cleaned_data=data_clean(str(d))
        _,updated_memory=check_duplication(cleaned_data,loaded_memory_data["facts"])
        loaded_memory_data["facts"]=updated_memory

    save_memomry(loaded_memory_data)

if not os.path.exists(file_path_memory):
    save_memomry({"preferences":[],"facts":[]})

if not os.path.exists(file_path_history):
    save_history([])

st.title("AI Memory Chat")

col1,col2=st.columns(2)
with col1:
    if st.button("➕ New Chat"):
        st.session_state.chat=[]
        save_history([])
        st.rerun()
with col2:
    if st.button("🗑️ Clear Memory"):
        save_memomry({"preferences":[],"facts":[]})
        st.success("Memory cleared")
        st.rerun()

if "chat" not in st.session_state:
    history=read_history()
    formatted=[]
    for msg in history:
        role=msg["role"]
        text=msg["parts"][0]["text"]
        formatted.append({"role":role,"parts":[{"text":text}]})
    st.session_state.chat=formatted

def typing_effect(text):
    placeholder=st.empty()
    full=""
    for ch in text:
        full+=ch
        placeholder.markdown(full)
        time.sleep(0.01)

userinput=st.chat_input("Type your message")

if userinput:
    response=get_response(userinput)
    memory_extractor(userinput)
    st.session_state.chat.append({"role":"user","parts":[{"text":userinput}]})
    st.session_state.chat.append({"role":"model","parts":[{"text":response}]})
    st.rerun()

for i,msg in enumerate(st.session_state.chat):
    role=msg["role"]
    text=msg["parts"][0]["text"]

    if role=="user":
        with st.chat_message("user"):
            st.markdown(text)
    else:
        with st.chat_message("assistant"):
            if i==len(st.session_state.chat)-1:
                typing_effect(text)
            else:
                st.markdown(text)

with st.expander("Memory"):
    memory_data=read_memory()
    st.subheader("Preferences")
    for item in memory_data["preferences"]:
        st.write(item["text"])
    st.subheader("Facts")
    for item in memory_data["facts"]:
        st.write(item["text"])