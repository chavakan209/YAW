import csv
from tqdm import tqdm
from transformers import LlamaTokenizer
import wikipedia
wikipedia.set_lang("th")
from langchain.chat_models import ChatOpenAI
from langchain.schema.messages import HumanMessage, SystemMessage
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
import json
import argparse
import random
import time
import openai
 
parser = argparse.ArgumentParser(description="")
parser.add_argument("--wiki", type=int, default=1, help="Num of random wikipedia pages, default value of 10.")
args = parser.parse_args()
wiki = int(args.wiki)

import requests
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures

OPEN_AI_API_KEY = "your-open-ai-api-key"
openai.api_key = OPEN_AI_API_KEY

def fetch_random_article(_):
    random_article = wikipedia.random()
    return random_article

titles_set = set()

with concurrent.futures.ThreadPoolExecutor() as executor, tqdm(position=0, leave=True) as pbar:
    while len(titles_set) < wiki:
        new_titles = set(executor.map(lambda _: fetch_random_article(_), range(wiki - len(titles_set))))
        titles_set.update(new_titles)
        pbar.update(len(new_titles))

# Convert the set to a list
titles = list(titles_set)

titles = list(set(titles))
MODEL_NAME = 'huggingface model' #'openthaigpt/openthaigpt-0.1.0-beta-ckpt-hf'
tokenizer = LlamaTokenizer.from_pretrained(MODEL_NAME)

def limit_text_length(text, max_length=1000):
    if len(text) > max_length:
        return text[:max_length]
    else:
        return text

def get_wiki_content(title):
    try:
        limited_text = limit_text_length(wikipedia.page(title).content)
        chat = ChatOpenAI(openai_api_key=OPEN_AI_API_KEY, timeout=60)  # Increase the timeout value
        messages = [
            SystemMessage(content="Generate a pair of question and answer in Thai from the following content. The generated answer will be given in JSON format with keys 'คำถาม', 'คำตอบ'"),
            HumanMessage(content=limited_text),  # wikicontent
        ]

        y = chat.invoke(messages).content
        return y, limited_text
    except Exception as e:
        return None

def combine_wiki_content(titles):
    txt1= ''
    txt2 = ''
    with open('longwiki.csv', encoding="utf8") as file_obj: 
        # Create reader object by passing the file  
        # object to reader method 
        csv_reader = csv.reader(file_obj) 
        _ = next(csv_reader, None)
        txt2 = next(csv_reader, None)[0]
        txt2 = txt2[:len(txt2)//3]
        # Iterate over each row in the csv  
        # file using reader object 

        with ThreadPoolExecutor() as executor:
            for i in range(0, len(titles), 10):
                lst = []
                with tqdm(total=10, desc=f'wiki{i}-{i+10}') as pbar:
                    # Define a function to update tqdm progress bar
                    def update_pbar(_):
                        pbar.update(1)

                    # Submit tasks to the ThreadPoolExecutor
                    futures = [executor.submit(get_wiki_content, title) for title in titles[i:i+10]]

                    # Add the update_pbar function as a callback to each future
                    for future in futures:
                        future.add_done_callback(update_pbar)

                    # Process results and chunk text
                    for future in futures:
                        try:
                            result = future.result()
                            txt1 = txt2
                            txt2 = next(csv_reader, None)[0]
                            txt2 = txt2[:len(txt2)//3]
                            if result is not None:
                                text = [txt1,txt2,result[1]]
                                data = json.loads(result[0])
                                random.shuffle(text)
                                data['ข้อมูล'] = " ".join(map(str, text))
                                lst.append(data)
                        except Exception as e:
                            continue

                with open(f'longwikiQA{i//10}.json', 'a', encoding="utf-8") as json_file:
                    json.dump(lst, json_file, ensure_ascii=False, indent=2)
                time.sleep(5)
        
        return None
        # return lst

combine_wiki_content(titles)
