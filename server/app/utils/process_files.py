from transformers import RobertaTokenizer, RobertaModel
from logic.create_embeddings import embedding

from logic.infinite_gpt import process_chunks

model_name = 'roberta-base'
tokenizer = RobertaTokenizer.from_pretrained(model_name)
model = RobertaModel.from_pretrained(model_name)

embedding_List = []

for_gpt = []

def process_file(file_path):
    
    with open(file_path, 'r', encoding='utf-8') as file:

        prompt_text = file.read()
        for_gpt.append(prompt_text)
        # Process the file here
        print(prompt_text)

        result = embedding(prompt_text)

        # embedding_List.append(result)
        # print(result)
        # Process the file here
        # print(prompt_text)
    return (prompt_text, result)
