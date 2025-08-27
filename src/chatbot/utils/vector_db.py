import json

import pandas as pd
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS


class VecDBManager:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.vec_db: FAISS | None = None

    def init_from_texts(self, texts: list[str]):
        embeddings = OpenAIEmbeddings(api_key=self.api_key)
        self.vec_db = FAISS.from_texts(texts, embeddings)

    def init_from_csv(self, csv_path: str):
        df = pd.read_csv(csv_path)

        texts = []
        for _, row in df.iterrows():
            combined_text = " | ".join([str(row[col]) for col in df.columns])
            texts.append(combined_text)

        self.init_from_texts(texts)

    def init_from_json(self, json_path: str):
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)

        texts = []
        if isinstance(data, list):
            # json array, each element is a dict
            for item in data:
                if isinstance(item, dict):
                    combined_text = " | ".join([f"{k}: {v}" for k, v in item.items()])
                else:
                    combined_text = str(item)
                texts.append(combined_text)
        elif isinstance(data, dict):
            # Single json object
            combined_text = " | ".join([f"{k}: {v}" for k, v in data.items()])
            texts.append(combined_text)
        else:
            texts.append(str(data))

        self.init_from_texts(texts)
