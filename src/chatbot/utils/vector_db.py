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

    def init_from_csv(self, csv_path: str, text_columns: list[str]):
        df = pd.read_csv(csv_path)

        texts = []
        for _, row in df.iterrows():
            combined_text = " | ".join([str(row[col]) for col in text_columns if col in df.columns])
            texts.append(combined_text)

        self.init_from_texts(texts)
