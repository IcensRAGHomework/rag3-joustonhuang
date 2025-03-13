import datetime
import chromadb
import traceback
import pandas as pd
import time
from chromadb.utils import embedding_functions
from model_configurations import get_model_configuration

# 設定 GPT 嵌入版本
GPT_EMB_VERSION = 'text-embedding-ada-002'
gpt_emb_config = get_model_configuration(GPT_EMB_VERSION)

# 資料庫路徑及 CSV 檔案名
DB_PATH = "./"
CSV_FILE = "COA_OpenData.csv"

def initialize_chroma_client():
    """初始化 Chroma 客戶端"""
    return chromadb.PersistentClient(path=DB_PATH)

def get_embedding_function():
    """取得 OpenAI 嵌入函數"""
    return embedding_functions.OpenAIEmbeddingFunction(
        api_key=gpt_emb_config['api_key'],
        api_base=gpt_emb_config['api_base'],
        api_type=gpt_emb_config['openai_type'],
        api_version=gpt_emb_config['api_version'],
        deployment_id=gpt_emb_config['deployment_name']
    )

def create_collection(chroma_client, embedding_function):
    """建立或取得集合"""
    return chroma_client.get_or_create_collection(
        name="TRAVEL",
        metadata={"hnsw:space": "cosine"},
        embedding_function=embedding_function
    )

def populate_collection(collection):
    """如果集合是空的，則從 CSV 檔案中填充資料"""
    if collection.count() == 0:
        df = pd.read_csv(CSV_FILE)
        for _, row in df.iterrows():
            metadata = {
                "file_name": CSV_FILE,
                "name": row["Name"],
                "type": row["Type"],
                "address": row["Address"],
                "tel": row["Tel"],
                "city": row["City"],
                "town": row["Town"],
                "date": int(datetime.datetime.strptime(row["CreateDate"], "%Y-%m-%d").timestamp())
            }
            document = row.get("HostWords", "")
            document_id = str(row["ID"])
            collection.add(ids=[document_id], documents=[document], metadatas=[metadata])
    return collection

def generate_hw01():
    """生成 HW01 集合"""
    chroma_client = initialize_chroma_client()
    embedding_function = get_embedding_function()
    collection = create_collection(chroma_client, embedding_function)
    return populate_collection(collection)

def generate_hw02(question, city, store_type, start_date, end_date):
    """生成 HW02 查詢結果"""
    collection = generate_hw01()
    query_results = collection.query(
        query_texts=[question],
        n_results=10,
        where={
            "$and": [
                {"date": {"$gte": int(start_date.timestamp())}},
                {"date": {"$lte": int(end_date.timestamp())}},
                {"type": {"$in": store_type}},
                {"city": {"$in": city}}
            ]
        }
    )
    metadatas = query_results['metadatas'][0]
    distances = query_results['distances'][0]
    sorted_results = sorted(
        zip(metadatas, distances),
        key=lambda x: 1 - x[1],  # similarity = 1 - distance
        reverse=True
    )
    filtered_results = [
        metadata['name'] for metadata, distance in sorted_results if (1 - distance) >= 0.8
    ]
    return filtered_results

def generate_hw03(question, store_name, new_store_name, city, store_type):
    """生成 HW03 查詢及更新結果"""
    collection = generate_hw01()

    # 1. 更新指定店家的資訊
    store_results = collection.query(query_texts=[store_name], n_results=1)
    if store_results["metadatas"] and store_results["metadatas"][0]:
        metadata = store_results["metadatas"][0][0]
        doc_id = store_results["ids"][0][0]
        metadata["new_store_name"] = new_store_name
        collection.upsert(ids=[doc_id], documents=[store_results["documents"][0][0]], metadatas=[metadata])

    # 2. 查詢符合條件的店家
    query_results = collection.query(
        query_texts=[question],
        n_results=10,
        where={
            "$and": [
                {"type": {"$in": store_type}},
                {"city": {"$in": city}}
            ]
        }
    )

    # 3. 篩選相似度 >= 0.80 的結果，並使用 new_store_name（如果存在）
    metadatas = query_results['metadatas'][0]
    distances = query_results['distances'][0]
    sorted_results = sorted(
        zip(metadatas, distances),
        key=lambda x: 1 - x[1],  # similarity = 1 - distance
        reverse=True
    )
    filtered_results = [
        metadata.get("new_store_name", metadata["name"])  # 優先顯示 new_store_name
        for metadata, distance in sorted_results if (1 - distance) >= 0.8
    ]
    return filtered_results

def demo(question):
    """展示功能"""
    chroma_client = initialize_chroma_client()
    embedding_function = get_embedding_function()
    collection = create_collection(chroma_client, embedding_function)
    return collection

if __name__ == "__main__":
    # question = "What are the best travel destinations?"
    # collection = demo(question)
    # print("Collection successfully created/retrieved:", collection.name)
    # generate_hw01()
    # question = "我想要找有關茶餐點的店家"
    # city = ["宜蘭縣", "新北市"]
    # store_type = ["美食"]
    # start_date = datetime.datetime(2024, 4, 1)
    # end_date = datetime.datetime(2024, 5, 1)
    # ans_list = generate_hw02(question, city, store_type, start_date, end_date)
    # print(ans_list)
    
    question = "我想要找南投縣的田媽媽餐廳，招牌是蕎麥麵"
    store_name = "耄饕客棧"
    new_store_name = "田媽媽（耄饕客棧）"
    city = ["南投縣"]
    store_type = ["美食"]
    ans_list2 = generate_hw03(question, store_name, new_store_name, city, store_type)
    print(ans_list2)
