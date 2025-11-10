import os
import json
from camel.storages import QdrantStorage
from camel.embeddings import SentenceTransformerEncoder
from camel.storages import VectorRecord

class QdrantDB:
    """简单的Qdrant向量数据库操作类"""
    
    def __init__(self, model_name: str = "./models/TencentBAC/Conan-embedding-v1"):
        """
        初始化Qdrant数据库
        
        任务：
        1. 设置数据存储路径
        2. 初始化embedding模型
        3. 创建QdrantStorage实例
        
        参数:
            model_name: huggingface模型名称
        """
        rootpath = os.path.join(os.path.dirname(__file__), "qdrant_data")
        os.makedirs(rootpath, exist_ok=True)
        
        self.embedding_instance = SentenceTransformerEncoder(model_name=model_name)

        self.storage_instance = QdrantStorage(
            vector_dim=self.embedding_instance.get_output_dim(),
            collection_name="my first collection",
            path=rootpath,
        )
        
    def save_text(self, text: str, source_file: str = "unknown"):
        """
        保存单个文本到数据库
        
        任务：
        1. 将文本转换为向量
        2. 创建VectorRecord
        3. 保存到数据库
        
        参数:
            text: 要保存的文本
            source_file: 文本来源文件名
        """
        # TODO: 使用embedding_instance将文本转换为向量
        # 提示：使用embed_list方法
        embedding = self.embedding_instance.embed_list([text])[0]
        
        # TODO: 创建payload字典，包含text和source_file信息
        payload = {
            "text": text,
            "content path": source_file
        }

        # TODO: 创建VectorRecord对象
        record = VectorRecord(vector=embedding, payload=payload)

        # TODO: 使用storage_instance.add()保存记录
        self.storage_instance.add([record])

    def save_from_json_file(self, json_path: str, source_file: str = "unknown"):
        with open(json_path, 'r', encoding='utf-8') as f:
            data_list = json.load(f)

        print(f"共有 {len(data_list)} 条记录...")

        records = []
        for idx, item in enumerate(data_list):
            # 只处理 type 为 "text" 的条目
            if item.get("type") != "text":
                continue
            
            text = item.get("text", "").strip()
            if not text:
                continue  # 跳过空文本

            page_idx = item.get("page_idx", -1)
            source_file = os.path.basename(json_path) 

            payload = {
                "text": text,
                "content path": source_file,
                "page_idx": page_idx,
                "original_index": idx
            }

            embedding = self.embedding_instance.embed_list([text])[0]
            record = VectorRecord(vector=embedding, payload=payload)
            records.append(record)

        self.storage_instance.add(records)
        print("json保存完毕")

        

# 使用示例
'''
# 1. 创建数据库实例
db = QdrantDB()

# 2. 保存文本
db.save_text("这是第一段文本", "文档1.txt")
json_file_path = os.path.join("data", "small_ocr_content_list.json")
db.save_from_json_file(json_file_path)
# db.save_text("这是第二段文本", "文档2.txt")

print("完成！")
'''
# 实习生任务：
# 完成上述TODO部分，实现一个能够将文本保存到Qdrant向量数据库的功能