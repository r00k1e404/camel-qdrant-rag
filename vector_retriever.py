# 从Qdrant向量数据库中召回相关文本

from camel.retrievers import VectorRetriever
from qdrant import QdrantDB

class VecRetriever:
    """简单的向量检索器"""
    
    def __init__(self, qdrant_db: QdrantDB):
        """
        初始化向量检索器
        
        任务：
        1. 保存qdrant_db实例
        2. 创建VectorRetriever对象
        
        参数:
            qdrant_db: QdrantDB数据库实例（来自qdrant.py）
        """
        # TODO: 保存qdrant_db实例
        self.qdrant_db = qdrant_db
        
        # TODO: 创建VectorRetriever实例
        # 提示：需要embedding_model和storage参数
        # embedding_model来自qdrant_db.embedding_instance
        # storage来自qdrant_db.storage_instance
        self.vector_retriever = VectorRetriever(
            embedding_model=self.qdrant_db.embedding_instance,
            storage=self.qdrant_db.storage_instance
        )

    def search(self, question: str, top_k: int = 3, score_threshold: float = 0.80):
        """
        根据问题检索相关文本
        
        任务：
        1. 使用vector_retriever.query()检索
        2. 格式化返回结果
        
        参数:
            question: 检索问题
            top_k: 返回结果数量
            
        返回:
            列表，每个元素包含{'file_name': '文件名', 'content': '内容'}
        """
        # TODO: 使用self.vector_retriever.query()进行检索
        # 提示：需要query、top_k参数
        items = self.vector_retriever.query(question, top_k=top_k)
        filtered_items = [item for item in items if float(item.get('similarity score', 0.0)) >= score_threshold]

        # TODO: 将检索结果格式化为指定格式
        # 原始结果格式：[{'content path': '文件名', 'text': '内容'}, ...]
        # 目标格式：[{'file_name': '文件名', 'content': '内容'}, ...]
        result = []
        for item in filtered_items:
            result.append({
                'file_name': item['content path'],
                'content': item['text']
            })

        return result

# 使用示例：
'''
# 1. 先创建并初始化数据库（使用qdrant.py中的QdrantDB）
db = QdrantDB()

# 2. 保存一些测试数据
# db.save_text("人工智能是计算机科学的一个分支", "AI教程.txt")
# db.save_text("机器学习是实现人工智能的方法之一", "ML指南.txt")

# 3. 创建检索器
retriever = VecRetriever(db)

# 4. 检索相关文本
results = retriever.search("劳动的二重性指的是什么？", top_k=3)


# 5. 查看结果
for item in results:
    print(f"文件：{item['file_name']}")
    print(f"内容：{item['content']}")
    print("-" * 50)
'''

# 实习生任务：
# 1. 确保已完成qdrant.py中的TODO部分
# 2. 完成本文件中的TODO部分，实现向量检索功能
# 3. 测试完整的存储->检索流程