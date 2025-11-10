from camel.agents import ChatAgent
from camel.messages import BaseMessage
from camel.types import RoleType
from vector_retriever import VecRetriever
from qdrant import QdrantDB  

import os
from dotenv import load_dotenv
from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType
from camel.configs import QwenConfig
from camel.agents import ChatAgent

load_dotenv()
api_key = os.getenv('QWEN_API_KEY')
model = ModelFactory.create(
    model_platform=ModelPlatformType.QWEN,
    model_type=ModelType.QWEN_MAX,
    model_config_dict=QwenConfig(temperature=0.2).as_dict(),
    api_key=api_key
)

def single_agent(query: str) -> str:
    # 1. 初始化数据库和检索器
    db = QdrantDB() 
    retriever = VecRetriever(db)
    
    # 2. 检索相关文本
    retrieved_items = retriever.search(query, top_k=2) 
    
    # 3. 调整文本格式
    retrieved_info = "\n\n".join(
        f"文件名: {item['file_name']}\n内容: {item['content']}"
        for item in retrieved_items
    )
    
    # 4. 设置系统提示
    assistant_sys_msg = """
        我会提供给你一个用户的原始查询，以及一组从知识库中检索到的相关上下文片段。
        你的任务是：仅基于这些检索到的上下文内容，准确、简洁、有条理地回答用户的问题。
        请严格遵循以下规则：
        1. 如果检索到的上下文包含足够信息，请直接引用或转述相关内容来回答问题，不要添加未提及的信息。
        2. 如果上下文与问题无关或信息不足，请明确回答：“我不知道” 或 “根据提供的信息无法回答该问题”。
        3. 不要编造答案，即使你具备相关常识，也必须依赖给定的上下文作答。
        4. 保持回答客观、中立，避免主观推测或假设。  
    """
    
    # 5. 构造输入
    user_msg = f"原始问题：{query}\n\n检索到的相关信息：\n{retrieved_info}"
    
    # 6. 调用大模型
    # zhipu_model = ZhipuLLM(model_name="glm-4", temperature=0.5)
    agent = ChatAgent(assistant_sys_msg, model=model)
    assistant_response = agent.step(user_msg)
    
    return assistant_response.msg.content


# 测试调用
if __name__ == "__main__":
    result = single_agent("劳动的二重性指什么?")
    print(result)