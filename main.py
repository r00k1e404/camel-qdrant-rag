import gradio as gr
from rag_agent import single_agent
from vector_retriever import VecRetriever
from qdrant import QdrantDB  
import os
from dotenv import load_dotenv
from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType
from camel.configs import QwenConfig
from camel.agents import ChatAgent

# 重新初始化模型（如果需要）
load_dotenv()
api_key = os.getenv('QWEN_API_KEY')
model = ModelFactory.create(
    model_platform=ModelPlatformType.QWEN,
    model_type=ModelType.QWEN_MAX,
    model_config_dict=QwenConfig(temperature=0.2).as_dict(),
    api_key=api_key
)

def query_interface_with_retrieval(query):
    """
    同时返回答案和检索信息
    """
    if not query.strip():
        return "请输入问题", "未提供检索到的信息"
    
    try:
        # 检索信息
        db = QdrantDB()
        retriever = VecRetriever(db)
        retrieved_items = retriever.search(query, top_k=2)
        
        retrieved_info = "\n\n".join(
            f"文件名: {item['file_name']}\n内容: {item['content']}"
            for item in retrieved_items
        )
        
        # 构造输入
        assistant_sys_msg = """
            我会提供给你一个用户的原始查询，以及一组从知识库中检索到的相关上下文片段。
            你的任务是：仅基于这些检索到的上下文内容，准确、简洁、有条理地回答用户的问题。
            请严格遵循以下规则：
            1. 如果检索到的上下文包含足够信息，请直接引用或转述相关内容来回答问题，不要添加未提及的信息。
            2. 如果上下文与问题无关或信息不足，请明确回答："我不知道" 或 "根据提供的信息无法回答该问题"。
            3. 不要编造答案，即使你具备相关常识，也必须依赖给定的上下文作答。
            4. 保持回答客观、中立，避免主观推测或假设。  
            5. 标注来源
        """
        
        user_msg = f"原始问题：{query}\n\n检索到的相关信息：\n{retrieved_info}"
        
        agent = ChatAgent(assistant_sys_msg, model=model)
        assistant_response = agent.step(user_msg)
        
        answer = assistant_response.msg.content
        
        return answer, retrieved_info
    except Exception as e:
        return f"发生错误: {str(e)}", "检索失败"

# 创建Gradio界面
with gr.Blocks(title="RAG问答系统") as demo:
    gr.Markdown("# 🤖 RAG问答系统")
    gr.Markdown("基于向量检索增强生成的问答系统，输入问题即可获得基于知识库的回答。")
    
    with gr.Row():
        with gr.Column():
            query_input = gr.Textbox(
                label="请输入您的问题",
                placeholder="例如：劳动的二重性指什么？",
                lines=3
            )
            submit_btn = gr.Button("提交问题", variant="primary")
        
        with gr.Column():
            answer_output = gr.Textbox(
                label="AI回答",
                placeholder="答案将在这里显示...",
                lines=8,
                interactive=False
            )
    
    with gr.Accordion("检索到的相关信息", open=False):
        retrieved_output = gr.Textbox(
            label="检索到的知识库内容",
            placeholder="检索到的相关信息将在这里显示...",
            lines=10,
            interactive=False
        )
    
    # 绑定事件
    submit_btn.click(
        fn=query_interface_with_retrieval,
        inputs=query_input,
        outputs=[answer_output, retrieved_output]
    )
    
    # 回车提交
    query_input.submit(
        fn=query_interface_with_retrieval,
        inputs=query_input,
        outputs=[answer_output, retrieved_output]
    )
    
    # 示例问题
    gr.Examples(
        examples=[
            "什么是商品的使用价值?",
            "交换价值是什么?",
            "劳动的二重性指什么?",
            "今天晚饭吃什么?",
        ],
        inputs=query_input,
        label="示例问题"
    )

# http://localhost:7860
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)