# 环境变量加载保持不变
from dotenv import load_dotenv

load_dotenv()
import os
# 1.Load 导入Document Loaders
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import UnstructuredExcelLoader

# 加载Documents
base_dir = './cfb' # 文档的存放目录
documents = []
for file in os.listdir(base_dir):
    # 构建完整的文件路径
    file_path = os.path.join(base_dir, file)
    if file.endswith('.pdf'):
        loader = PyPDFLoader(file_path)
        documents.extend(loader.load())
#    elif file.endswith('.docx'):
 #       loader = Docx2txtLoader(file_path)
  #      documents.extend(loader.load())
#    elif file.endswith('.xlsx'):
 #       loader = UnstructuredExcelLoader(file_path)
  #      documents.extend(loader.load())

# 2.Split 将Documents切分成块以便后续进行嵌入和向量存储
from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=10)
chunked_documents = text_splitter.split_documents(documents)

# 3.Store 将分割嵌入并存储在矢量数据库Qdrant中
from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import HuggingFaceEmbeddings

# 加载 HuggingFace Hub 模型名称
model_path = "BAAI/bge-small-zh-v1.5"
embeddings = HuggingFaceEmbeddings(model_name=model_path)

vectorstore = Qdrant.from_documents(
    documents=chunked_documents, # 以分块的文档
    embedding=embeddings,
    location=":memory:",  # in-memory 存储
    collection_name="my_documents",) # 指定collection_name

# 4. Retrieval 准备模型和Retrieval链
import logging # 导入Logging工具
from langchain_community.chat_models import ChatOpenAI # ChatOpenAI模型
from langchain.retrievers.multi_query import MultiQueryRetriever # MultiQueryRetriever工具
from langchain.chains import RetrievalQA # RetrievalQA链

# 设置Logging
logging.basicConfig()
logging.getLogger('langchain.retrievers.multi_query').setLevel(logging.INFO)

# 实例化一个大模型工具 - OpenAI的GPT-3.5
llm = ChatOpenAI(
            model="deepseek-r1",
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )

# 实例化一个MultiQueryRetriever
retriever_from_llm = MultiQueryRetriever.from_llm(retriever=vectorstore.as_retriever(), llm=llm)

# 实例化一个RetrievalQA链
#qa_chain = RetrievalQA.from_chain_type(llm,retriever=retriever_from_llm)


from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
#from langchain_openai import ChatOpenAI


retriever = retriever_from_llm  # Your retriever

system_prompt = (
    "Use the given context to answer the question. "
    "If you don't know the answer, say you don't know. "
    "Use three sentence maximum and keep the answer concise. "
    "Context: {context}"
)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)
question_answer_chain = create_stuff_documents_chain(llm, prompt)
chain = create_retrieval_chain(retriever, question_answer_chain)

#chain.invoke({"input": query})


# 5. Output 问答系统的UI实现
from flask import Flask, request, render_template

app = Flask(__name__)  # Flask APP


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # 接收用户输入作为问题
        question = request.form.get('question')

        # RetrievalQA链 - 读入问题，生成答案
        #result = qa_chain({"query": question})
        result = chain.invoke({"input": question})
        print(result)

        # 把大模型的回答结果返回网页进行渲染
        return render_template('index.html', result=result)

    return render_template('index.html')


if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True, port=5000)