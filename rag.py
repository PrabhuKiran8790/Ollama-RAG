from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_models import ChatOllama
from langchain.chains.conversational_retrieval.base import BaseConversationalRetrievalChain
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler


class Rag:
    def __init__(self, file_path: str, model: str = 'mistral:7b-instruct-q4_0', chunk_size: int = 200, chunk_overlap: int = 20) -> None:
        self.file_path = file_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.llm: ChatOllama
        self.qa_chain: BaseConversationalRetrievalChain
        self.model: str = model
        self.retriever = self.create_embeddings()

    def __repr__(self) -> str:
        return f"Rag(file: '{self.file_path}', Model: {self.model})"

    def text_splitter(self):
        loader = PyPDFLoader(self.file_path)
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        texts = text_splitter.split_documents(documents)
        return texts


    def create_embeddings(self):
        texts = self.text_splitter()
        embeddings = HuggingFaceEmbeddings()
        db = Chroma.from_documents(texts, embeddings)
        retriever = db.as_retriever(search_kwargs={'k': 2})
        return retriever

    def query(self, prompt: str, chat_history: list) -> str:
        self.llm = ChatOllama(model=self.model, temperature=0, callbacks=[StreamingStdOutCallbackHandler()])
        
        if self.retriever is None:
            self.retriever = self.create_embeddings()

        self.qa_chain = ConversationalRetrievalChain.from_llm(self.llm, self.retriever, return_source_documents=True)
        result = self.qa_chain.invoke({'question': prompt, 'chat_history': chat_history})
        return result['answer']