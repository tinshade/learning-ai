import os

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class ChatWDocs:
    def __init__(self, base_pdf_file_path: str = ""):
        self.base_pdf_file_path: str = base_pdf_file_path or os.path.join(
            os.getcwd(), "data"
        )
        self.local_model_name = "mistral"
        self.chunk_size: int = 800
        self.chunk_overlap: int = 80
        self.models: dict = {"ollama": "nomic-embed-text"}
        self.db_persistence_paths: dict = {
            "chroma": os.path.join(os.getcwd(), "memory", "chroma")
        }

        self.prompt_template = """
        Answer the question based only on the following context:
        {context}

        ---
        Answer the question based on the above context: {question}

        """

    def load_documents(self, file_path: str):
        document_loader = PyPDFDirectoryLoader(file_path)
        return document_loader.load()

    def split_documents(self, documents: list[Document]):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )
        return text_splitter.split_documents(documents)

    def assign_chunk_ids(self, chunks: list[Document]):
        last_page_id = None
        current_chunk_index = 0
        for chunk in chunks:
            source = chunk.metadata.get("source")
            page = chunk.metadata.get("page")
            current_page_id = f"{source}:{page}"
            if current_page_id == last_page_id:
                current_chunk_index += 1
            else:
                current_chunk_index = 0
            chunk.metadata["id"] = f"{current_page_id}:{current_chunk_index}"
            last_page_id = current_page_id
        return chunks

    def get_embedding_function(self):
        return OllamaEmbeddings(model=self.models["ollama"])

    def add_to_chroma(self, chunks: list[Document]):
        db = Chroma(
            persist_directory=self.db_persistence_paths["chroma"],
            embedding_function=self.get_embedding_function(),
        )
        existing_items = db.get(include=[])
        existing_ids = set(existing_items["ids"])
        print(f"Number of existing documents in DB: {len(existing_ids)}")

        new_chunks = []
        new_chunk_ids = []
        for chunk in chunks:
            if chunk.metadata["id"] not in existing_ids:
                new_chunks.append(chunk)
                new_chunk_ids.append(chunk.metadata["id"])

        if new_chunks:
            print(f"Adding {len(new_chunks)} new chunks to DB")
            db.add_documents(new_chunks, ids=new_chunk_ids)
        else:
            print("No new chunks to add")

    def query_rag(self, question: str):
        db = Chroma(
            persist_directory=self.db_persistence_paths["chroma"],
            embedding_function=self.get_embedding_function(),
        )
        results = db.similarity_search_with_score(question, k=5)
        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
        return context_text

    def invoke_llm(self, question: str):
        context = self.query_rag(question)
        prompt = self.prompt_template.format(context=context, question=question)
        model = Ollama(model=self.local_model_name)
        response_text = model.invoke(prompt)
        return response_text

    def start_taking_questions(self) -> None:
        while True:
            question: str = str(input("Ask away!\n->"))
            if question in ["quit", "exit"]:
                return
            response: str = self.invoke_llm(question=question)
            print(response)


if __name__ == "__main__":
    manager = ChatWDocs()
    documents = manager.load_documents(manager.base_pdf_file_path)
    chunks = manager.split_documents(documents)
    chunks = manager.assign_chunk_ids(chunks)
    manager.add_to_chroma(chunks)
    manager.start_taking_questions()
