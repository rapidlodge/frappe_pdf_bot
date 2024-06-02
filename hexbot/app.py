import os
import frappe
from langchain_openai import ChatOpenAI
# from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_core.prompts import PromptTemplate
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain_community.vectorstores import DocArrayInMemorySearch
import param

global_chat = None

class chat_gen():
    chat_history = param.List([])

    def __init__(self):
        self.chain = self.load_model()
    def load_doc(self):
        doc = frappe.db.get_value('hexbot-1', 'hexbot01', ['file1', 'file2', 'file3', 'file4'], as_dict=1)
        loaders = []
        if any(doc.values()):
            print()
            print()
            print(doc.values())
            print()
            for value in doc.values():
                loader = [PyPDFLoader(f'/home/frappe-shahadat/shuvo/frappe-bench/sites/myerp/public{value}')]
                loaders.extend(loader)
        else:
            loaders = PyPDFLoader(f'/home/frappe-shahadat/shuvo/frappe-bench/sites/myerp/public{doc.file2}')
        documents = []
        for loader in loaders:
            documents.extend(loader.load())
        documents = loader.load()
        # print(docs[0].page_content)
        text_splitter = RecursiveCharacterTextSplitter(
            # Set a really small chunk size, just to show.
            chunk_size=500,
            chunk_overlap=100,
            # length_function=len,
            # is_separator_regex=False,
        )
        # texts = text_splitter.create_documents([documents[0].page_content])
        docs = text_splitter.split_documents(documents)
        embeddings = OpenAIEmbeddings(api_key="")

        db = DocArrayInMemorySearch.from_documents(docs, embeddings)
        retriever = db.as_retriever()
        return retriever

    def load_model(self):
        llm = ChatOpenAI(openai_api_key="")

        # Define your system instruction
        # system_instruction = """ As an AI assistant, you must answer the query from the user from the retrieved
        #   content, no relavant information is available, just say you are not permitted to answer other than that"""
        system_instruction = """As an AI assistant, your primary task is to respond to user queries based on the content you have retrieved.
          In cases where no relevant information is available, you must inform the user that you are not permitted to answer beyond that point.
            Clearly explain the approach you take to ensure accurate and relevant responses, and detail how you handle situations where the necessary information cannot be found. 
            Additionally, provide an example scenario to illustrate how you would communicate this limitation to the user in a clear and concise manner."

                                1. This enhanced prompt:

                                2. Clarifies the AI's primary task and the context in which it operates.
                                3. Encourages the AI to explain its approach and methods.
                                4. Specifies the need to detail how it handles lack of information.
                                5. Requests an example scenario to illustrate the communication process."""

        # Define your template with the system instruction
        template = (
            f"{system_instruction} "
            "Combine the chat history{chat_history} and follow up question into "
            " {context}. "
            "Follow up question: {question}"
        )

        prompt = PromptTemplate.from_template(template)
        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=self.load_doc(),
            #condense_question_prompt=prompt,
            combine_docs_chain_kwargs={'prompt': prompt},
            chain_type="stuff",
        )
        return chain

    def ask_pdf(self,user_input):
        result = self.chain.invoke({"question":user_input,"chat_history": self.chat_history})
        self.chat_history.append((user_input, result["answer"]))
        # print(result)
        return result['answer']


# Initialize chat_gen object if not already initialized
if global_chat is None:
    global_chat = chat_gen()

@frappe.whitelist()
def get_chat_response(user_input):
    res = global_chat.ask_pdf(user_input)
    return res


# chat = chat_gen()
# ct = chat.load_doc()
# print(ct)
# print(chat.ask_pdf("what is water logging?"))
# print(chat.ask_pdf("Does it cause for global warming?"))