# # from langchain.chat_models import ChatOpenAI
# from langchain_openai import ChatOpenAI
# from langchain_openai import OpenAI
# # from langchain.llms import OpenAI
# # from langchain.memory import RedisChatMessageHistory, ConversationBufferMemory
# # from langchain.chains import ConversationChain, LLMChain
# # from langchain.prompts import PromptTemplate
# from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, PromptTemplate

# # from langchain.embeddings.openai import OpenAIEmbeddings
# # from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
# # from langchain_community.document_loaders import TextLoader
# from langchain.chains import (
#     StuffDocumentsChain, LLMChain, ConversationalRetrievalChain
# )
# from langchain_core.prompts import PromptTemplate
# from langchain_community.llms import OpenAI
# from langchain_community.vectorstores import DocArrayInMemorySearch
# from langchain_openai import OpenAIEmbeddings
# from langchain_text_splitters import CharacterTextSplitter
# from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_openai import OpenAIEmbeddings
# from langchain_community.vectorstores import DocArrayInMemorySearch
# # from langchain_community.document_loaders import TextLoader, PyPDFLoader
# # from langchain.chains import RetrievalQA,  ConversationalRetrievalChain
# # from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
# # from langchain_community.document_loaders import TextLoader
# # from langchain.document_loaders import PyPDFLoader
# # from langchain.document_loaders.csv_loader import CSVLoader
# # import param
# # import frappe
# # import os
import frappe
def get_context(context):
    context.users = frappe.get_list("hexbot-1", fields=["file1", "file2", "file3", "file4"])
    print(context.users)

doc = frappe.db.get_value('hexbot-1', 'hexbot01', ['file1', 'file2'], as_dict=1)

print(doc.file1)
# # custom_template = """Use the following pieces of context to answer the question at the end. Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language. If you do not know the answer reply with 'I am sorry'.
# #                         Make the response short. Do not make any extra response.

# # {context}
# # Follow Up Input: {question}
# # Your response:
# # """

# # customs_template = """Use the following pieces of context to answer the question at the end. \n
# # If you don't know the answer, just say that you don't know, don't try to make up an answer. \n
# # If the question is irrelevant to the context, just tell "Please! make your question with relevant info". 
# #  Keep the answer as concise as possible. Always say "thanks for asking!" at the end of the answer. 
# #  use previous chat history to make answer.
# # current conversation: {chat_history}
# # {context}
# # Question: {question}
# # Helpful Answer:"""

# # QA_CHAIN_PROMPT = PromptTemplate.from_template(customs_template)

# # prompt = PromptTemplate(
# #     input_variables=["chat_history","question", "context"], template=customs_template
# # )

# # CUSTOM_QUESTION_PROMPT = PromptTemplate.from_template(custom_template)
# # # prompts = CUSTOM_QUESTION_PROMPT.format(chat_history="chat_history", question="question")
# # # prompt = PromptTemplate(
# # #     input_variables=["question"],
# # #     template=custom_template
# # # )

# # os.environ["OPENAI_API_KEY"] = "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"

# # llm = OpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key="OPEN_AI_KEY")

# # loaders = [PyPDFLoader("/home/frappe-shahadat/shuvo/frappe-bench/apps/hexbot/hexbot/public/file/Learning_Python.pdf")]

# # documents = []
# # for loader in loaders:
# #     documents.extend(loader.load())

# # # documents = loader.load()

# # chat_history = []


# # class getChat():
# #     def get_response(self, user_input):
# #         # response = llm(user_input)
# #         # response = prompt.format(user_question=user_input)
# #         chain = LLMChain(llm=llm, prompt=prompt)
# #         response = chain.run(user_input)
# #         return response
    
# #     def get_chat(self):
# #         # chat_history = [(query, result["answer"])]
# #         if not chat_history:
# #             return f'No history Yet'
# #         rlist = chat_history
# #         for exchange in chat_history:
# #             rlist.append(exchange)
# #         print(rlist)
# #         return rlist

# # class getChat2(param.Parameterized):
# #     chat_history = param.List([])

# #     def get_response(self, user_input):
# #          # split documents
# #         text_splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=30)
# #         docs = text_splitter.split_documents(documents)

# #         # define embedding
# #         embeddings = OpenAIEmbeddings()

# #         # create vector database from data
# #         db = DocArrayInMemorySearch.from_documents(docs, embeddings)

# #         memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
# #         # define retriever
# #         retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 4})

# #         # qa_chain = RetrievalQA.from_chain_type(
# #         #             llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0),
# #         #             retriever=retriever,
# #         #             memory=memory,
# #         #             # return_source_documents=True,
# #         #             chain_type_kwargs={"prompt": prompt}
# #         #         )

# #         qa = ConversationalRetrievalChain.from_llm(
# #                 llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0),
# #                 # chain_type="stuff",
# #                 retriever=retriever,
# #                 memory=memory,
# #                 # condense_question_prompt=promptss,
# #                 # return_source_documents=True,
# #                 # return_generated_question=True,
# #             )
# #         result = qa({"question": user_input, "chat_history": self.chat_history})
# #         # result = qa_chain({"query": user_input, "chat_history": chat_history})
# #         self.chat_history.extend([(user_input, result["answer"])])
# #         response = result['answer']
# #         response1 = result['chat_history']
# #         print(response)
# #         print(self.chat_history)
# #         # print(result["generated_question"])
# #         # print(result["source_documents"])
# #         return response
    
# #     def get_chats(self):
# #        if not self.chat_history:
# #            return f"This is empty"
# #        rlist= []
# #        for exchange in self.chat_history:
# #            rlist.append(str(exchange))
# #        return rlist
    
# #     def clr_history(self):
# #         self.chat_history = []
# #         return self.chat_history

# # # chats = getChat()
# # chats2 = getChat2()
# # # while True:
# # #     user_input = input("> ")
# # #     chats2.get_response(user_input)
# # @frappe.whitelist()
# # def get_chat_response(user_input):
# #     response = chats2.get_response(user_input)
# #     return response 


# # @frappe.whitelist()
# # def get_chat_history():
# #      getChats = chats2.get_chats()
# #      return getChats

# # # @frappe.whitelist()
# # # def clear_history():
# # #     clrHist = chats.clr_history()
# # #     return clrHist

# # import frappe 

# # # create a new document 

# # os.environ["OPENAI_API_KEY"] = "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"

# # prmpt = PromptTemplate.from_template("You are my {subject} instructor")
# chatprmpt = ChatPromptTemplate.from_messages([
#     (
#         "system", "You are a helpful assistant that translates {input_lang} to {output_lang}"
#     ),
#     (
#         "human", "{text}"
#     )
# ])
# # frmt_prmpt = prmpt.format(subject="Math")
# frmtd_chatprmpt = chatprmpt.format(
#     input_lang="English",
#     output_lang="hindi",
#     text="I love Islam"
# )
# llm = ChatOpenAI(openai_api_key="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")
# # response  = llm.invoke(frmtd_chatprmpt)
# # print(response.content)


# loader = PyPDFLoader('/home/frappe-shahadat/shuvo/frappe-bench/sites/myerp/public/files/constitution.pdf')
# documents = loader.load()
# # print(docs[0].page_content)
# text_splitter = RecursiveCharacterTextSplitter(
#     # Set a really small chunk size, just to show.
#     chunk_size=500,
#     chunk_overlap=100,
#     # length_function=len,
#     # is_separator_regex=False,
# )
# # texts = text_splitter.create_documents([documents[0].page_content])
# docs = text_splitter.split_documents(documents)
# embeddings = OpenAIEmbeddings(api_key="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")

# db = DocArrayInMemorySearch.from_documents(docs, embeddings)
# retriever = db.as_retriever()
# docs = retriever.invoke("What is water logging?")
# # query = "What is water logging?"
# # docs = db.similarity_search(query)
# # print(docs[0].page_content)
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# SYSTEM_TEMPLATE = """
# Answer the user's questions based on the below context. 
# If the context doesn't contain any relevant information to the question, don't make something up and just say "I don't know":

# <context>
# {context}
# </context>
# """

# question_answering_prompt = ChatPromptTemplate.from_messages(
#     [
#         (
#             "system",
#             SYSTEM_TEMPLATE,
#         ),
#         MessagesPlaceholder(variable_name="messages"),
#     ]
# )

# # document_chain = StuffDocumentsChain(llm, question_answering_prompt)
# llm_chain = LLMChain(llm=llm, prompt=prompt)
# document_chain = StuffDocumentsChain(
#     llm_chain=llm_chain,
#     document_prompt=question_answering_prompt,
#     document_variable_name=document_variable_name
# )
# # combine_docs_chain = StuffDocumentsChain(...)

# # This controls how the standalone question is generated.
# # Should take `chat_history` and `question` as input variables.
# template = (
#     "Combine the chat history and follow up question into "
#     "a standalone question. Chat History: {chat_history}"
#     "Follow up question: {question}"
# )
# prompt = PromptTemplate.from_template(template)
# llm = OpenAI(api_key="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")
# question_generator_chain = LLMChain(llm=llm, prompt=prompt)
# chain = ConversationalRetrievalChain(
#     combine_docs_chain=document_chain,
#     retriever=retriever,
#     question_generator=question_generator_chain,
# )
# put=chain.run("What is water logging")
# # print(put)

# class chat_gen():
#     def __init__(self):
#         self.chat_history=[]


#     def load_doc(self,document_path):
#         loader = PyPDFLoader(document_path)
#         documents = loader.load()
#         # Split document in chunks
#         text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=30, separator="\n")
#         docs = text_splitter.split_documents(documents=documents)
#         embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
#         # Create vectors
#         vectorstore = FAISS.from_documents(docs, embeddings)
#         # Persist the vectors locally on disk
#         vectorstore.save_local("faiss_index_datamodel")

#         # Load from local storage
#         persisted_vectorstore = FAISS.load_local("faiss_index_datamodel", embeddings)
#         return persisted_vectorstore


#     def load_model(self,):
#         llm = AzureChatOpenAI(deployment_name=DEPLOYEMENT_NAME,
#                             temperature=0.0,
#                             max_tokens=4000,
#                             )

#         # Define your system instruction
#         system_instruction = """ As an AI assistant, you must answer the query from the user from the retrieved content,
#         if no relavant information is available, answer the question by using your knowledge about the topic"""

#         # Define your template with the system instruction
#         template = (
#             f"{system_instruction} "
#             "Combine the chat history{chat_history} and follow up question into "
#             "a standalone question to answer from the {context}. "
#             "Follow up question: {question}"
#         )

#         prompt = PromptTemplate.from_template(template)
#         chain = ConversationalRetrievalChain.from_llm(
#             llm=llm,
#             retriever=self.load_doc("./data/snowflake_container.pdf").as_retriever(),
#             #condense_question_prompt=prompt,
#             combine_docs_chain_kwargs={'prompt': prompt},
#             chain_type="stuff",
#         )
#         return chain

#     def ask_pdf(self,query):
#         result = self.load_model()({"question":query,"chat_history": self.chat_history})
#         self.chat_history.append((query, result["answer"]))
#         #print(result)
#         return result['answer']