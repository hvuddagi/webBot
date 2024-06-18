# pip install streamlit langchain lanchain-openai beautifulsoap4 weaviate

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import weaviate
from langchain_community.vectorstores.weaviate import Weaviate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_weaviate.vectorstores import WeaviateVectorStore
from weaviate.connect import ConnectionParams
from weaviate.classes.init import AdditionalConfig, Timeout
from dotenv import load_dotenv

load_dotenv()


# Website Loading and Embedding
def get_vectorstore_from_url(url):
    # get the text document form
    loader = WebBaseLoader(url)
    document = loader.load()

    # Split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter()
    document_chunks = text_splitter.split_documents(document)

    # Create a vectorstore from the chunks
    client = weaviate.WeaviateClient(
        connection_params=ConnectionParams.from_params(
            http_host="weaviatedb-rcdn-dev.cisco.com",
            http_port=6380,
            http_secure=True,
            grpc_host="weaviatedb-rcdn-dev-grpc.cisco.com",
            grpc_port=6381,
            grpc_secure=True
        ),
        auth_client_secret=weaviate.auth.AuthApiKey("wqI35r7ZR3w3"),
        additional_config=AdditionalConfig(
            timeout=Timeout(init=10, query=45, insert=120),
            skip_init_checks=True
        )
    )
    client.connect()

    # client = weaviate.Client(url="https://weaviatedb-rcdn-dev.cisco.com:6380",
    #                          auth_client_secret=weaviate.auth.AuthApiKey(api_key="wqI35r7ZR3w3")
    #                         )

    # vector_store_db = Weaviate(client, index_name="webvector", text_key="content", embedding=OpenAIEmbeddings())

    vector_store = WeaviateVectorStore.from_documents(document_chunks, OpenAIEmbeddings(), client=client)
    # vector_store = vector_store_db.from_documents(document_chunks, OpenAIEmbeddings())
    return vector_store


def get_context_retriever_chain(vector_store):
    llm = ChatOpenAI()

    retriever = vector_store.as_retriever()

    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a search query to look up in order"
                 " to get information relevant to the conversation.")
    ])

    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)

    return retriever_chain


def get_conversational_rag_chain(retriever_chain):

    llm = ChatOpenAI()

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's questions based on the below context:\n\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])

    stuff_docs_chain = create_stuff_documents_chain(llm, prompt)

    return create_retrieval_chain(retriever_chain, stuff_docs_chain)


def get_response(user_input):
    # create conversational chain
    retriever_chain = get_context_retriever_chain(st.session_state.vector_store)
    conversation_rag_chain = get_conversational_rag_chain(retriever_chain)

    llm_response = conversation_rag_chain.invoke({
        "chat_history": st.session_state.chat_history,
        "input": user_query,
    })

    return llm_response['answer']


# app config
st.set_page_config(page_title="Chat with Web Bot", page_icon="favicon.ico", layout="wide")
st.title("Chat with Web Bot")


# sidebar
with st.sidebar:
    st.header("Settings")
    website_url = st.text_input("Website URL")

if website_url is None or website_url == "":
    st.info("Please enter a Website URL")

else:
    # session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(content="Hello, I am a Bot. How can I help you?"),
        ]

    if "vector_store" not in st.session_state:
        st.session_state.vector_store = get_vectorstore_from_url(website_url)

    # with st.sidebar:
    #     st.write(documents)

    # User Input
    user_query = st.chat_input("Type your message here....")
    if user_query is not None and user_query != "":
        response = get_response(user_query)
        # st.write(response)
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        st.session_state.chat_history.append(AIMessage(content=response))

        # Testing to see if documents are retrieved correctly !!!
        # retrieved_documents = retriever_chain.invoke({
        #     "chat_history": st.session_state.chat_history,
        #     "input": user_query
        # })
        # st.write(retrieved_documents)

    # Display the messages in Sidebar as in debug mode
    # with st.sidebar:
    #     st.write(st.session_state.chat_history)

    # Conversation
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
                # st.balloons()
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)
                # st.balloons()
    # client.close()
