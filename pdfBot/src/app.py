import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_core.messages import AIMessage, HumanMessage
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
import os
import time


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(raw_text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(raw_text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()

    # Integrate with Hugging Face local model
    # model_name = "hkunlp/instructor-large"
    # model_kwargs = {'device': 'cpu'}
    # encode_kwargs = {'normalize_embeddings': True}
    # embeddings = HuggingFaceInstructEmbeddings(
    #     model_name=model_name,
    #     model_kwargs=model_kwargs,
    #     encode_kwargs=encode_kwargs
    # )

    # Connect to Pinecone Vector DB
    pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
    index_name = "pdf-bot"
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        while not pc.describe_index(index_name).status["ready"]:
            time.sleep(1)
    else:
        index = pc.Index(index_name)

    vectorstore = PineconeVectorStore.from_texts(
        texts=text_chunks,
        embedding=embeddings,
        index_name=index_name,
        namespace="mongo-apis")
    return vectorstore


def get_context_retriever_chain(vectorstore):
    llm = ChatOpenAI()

    retriever = vectorstore.as_retriever()

    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a search query to look up in order"
                 " to get information relevant to the conversation.")
    ])

    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)

    return retriever_chain


def get_conversation_chain(retriever_chain):

    llm = ChatOpenAI()

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's questions based on the below context:\n\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])

    stuff_docs_chain = create_stuff_documents_chain(llm, prompt)

    return create_retrieval_chain(retriever_chain, stuff_docs_chain)


# def get_conversation_chain(vectorstore):
#     llm = ChatOpenAI()
#     llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})
#
#     memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
#
#     conversation_chain = ConversationalRetrievalChain.from_llm(
#         llm=llm,
#         retriever=vectorstore.as_retriever(),
#         memory=memory
#     )
#     return conversation_chain


def get_response(user_input):
    # create conversational chain
    retriever_chain = get_context_retriever_chain(st.session_state.vectorstore)
    conversation_chain = get_conversation_chain(retriever_chain)

    llm_response = conversation_chain.invoke({
        "chat_history": st.session_state.chat_history,
        "input": user_input,
    })
    return llm_response['answer']


def main():
    load_dotenv()
    st.set_page_config(page_title="PDFs ChatBot", page_icon=":books:", layout="wide")

    # st.write(css, unsafe_allow_html=True)

    with st.sidebar:
        # session state
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = [
                AIMessage(content="Hello, I am a Bot. How can I help you?"),
            ]
        if "text_chunks" not in st.session_state:
            st.session_state.text_chunks = []
        st.subheader("Your Documents")
        pdf_docs = st.file_uploader(
            label="Upload your PDFs here",  accept_multiple_files=True, type=["pdf"])

        if st.button("Upload & Process"):
            with st.spinner("Processing..."):
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)
                # st.write(raw_text)

                # get the text chunks
                st.session_state.text_chunks = get_text_chunks(raw_text)
                # st.write(text_chunks)

                # create vector store
                #if "vectorstore" not in st.session_state:
                st.session_state.vectorstore = get_vectorstore(st.session_state.text_chunks)

    # create conversation chain
    # st.session_state.conversation = get_conversation_chain(vectorstore)
    st.header("PDFs ChatBot :books:")
    st.text_input("Ask a question about your documents:")

    user_question = st.chat_input("What's your question?")
    if user_question is not None and user_question != "":
        response = get_response(user_question)
        st.session_state.chat_history.append(HumanMessage(content=user_question))
        st.session_state.chat_history.append(AIMessage(content=response))

    # Conversation
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
                # st.balloons()
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)



    # session state
    # if "chat_history" not in st.session_state:
    #     st.session_state.chat_history = [
    #         AIMessage(content="Hello, I am a Bot. How can I help you?"),
    #     ]

    # if "conversation" not in st.session_state:
    #     st.session_state.conversation = None
    #
    # st.header("PDFs ChatBot :books:")
    # # st.text_input("Ask a question about you documents:", key="question")
    # user_question = st.text_input("Ask a question about your documents:")
    # if user_question:
    #     handle_userinput(user_question)

    # st.write(user_template.replace("{{MSG}}", "Hello ChatBot"), unsafe_allow_html=True)
    # st.write(bot_template.replace("{{MSG}}", "Hello Human"), unsafe_allow_html=True)

    # with st.sidebar:
    #     st.subheader("Your Documents")
    #     pdf_docs = st.file_uploader(
    #         label="Upload your PDFs here",  accept_multiple_files=True, type=["pdf"])
    #     if st.button("Upload & Process"):
    #         with st.spinner("Processing..."):
    #             # get pdf text
    #             raw_text = get_pdf_text(pdf_docs)
    #             # st.write(raw_text)
    #
    #             # get the text chunks
    #             text_chunks = get_text_chunks(raw_text)
    #             # st.write(text_chunks)
    #
    #             # create vector store
    #             vectorstore = get_vectorstore(text_chunks)

                # create conversation chain
                # st.session_state.conversation = get_conversation_chain(vectorstore)



if __name__ == '__main__':
    main()
