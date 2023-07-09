import os
import streamlit as st
from langchain.vectorstores import FAISS
from InstructorEmbedding import INSTRUCTOR
from langchain.schema import Document
from streamlit_chat import message
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.document_loaders import YoutubeLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from youtube_transcript_api import YouTubeTranscriptApi

os.environ["OPENAI_API_KEY"] = st.secrets["api_key"]

@st.cache_resource
def instructor_embeddings():
    instructor_embed = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-base", model_kwargs={"device": "cpu"})
    return instructor_embed

def youtube_loader(yt_link):
    loader = YoutubeLoader.from_youtube_url(yt_link, add_video_info=True)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    return texts

@st.cache_data
def ori_data(_texts,_embed):
    db = FAISS.from_documents(texts, embed)
    st.success('Database succussfully created!', icon="✅")
    return db

def history_db(_query):
    history = st.session_state.db.similarity_search(query)
    return history

def update_history_db(_query,_answer):
    text_01 = [Document(page_content=query, metadata=dict(page="Question"))]
    text_02 = [Document(page_content=answer, metadata=dict(page="Answer"))]
    database = FAISS.from_documents(text_01, embed)
    st.session_state.db.merge_from(database)
    database = FAISS.from_documents(text_02, embed)
    st.session_state.db.merge_from(database)
    return st.session_state.db

def last_3():
    docstore_list = list(st.session_state.db.docstore._dict.values())
    last_element = []
    if len(docstore_list) > 2:
        for i in range(len(docstore_list)-1, len(docstore_list)-4, -1):
            last_element.append(docstore_list[i])
    else:
        for i in range(len(docstore_list)):
            last_element.append(docstore_list[i])
    return last_element

embed = instructor_embeddings()

chat = ChatOpenAI(temperature=0.0)
template_format ="""
Information from the Database:
{database_data}

Information from Conversation History:
{history_data}

Last 3 Conversations:
{Conversation_1_to_3}

Query: {User_query}

Response:"""

prompt_template = ChatPromptTemplate.from_template(template_format)
query = ""

if 'db' not in st.session_state:
    st.session_state.db = FAISS.from_texts("a", embed)

if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []

if "bool" not in st.session_state:
    st.session_state["bool"] = True

if "new_db" not in st.session_state:
    st.session_state.new_db = None

text_input_container = st.empty()
yt_link = text_input_container.text_input("Enter Youtube link here!")

if yt_link != "":
    text_input_container.empty()
    st.video(yt_link)

#yt_link = st.text_input("Provide yt link")

if yt_link:
    if st.session_state["bool"]:
        st.session_state["bool"] = False
        texts = youtube_loader(yt_link)
        st.session_state.new_db = ori_data(embed,texts)
    query = st.text_input("Ask something?")

if query:
    docs = st.session_state.new_db.similarity_search(query,k=2)
    his_qu = history_db(query)
    last_conv = last_3()
    gen_messages = prompt_template.format_messages(database_data=docs,history_data=his_qu,Conversation_1_to_3=last_conv,User_query=query)
    response = chat(gen_messages)
    answer = response.content
    st.session_state.past.append(query)
    st.session_state.generated.append(answer)
    if st.session_state["generated"]:
        for i in range(len(st.session_state["generated"]) - 1, -1, -1):
            message(st.session_state["generated"][i],avatar_style="adventurer",seed=122, key=str(i))
            #message(st.session_state["source"][i], key=str(i+99))
            #message(st.session_state["time_sec"][i], key=str(i+999))
            message(st.session_state["past"][i], avatar_style="adventurer",seed=121, is_user=True, key=str(i+9999) + "_user")

    history = update_history_db(query,answer)


