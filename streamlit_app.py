# https://github.com/avrabyt/MemoryBot/blob/main/memorybot.py

import streamlit as st
from langchain import ConversationChain, PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationEntityMemory, ConversationBufferMemory, CombinedMemory
from langchain.memory.prompt import ENTITY_MEMORY_CONVERSATION_TEMPLATE
from langchain.text_splitter import MarkdownHeaderTextSplitter

import prompts
import openai
import os
from langchain.vectorstores import Chroma
# ---------------------------------------------------

# Initialize session states
if "stored_session" not in st.session_state:
    st.session_state.stored_session = []


# Define function to start a new chat
def new_chat():
    """
    Clears session state and starts a new chat.
    """
    st.session_state.stored_session.append(st.session_state.messages)
    st.session_state.entity_memory.entity_store.clear()
    st.session_state.entity_memory.buffer.clear()
    st.session_state.messages = []


# Set up sidebar with various options
openai_api_key = st.sidebar.text_input(
        label="OpenAI API Key",
        type="password",
        value=st.session_state['OPENAI_API_KEY'] if 'OPENAI_API_KEY' in st.session_state else '',
        placeholder="sk-..."
        )
if openai_api_key:
    st.session_state['OPENAI_API_KEY'] = openai_api_key
    os.environ['OPENAI_API_KEY'] = openai_api_key
    openai.api_key = st.session_state['OPENAI_API_KEY']
else:
    st.error("Please add your OpenAI API key to continue.")
    st.info("Obtain your key from this link: https://platform.openai.com/account/api-keys")
    st.stop()

with st.sidebar.expander("🛠️ ", expanded=False):
    MODEL = st.selectbox(label='Model',
                         options=['gpt-4', 'gpt-3.5-turbo', 'text-davinci-003', 'text-davinci-002', 'code-davinci-002'])
    K = st.number_input(' (#)Summary of prompts to consider', min_value=3, max_value=1000)

# Add a button to start a new chat
st.sidebar.button("New Chat", on_click=new_chat, type='primary')

# Allow the user to clear all stored conversation sessions
if st.session_state.stored_session:
    if st.sidebar.checkbox("Clear-all"):
        st.session_state.stored_session = []
        st.session_state.stored_session.append(st.session_state.messages)
        st.session_state.entity_memory.entity_store.clear()
        st.session_state.entity_memory.buffer.clear()
        st.session_state.messages = []

# Set up the Streamlit app layout
st.title("️🐝 BeeBytes")
st.subheader("Lose No Insight, Gain Organization")

# ## AI MODEL


# Create an OpenAI instance
llm = ChatOpenAI(model_name=MODEL, temperature=0)

# Create a ConversationEntityMemory object if not already created
if 'entity_memory' not in st.session_state:
    st.session_state.entity_memory = ConversationEntityMemory(
        llm=llm,
        entity_extraction_prompt=prompts.ENTITY_EXTRACTION_PROMPT,
        entity_summarization_prompt=prompts.ENTITY_SUMMARIZATION_PROMPT,
    )

# Load existing vector database
#persist_directory = 'chroma/'
#embedding = OpenAIEmbeddings()

#vectordb = Chroma(
#    persist_directory=persist_directory,
#    embedding_function=embedding
#)

# read the text file into string
text_file = open("KnowledgeBase.txt", "r")  # open text file in read mode
KnowledgeBase = text_file.read()  # read whole file to a string
text_file.close()  # close file

headers_to_split_on = [
    ("#", "Headline")
]

markdown_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on
)
md_header_splits = markdown_splitter.split_text(KnowledgeBase)  # markdown header splits

# Create the ConversationChain object with the specified configuration
conversation = ConversationChain(
    llm=llm,
    verbose=True,
    prompt=prompts.ENTITY_MEMORY_CONVERSATION_PROMPT,
    memory=st.session_state.entity_memory
)

# ## FRONT INTERFACE


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if user_message := st.chat_input("Let's explore your issues?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(user_message)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_message})
    # Retrieve relevant knowledge from knowledge base
    # retrieved_knowledge_base = vectordb.max_marginal_relevance_search(user_message, k=1)
    # Create the response to user message
    # response = Conversation(
    #    {"input_documents": retrieved_knowledge_base, "human_input": user_message},
    #    return_only_outputs=True
    # )['output_text']
    response = conversation.run(user_message)
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

#st.sidebar.download_button(
#    label="Download the conversation",
#    data=st.session_state.entity_memory.buffer,
#    file_name='LeanBoost_ConversationHistory.txt',
#    mime='text',
#)

with st.sidebar.expander("📦 Your BeeBytes", expanded=False):
    number_of_entities = len(st.session_state.entity_memory.entity_store.store)
    st.info(f"You have {number_of_entities} BeeBytes.")
    #st.write(st.session_state.entity_memory.entity_store)
    for entity, description in st.session_state.entity_memory.entity_store.store.items():
        st.write(f"## {entity}: \n {description}")

    st.info("BeeBytes dictionary")
    st.write(st.session_state.entity_memory.entity_store)

with st.sidebar.expander("📚 Last stored conversation", expanded=False):
    st.header("Stored last conversation")
    # Display stored conversation sessions in the sidebar
    st.write(st.session_state.stored_session)

with st.sidebar.expander("Knowledge base", expanded=False):
    for i in range(len(md_header_splits)):
        st.header(md_header_splits[i].metadata['Headline'])
        st.write(md_header_splits[i].page_content)

st.sidebar.write('''<p style="font-size:10px; color:black;"><i>
<b>Disclaimer:</b> We do not take responsibility for any misuse or unintended consequences arising \
from the use of the results suggested by <u>BeeBytes</u>. Users are advised to exercise discretion and judgment while \
interpreting and using the results provided by the app. We are not liable for any consequences, whether direct or \
indirect, that may arise from the use or misuse of the app
</i></p>''', unsafe_allow_html=True)

st.sidebar.write('<p style="font-size:10px; color:black;">Powered by 🦜LangChain + OpenAI + Streamlit</p>', unsafe_allow_html=True)

