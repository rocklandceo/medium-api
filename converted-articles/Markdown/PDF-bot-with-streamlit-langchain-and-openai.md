# PDF Based Chatbot Using Streamlit (🦜️🔗 LangChain, ⚙OpenAI)

Natural language processing refers to the branch of computer science and more specifically, the branch of artificial intelligence concerned with giving computers the ability to understand text and spoken words in much the same way human beings can.

**In this article, I am discussing PDF based Chatbot using streamlit (LangChain & OpenAI).**

![](https://miro.medium.com/1*agou53wb5a9LAjb8Uq6xzQ.png)

**First, briefly discuss about LangChain, Streamlit, LLM**

1. **LANGCHAIN:**

**LangChain** is a framework for developing applications powered by language models. We believe that the most powerful and differentiated applications will not only call out to a language model, but will also be:

1. **_Data-aware_:** connect a language model to other sources of data

2. **_Agentic_:** allow a language model to interact with its environment

![Credit: Online](https://miro.medium.com/1*bIeEo-QBjf6DHC8IOorlfw@2x.jpeg)

### Modules:

These modules are the core abstractions which we view as the building blocks of any LLM-powered application.

For each module LangChain provides standard, extendable interfaces. LangChain also provides external integrations and even end-to-end implementations for off-the-shelf use.

1. **[Models](https://github.com/VK-Ant/PDFBasedChatBot_Streamlit/blob/main/langchain_llm.ipynb)**: Supported model types and integrations

2. **[Prompts](https://github.com/VK-Ant/PDFBasedChatBot_Streamlit/blob/main/langchain_llm.ipynb)**: Prompt management, optimization, and serialization.

3. **[Memory](https://github.com/VK-Ant/PDFBasedChatBot_Streamlit/blob/main/langchain_llm.ipynb)**: Memory refers to state that is persisted between calls of a chain/agent.

4. **[Indexes](https://github.com/VK-Ant/PDFBasedChatBot_Streamlit/blob/main/langchain_llm.ipynb)**: Language models become much more powerful when combined with application-specific data - this module contains interfaces and integrations for loading, querying and updating external data.

5. **[Chains](https://github.com/VK-Ant/PDFBasedChatBot_Streamlit/blob/main/langchain_llm.ipynb)**: Chains are structured sequences of calls (to an LLM or to a different utility).

6. **[Agents](https://github.com/VK-Ant/PDFBasedChatBot_Streamlit/blob/main/langchain_llm.ipynb)**: An agent is a Chain in which an LLM, given a high-level directive and a set of tools, repeatedly decides an action, executes the action and observes the outcome until the high-level directive is complete.

7. **[Callbacks](https://github.com/VK-Ant/PDFBasedChatBot_Streamlit/blob/main/langchain_llm.ipynb)**: Callbacks let you log and stream the intermediate steps of any chain, making it easy to observe, debug, and evaluate the internals of an application.

The aforementioned models are all crucial to the Lang-chain architecture, and i have also experimented with each one. [Visit the link and investigate all the processes and gain a thorough understanding of all the ideas there.](https://github.com/VK-Ant/PDFBasedChatBot_Streamlit/blob/main/langchain_llm.ipynb)

**2. Large Language Model (LLM):**

A large language model (LLM) is a language model consisting of a neural network with many parameters (typically billions of weights or more), trained on large quantities of unlabeled text using self-supervised learning or semi-supervised learning.

Some of the most popular large language models are: GPT-3 (Generative Pretrained Transformer 3) - developed by OpenAI. BERT (Bidirectional Encoder Representations from Transformers) - developed by Google.

![](https://miro.medium.com/1*7DEgKHmuYmcAWFXHSyNWeA.png)

If you want more about LLM: [link](https://paperswithcode.com/paper/low-code-llm-visual-programming-over-llms)

**3. Streamlit:**

[Streamlit](https://www.streamlit.io/) is an open-source Python library that makes it easy to create and share beautiful, custom web apps for machine learning and data science. In just a few minutes you can build and deploy powerful data apps.

![Output](https://miro.medium.com/1*p8Z8dzs9a71lOxL3su6bAA.png)

Streamlit is a good web deployment tool that makes it easy to share projects and ideas with other data professionals. Only 10 lines of streamlit code make up the output shown above.

---

**OK, I think you guys understand the basic terms of our project. Now Step by step guidance of my project.**

**So, In this article, we are discussed about PDF based Chatbot using streamlit (LangChain & OpenAI.**

**What is PDF based Chatbot?**

Chat PDF is an artificial intelligence-based tool that provides users with a way to interact with their PDF files as if the information in these files was processed by a human being. The way it works is by analyzing the provided PDF file to give a summary or extract the necessary information.

**Block diagram:**

Block diagrams show how my chatbot functions on the front end, back end, and user

1. Question regarding the uploaded PDF's user agent
2. Fronend: Streamlit transmits the message to the backend system in front and back.
3. Backend: The user communicates with the backend via the frontend and provides input. The backend block uses OpenAI (LLM mode) and returns the results in user agent within a second.

![BlockDiagram](https://miro.medium.com/1*sco2HF3q3QdiiQ1YybQuvA.jpeg)

**Architecture Of Pdf based Chatbot:**

![Workflow of chatbot backend process](https://miro.medium.com/1*LaLK1PkmpGcBYZIHitvjMA.jpeg)

1. **Pdf Input file:** User-uploaded PDF document

**2. Extract context:** Computers analyze human spoken languages to extract meaningful insights. With NLP in data mining, computers can analyze text and voice data to derive meaningful insights.

![Output](https://miro.medium.com/1*MCUvRk1Avg6qYxCFAvBoqw.png)

**3. Text Chunks:** Chunking is defined as the process of natural language processing used to identify parts of speech and short phrases present in a given sentence.

![credit: Online](https://miro.medium.com/1*Pv3zWkAFXOIKOPA9nk6ACQ.jpeg)

![](https://miro.medium.com/1*O26HUQFAIOlni9MDtPZE9A.png)

![Output](https://miro.medium.com/1*PVZimeFhcyX12B_0fFRKUQ.png)

**4. Embedding (Vector store):**

A vector store is a particular type of database optimized for storing documents and their embeddings, and then fetching of the most relevant documents for a particular query, ie. those whose embeddings are most similar to the embedding of the query.

![Credit: Online](https://miro.medium.com/1*6T0bvF8ZQZpK0iGnn5_EyA.jpeg)

After the procedure is finished, create the.pickle file and save it to your documents.

**5. Build Semantic Index:** Latent semantic indexing (also referred to as Latent Semantic Analysis) is a method of analyzing a set of documents in order to discover statistical co-occurrences of words that appear together which then give insights into the topics of those words and documents.

**6. Knowledge Box:** Knowledge base question answering (KBQA) is an important task in natural language processing. Ex- isting methods for KBQA usually start with en- tity linking, which considers mostly named entities found in a question as the starting points in the KB to search for answers to the question.

Your PDF document's vector storage is a crucial component of the context saved in Knowledge Box.

**7. LLM Generator:** An LLM is a machine-learning neural network trained through data input/output sets; frequently, the text is unlabeled or uncategorized, and the model is using self-supervised or semi-supervised learning methodology. (OpenAI)

![Credit: Online](https://miro.medium.com/1*ztd9S6xZgrVeAPTmIp2dGg.png)

**8. Rank Result:** Compare the results from the LLM generator and view the top 5 results.

**9. Questions & Query Embedding:** User asks a question, which is embedded and provided by an LLM generator. User then provides an answer, ranks the results, and uploads a PDF document.

---

**Getting started with PDF based chatbot using Streamlit (OpenAI, LangChain):**

**PROJECT DESCRIPTION:**

1. Install requirement file.

2. Add your project folder to the.env folder you created (put your openai api).

3. Run the main file

4. Upload your pdf and summarize the main content of pdf

```python
#coding part
import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from PyPDF2 import PdfFileReader, PdfFileWriter,PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback

import pickle
import os
#load api key lib
from dotenv import load_dotenv
import base64


#Background images add function
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"jpeg"};base64,{encoded_string.decode()});
        background-size: cover;
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
#add_bg_from_local('images.jpeg')  

#sidebar contents

with st.sidebar:
    st.title('🦜️🔗 VK - PDF BASED LLM-LANGCHAIN CHATBOT🤗')
    st.markdown('''
    ## About APP:

    The app's primary resource is utilised to create::

    - [streamlit](https://streamlit.io/)
    - [Langchain](https://docs.langchain.com/docs/)
    - [OpenAI](https://openai.com/)

    ## About me:

    - [Linkedin](https://www.linkedin.com/in/venkat-vk/)
    
    ''')

    add_vertical_space(4)
    st.write('💡 All about pdf based chatbot, created by VK🤗')

load_dotenv()

def main():
    st.header("📄 Chat with your pdf file🤗")

    #upload a your pdf file
    pdf = st.file_uploader("Upload your PDF", type='pdf')
    st.write(pdf.name)

    if pdf is not None:
        pdf_reader = PdfReader(pdf)

        text = ""
        for page in pdf_reader.pages:
            text+= page.extract_text()

        #langchain_textspliter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 1000,
            chunk_overlap = 200,
            length_function = len
        )

        chunks = text_splitter.split_text(text=text)

        
        #store pdf name
        store_name = pdf.name[:-4]
        
        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl","rb") as f:
                vectorstore = pickle.load(f)
            #st.write("Already, Embeddings loaded from the your folder (disks)")
        else:
            #embedding (Openai methods) 
            embeddings = OpenAIEmbeddings()

            #Store the chunks part in db (vector)
            vectorstore = FAISS.from_texts(chunks,embedding=embeddings)

            with open(f"{store_name}.pkl","wb") as f:
                pickle.dump(vectorstore,f)
            
            #st.write("Embedding computation completed")

        #st.write(chunks)
        
        #Accept user questions/query

        query = st.text_input("Ask questions about related your upload pdf file")
        #st.write(query)

        if query:
            docs = vectorstore.similarity_search(query=query,k=3)
            #st.write(docs)
            
            #openai rank lnv process
            llm = OpenAI(temperature=0)
            chain = load_qa_chain(llm=llm, chain_type= "stuff")
            
            with get_openai_callback() as cb:
                response = chain.run(input_documents = docs, question = query)
                print(cb)
            st.write(response)



if __name__=="__main__":
    main()
```

![](https://miro.medium.com/1*z9vq_bdJVxPC5iCiRB5k_Q.png)

![Output](https://miro.medium.com/1*p8Z8dzs9a71lOxL3su6bAA.png)

This project's open source nature, one-month free use of the OpenAI API, and need to purchase an API key are its only drawbacks.
However, in my opinion, it's worthwhile.

---

**Github : [link](https://github.com/VK-Ant/PDFBasedChatBot_Streamlit)**

Thanks for visiting guys! **if you have any queries and error in this article comment it!** or lets connect[ linkedin](https://www.linkedin.com/in/venkat-vk), [kaggle](https://www.kaggle.com/venkatkumar001) discussion.

🦜️🔗 F**ull credit:**

1. [https://youtu.be/RIWbalZ7sTo](https://youtu.be/RIWbalZ7sTo)

2. [Streamlit](https://streamlit.io/), [Langchain](https://python.langchain.com/en/latest/index.html), [OpenAI ](https://platform.openai.com/auth/callback?code=DeIQUnJZldVLT_ObESOcQMqWS5YOmzz8AgSN1-DCaq_dk&state=bzVPUS1nTzlEQ0lDMHdIM2IzY1F3LU5Fb0ZvY0YyWkRuM0VJMkJIaHpmUg%3D%3D)community

3. [Venkatesan](https://www.linkedin.com/in/venkatesan-d-22aa42256/) (My colleague)

🦜️🔗 S_**ocial network : l[inkedin,](https://www.linkedin.com/in/venkat-vk/) K[aggle,](https://www.kaggle.com/venkatkumar001) G[ithub](https://github.com/VK-Ant)**_