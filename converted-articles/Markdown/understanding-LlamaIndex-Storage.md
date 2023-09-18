# LlamaIndex: Comprehensive guide on storage

Index store, vector store or embedding store and document store.

In my third post of the PDF AI Chatbot building series, my intention was to dive deeper into the practical implementation discussed in part 2. However, as I continue writing, I realize the crucial significance of having a reliable storage system for your application. Unfortunately, when it comes to building an LLM application, the importance of storage amplifies. Not only do you need to handle traditional storage options like SQL or NoSQL databases and file storage systems, but you also have to tackle emerging types of storage such as index storage and vector storage.

![LlamaIndex](https://miro.medium.com/0*1OVaDQHr25PTKs-R.png)

And I bet you've gone through a heap of tutorials on the internet. You may find that a large portion of the tutorials out there on building LLM applications with Langchain or LlamaIndex primarily cover the basics and often neglect the vital topic of storage. These tutorials typically provide simple instructions without delving into the details of storage management, which can be frustrating for developers. As a result, many aspiring LLM application builders are left without a clear understanding of how to handle storage effectively.

If this is your [first](https://medium.com/how-ai-built-this/zero-to-one-a-guide-to-building-a-first-pdf-chatbot-with-langchain-llamaindex-part-1-7d0e9c0d62f) time reading this article, you can check my **first** and **[second](https://medium.com/@ryanntk/llamaindex-how-to-use-index-correctly-6f928b8944c6)** posts of this series. And also if you are unfamiliar with some data storage such as Pinecone, Chroma, DeepLake, and Weaviate, you can skip this section and go straight to the section build section (What we will do in this post). Otherwise, I would think the following content will certainly help you put more into perspective and understand more about the components of the LLM app and why we need such new storage.

## The Significance of Storage in LLM Applications

### Why Storage is Crucial in LLM Applications

When it comes to building LLM (Language Model) applications, the importance of a reliable storage system cannot be overstated. LLM applications heavily rely on data, both for training the models and for serving predictions in real time. Therefore, having a robust and efficient storage system is crucial for the overall performance and scalability of your application.

1. **Training Data Storage**: During the training phase of an LLM application, large amounts of data are used to train the language models. This data may include text corpora, labelled datasets, and even pre-trained models. The storage system needs to handle the vast volume of training data and provide efficient access and retrieval capabilities. Whether you are working with SQL or NoSQL databases, or even distributed file systems, the storage infrastructure must be able to handle the data processing requirements of training LLM models effectively.

2. **Real-Time Prediction Storage**: Once the LLM models are trained, they are deployed to serve predictions in real time. This requires storing the models, as well as any associated metadata or configuration files, in a way that enables fast and efficient retrieval. Additionally, the storage system must support high concurrency and low-latency access, as LLM applications often experience a large number of incoming requests simultaneously. It is essential to choose a storage solution that can handle the scale and demands of real-time prediction serving.

3. **Emerging Storage Types**: As LLM applications evolve, new types of storage are emerging to address specific requirements. Two notable examples are index storage and vector storage.

- **Index Storage**: LLM models often require indexing mechanisms to optimize search and retrieval operations. Index storage systems, such as Langchain or LlamaIndex, provide efficient indexing capabilities specifically designed for LLM applications. These systems enable faster query processing and facilitate advanced search functionalities.

- **Vector Storage**: LLM models often represent text inputs as dense vectors in high-dimensional spaces. Efficient storage and retrieval of these vectors are critical for similarity search, clustering, and other advanced operations. Vector storage systems, like VectorDB or DeepStorage, offer specialized support for storing and querying high-dimensional vectors, enabling efficient processing of LLM-related tasks.

So, the storage system for an LLM application must handle the storage and retrieval of training data, facilitate real-time prediction serving with low-latency access, and potentially incorporate emerging storage types such as index storage and vector storage. We won't touch into section 1 as it is not our primary concern, but our focus will be more on introducing new storage systems

### Challenges in Handling Storage for LLM Applications

Building an LLM (Language Model) application introduces unique challenges when it comes to handling storage. Let's explore the difficulties encountered in managing storage for LLM applications, considering both traditional and emerging storage options.

**Traditional Storage Options**: LLM applications often require handling large volumes of data, both during training and real-time prediction serving. Traditional storage options like SQL and NoSQL databases, as well as file storage systems, present their own challenges:

- **Scalability**: As the size of the data grows, traditional storage systems may struggle to handle the increasing demands of LLM applications, resulting in performance bottlenecks and slower query processing.

- **Flexibility**: Adapting traditional storage systems to the specific needs of LLM applications, such as efficient indexing or managing high-dimensional vectors, can be complex and require additional customizations.

- **Data Integrity**: Ensuring data integrity and consistency can be challenging in distributed environments, where multiple instances of LLM models may be running simultaneously and accessing the storage system concurrently.

**Emerging Storage Types**: LLM applications have also seen the emergence of specialized storage options that address specific requirements:

- Index Storage: LLM models often benefit from indexing mechanisms to optimize search and retrieval operations. Index storage systems can be any fast storage such as MongoDB, S3, and Azure Blob offer efficient indexing capabilities designed specifically for LLM applications. However, integrating and managing these specialized storage solutions can pose its own set of challenges.

- Vector Storage: LLM models often represent text inputs as high-dimensional vectors. Efficient storage and retrieval of these vectors are crucial for similarity search, clustering, and other advanced operations. Vector storage systems, like ChromaDB or Pinecone, provide specialized support for storing and querying high-dimensional vectors. However, efficiently managing and querying these vectors can be complex, requiring careful consideration of indexing and computational requirements.

I am confident that is more than enough on why we need a new kind of storage for the LLM app. By now, you only need to remember two points

- Traditional storage like SQL or NoSQL is not efficient to perform searches on large chunks of text with similar meanings.

- Vector Storage such as ChromaDB or Pincone will store your embedding data and Index Storage will be used to store indexing of those embeddings data

Next, we will explore two approaches when coming to select storage. You can use local storage such as file system or S3 for your MVP or you can go with cloud storage to make your app more scalable and flexible.

## What we will do in this post

Simple enough with Jupyter Notebook and a few text files. We will demonstrate how we use different types of storage the LlamaIndex offer on local storage and cloud storage.

Once you've gone through this section, you'll be well-prepared to implement storage features in your LLM application. In the subsequent post, we'll delve into part 2, so kindly exercise some patience.

### Local Storage

```python
import logging
import sys
import os
os.environ["OPENAI_API_KEY"] = "<YOUR OPENAI API KEY>"
```

We won't use OpenAI API for embedding in this tutorial but feel free to use OpenAI's embedding model if you want. I don't want to break my bank account so I will go with HuggingFace's embedding.

Here is the folder structure

```
--PDF_Chatbot/
----data/
------AWS_Well-Architected_Framework.pdf
------Apple-10k-Q1-2023.pdf
------paul_graham_essay.txt
----storage/
------index_storage/
------vector_storage/
------document_storage/
----notebook.ipynb
```

**Local Storage with Chroma**

```python
from langchain.embeddings import HuggingFaceEmbeddings
from llama_index import LangchainEmbedding, ServiceContext, StorageContext, download_loader, LLMPredictor
from langchain.chat_models import ChatOpenAI
from llama_index.vector_stores import ChromaVectorStore

import chromadb
from chromadb.config import Settings

# init Chroma collection
chroma_client = chromadb.Client(
          Settings(chroma_db_impl="duckdb+parquet",
           persist_directory="./storage/vector_storage/chromadb/"
   ))

## create collection
chroma_collection = chroma_client.create_collection("apple_10k_report")
```

Initialize storage context and service context.

```python
## I use OpenAI ChatAPT as LLM Model. This will cost you money
llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0.2, max_tokens=512, model_name='gpt-3.5-turbo'))

## by default, LlamIndex use OpenAI's embedding, we will use HuggingFace's embedding instead
embed_model = LangchainEmbedding(HuggingFaceEmbeddings())

## init ChromaVector storage for storage context
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

## init service context
service_context = ServiceContext.from_defaults(
      llm_predictor=llm_predictor,
      embed_model=embed_model
)
```

Now, we will load a single file and store it in our local storage

```python
from llama_index import GPTVectorStoreIndex, SimpleDirectoryReader

# load document
documents = SimpleDirectoryReader(input_files='./data/Apple-10k-Q1-2023.pdf').load_data()
# use GPTVectorStoreIndex, it will call embedding mododel and store the 
# vector data (embedding data) in the your storage folder
index = GPTVectorStoreIndex.from_documents(documents=documents, 
                                           storage_context=storage_context,
                                           service_context=service_context)
```

You will see that there is new folder "**chromadb**" is created in vector_storage folder. This is embedding data that we produce via index construction.

![](https://miro.medium.com/1*OdhF22FK0VMj7xhXCwPl0Q.png)

Because you are working in local storage, the index won't be saved automatically, run the following shell to save the index.

```bash
## save index
index.set_index_id("gptvector_apple_finance")
index.storage_context.persist('./storage/index_storage/apple/')
```

![](https://miro.medium.com/1*KYgsn-4ZeWoiBU419WOALw.png)

Easy, isn't it?

Now, if you terminate the Jupyter Notebook and run it again, all the in-memory data will be lost. To avoid the index construction process, you can now just need to reload whatever you have already saved in your local storage.

```python
## load index
from llama_index import load_index_from_storage
from llama_index.vector_stores import ChromaVectorStore
from llama_index.storage.index_store import SimpleIndexStore

## create ChromaClient again
chroma_client = chromadb.Client(
        Settings(chroma_db_impl="duckdb+parquet",
                 persist_directory="./storage/vector_storage/chromadb/"
        ))

# load the collection
collection = chroma_client.get_collection("apple_10k_report")

## construct storage context
load_storage_context = StorageContext.from_defaults(
    vector_store=ChromaVectorStore(chroma_collection=collection),
    index_store=SimpleIndexStore.from_persist_dir(persist_dir="./storage/index_storage/apple/"),
)

## init LLM Model
llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0.2, max_tokens=512, model_name='gpt-3.5-turbo'))

## init embedding model
embed_model = LangchainEmbedding(HuggingFaceEmbeddings())

## construct service context
load_service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor,embed_model=embed_model)

## finally to load the index
load_index = load_index_from_storage(service_context=load_service_context, 
                                     storage_context=load_storage_context)
```

Alrightly, you now can query index like normal

```python
query = load_index.as_query_engine()
query.query("What is the operating income of Q1 2023?")
```

![](https://miro.medium.com/1*G92Hry37NtI5nEv5zNHY2w.png)

For an in-depth understanding of ChromaDB, please refer to its official website located at [here](https://www.trychroma.com/). In essence, ChromaDB stands as a nimble and robust vector database tailored specifically for AI-driven applications.

Beyond its role as a vector database, ChromaDB empowers users to conduct similarity searches within the database, enabling retrieval of pertinent information pertaining to their queries. Furthermore, thanks to its open-source nature, ChromaDB facilitates the deployment of personal vector database servers on either local machines or cloud-based platforms.

### Cloud Storage

Up to this point, our focus has been on local storage for vector data (ChromaDB) and index storage. However, for long-term sustainability and global accessibility of your LLM application, it is crucial to implement a solution that enables worldwide access.

The most straightforward and effective approach is transitioning from local storage to cloud storage options like AWS S3, Azure Blob, or GCP Storage. The process of making this transition is quite evident. Instead of storing index and vector data on a local machine, these data will be stored in cloud storage, and retrieval of the index and embedded data will be performed on demand.

To accomplish this, you simply need to incorporate and modify the following code accordingly.

```python
import s3fs

# set up s3fs
AWS_KEY = os.environ['AWS_ACCESS_KEY_ID']
AWS_SECRET = os.environ['AWS_SECRET_ACCESS_KEY']
R2_ACCOUNT_ID = os.environ['R2_ACCOUNT_ID']

assert AWS_KEY is not None and AWS_KEY != ""

s3 = s3fs.S3FileSystem(
   key=AWS_KEY,
   secret=AWS_SECRET,
   endpoint_url=f'https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com',
   s3_additional_kwargs={'ACL': 'public-read'}
)

# this is {bucket_name}/{index_name}
index.storage_context.persist('llama-index/storage_demo', fs=s3)
# load index from s3
sc = StorageContext.from_defaults(persist_dir='llama-index/storage_demo', fs=s3)
```

> **BUT WHERE IS VECTOR DATA?**

My point exactly, we will employ the recommended combination of MongoDB as our primary data storage for indexing, and other hosted vector stores to accommodate our embedding data.

If you are not yet familiar with MongoDB, it was once a highly sought-after document database known for its scalability and flexibility in terms of querying and indexing. While it remains a popular choice today, it faces increasing competition from other players in the market.

In place of Chroma, we will utilize Pinecone as our vector data storage solution.

Without further ado, let's commence the implementation process.

**Firstly**, please proceed with signing up for [MongoDB](https://www.mongodb.com/cloud/atlas/signup). Rest assured, the free version offered by MongoDB will be more than sufficient for the purposes of this guide. Payment is only required if you decide to build your own application in the future.

**Next**, complete the signup process for [PineconeDB](https://www.pinecone.io/). Similarly, there is no need to make any payments as the free version adequately caters to most requirements. However, if you require rapid development and scalability, you may consider opting for paid services.

Once you have successfully completed the signup process for both MongoDB and PineconeDB, we can proceed to the next step: installing the necessary MongoDB and Pinecone libraries.

Let's get started.

```python
!pip install pymongo
!pip install pinecone-client
```

Set thing up

Find the Pinecone API key in your Pinecone dashboard.

![](https://miro.medium.com/1*frx-BedutFa11qjJx9O7_g.png)

```python
import logging
import sys
import os
os.environ["OPENAI_API_KEY"] = "<YOUR OPENAI API KEY>"
os.environ["PINECONE_API_KEY"] = "<YOUR PINECONE API KEY>"

api_key = os.environ['PINECONE_API_KEY']
## if you are using free version, then it is probably use us-central1-gcp
pinecone.init(api_key=api_key, environment="us-central1-gcp")
```

Now, let's create the Pinecone index

```python
## creating index
pinecone.create_index("quickstart", 
          dimension=768,
          metric="euclidean", 
          pod_type="p1")
pinecone_index = pinecone.Index("quickstart")
```

So why do we put **dimension=768**, it is not an arbitrary number. The 768 is actually the dimension of Huggingface embedding. If you are using OpenAI's embedding model `text-embedding-ada-002` then the dimension should be **1536.**

It will take time to create a new index, after a couple of minutes, you will see this pop up in your pinecone dashboard.

![](https://miro.medium.com/1*nl67v2HEwzBmDB99MvioqA.png)

Set a few things up

```python
from llama_index import SimpleDirectoryReader, LLMPredictor, ServiceContext, StorageContext
from llama_index.indices.document_summary import GPTDocumentSummaryIndex
from llama_index import GPTVectorStoreIndex, GPTListIndex, GPTTreeIndex
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import HuggingFaceEmbeddings, SentenceTransformerEmbeddings

from llama_index import ResponseSynthesizer
from llama_index.vector_stores import PineconeVectorStore
from llama_index.storage.index_store import MongoIndexStore
```

load the documents

```python
docs = ['Apple-10k-Q1-2023.pdf']

docs_loader = []
for d in docs:
    doc = SimpleDirectoryReader(input_files=[f"./data/{d}"]).load_data()
    doc[0].doc_id = d
```

Setup basic contexts:

```python
llm_predictor_chat = LLMPredictor(llm=ChatOpenAI(temperature=0.2, model_name="gpt-3.5-turbo"))
vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
embed_model = LangchainEmbedding(HuggingFaceEmbeddings())
# create (or load) index store
index_store = MongoIndexStore.from_uri(
      uri="mongodb+srv://<your_mongodb_username>:<your_mongodb_password>@<your_database>.<your_mongodb_server>/?retryWrites=true&w=majority",
                                       db_name="<your_database>")
storage_context = StorageContext.from_defaults(index_store=index_store, vector_store=vector_store)
service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor_chat, embed_model=embed_model)
```

Now, let's build the first index

```python
doc_summary_index = GPTVectorStoreIndex.from_documents(
    alldocs, 
    service_context=service_context,
    storage_context=storage_context,
)
doc_summary_index.set_index_id("apple_Q1_2023_index")
```

After running the previous shell, two things will happen

- There will be a new index created in MongoDB

![](https://miro.medium.com/1*lF7HHGi66uPwhxnadGDNgg.png)

- And there will be new vector data stored on Pinecone

![](https://miro.medium.com/1*GgstgPwLdF-QRL2TYYQm5w.png)

How do we retrieve the data now?

```python
## load index
from llama_index import load_index_from_storage

pinecone_index_load = pinecone.Index("quickstart")
index_store_load = MongoIndexStore.from_uri(db_name="<your_database>",
                                            uri="mongodb+srv://<your_mongodb_username>:<your_mongodb_password>@<your_database>.<your_mongodb_server>/?retryWrites=true&w=majority",)
load_storage_context = StorageContext.from_defaults(
    vector_store=PineconeVectorStore(pinecone_index=pinecone_index_load),
    index_store=index_store_load,
)
llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0.2, max_tokens=512, model_name='gpt-3.5-turbo'))
embed_model = LangchainEmbedding(HuggingFaceEmbeddings())
load_service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor,embed_model=embed_model)

load_index = load_index_from_storage(service_context=load_service_context, 
                                     storage_context=load_storage_context,
                                    index_id='apple_Q1_2023_index')

## and perform the query
query = load_index.as_query_engine()
query.query("What is the operating income of Q1 2023?")
```

---

That is it, friends, By now, you have gained a comprehensive understanding of LlamaIndex storage and its utilization with various cutting-edge database systems. I trust that this guide has been presented in a straightforward and lucid manner, enabling you to comprehend the fundamental building blocks necessary for developing a resilient and scalable LLM (LlamaLink Manager) application.

While we have covered a significant portion, there are still several avenues left to explore. Similar to the courses you have undertaken during your academic journey, it is advisable to delve deeper into these topics on your own. Remember, the most effective way to learn is through active engagement, questioning, and discovering answers to your inquiries. This guide serves as a launching pad for you to embark on an expedited exploration of these new database systems.

If you have any questions or require further clarification, please feel free to leave a comment below. Additionally, if you found this article helpful, show your appreciation with a round of applause. Be sure to follow me, as I will be covering more valuable content related to the foundational elements of the LLM app in future posts.

Here are a few topics for you to delve into:

1. How can we update the vector database? Similar to traditional SQL databases, what methods can we employ to add, update, and delete entries within vector databases such as Pinecone and ChromaDB?

2. How do we organize indices? Given the multitude of index types available, what strategies can we employ to structure indices within MongoDB for a chatbot application that deals with documents?

Enjoy your exploration, and may your journey be filled with enlightening discoveries!

> As always, stay curious and keep learning. Happy coding.

## Reference List:

**MongoDB**: [https://www.mongodb.com/](https://www.mongodb.com/)

**ChromaDB**: [https://docs.trychroma.com/api-reference](https://docs.trychroma.com/api-reference)

**Pinecone**: [https://docs.pinecone.io/docs/overview](https://docs.pinecone.io/docs/overview)

**LlamaIndex storage**: [https://gpt-index.readthedocs.io/en/latest/how_to/storage.html](https://gpt-index.readthedocs.io/en/latest/how_to/storage.html)

**Index structure**: [https://gpt-index.readthedocs.io/en/latest/how_to/indices.html](https://gpt-index.readthedocs.io/en/latest/how_to/indices.html)

**Reach out** to me on my LinkedIn:[https://www.linkedin.com/in/ryan-nguyen-abb844a4/](https://www.linkedin.com/in/ryan-nguyen-abb844a4/)