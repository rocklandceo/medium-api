# Remembering Conversations: Building Chatbots with Short and Long-Term Memory on AWS

> **TLDR:** This article is a comprehensive guide on building a fully functional domain-specific chatbot that incorporates both short and long-term memory, emulating human cognition. To achieve this, we will leverage Foundation models from SageMaker JumpStart and utilize various AWS services such as DynamoDB, OpenSearch, and Lambda, etc. By combining all of these components, we will develop a cognitive architecture that mimics the memory capabilities of a human brain. The article will serve as a valuable resource, providing step-by-step instructions on setting up the necessary AWS components to harness the power of large language models (LLMs). By employing retrieval-augmented generation (RAG), our chatbot will be able to provide factually verified responses and efficiently access domain-specific data. Furthermore, it will possess the ability to remember information throughout conversations and recall past interactions. To facilitate your learning experience, we have included accompanying [notebooks](https://github.com/arunprsh/aws-sagemaker-chatbot-memory.git) and all the essential materials required for you to build and execute this architecture from scratch. This work builds upon our [previous exploration](https://medium.com/@shankar.arunp/augmenting-large-language-models-with-verified-information-sources-leveraging-aws-sagemaker-and-f6be17fb10a8) of RAG, taking it a step further. In addition, we will discuss various architecture enhancements that can promote effective design and underscore the significance of memory in LLMs. By drawing comparisons with human cognition, we will highlight the critical role memory plays in the chatbot's performance.

![](https://miro.medium.com/1*f4ykS5NS743HSzOYj9KZsw.jpeg)

Generative AI, an emerging subfield of artificial intelligence, employs cutting-edge machine learning methodologies to revolutionize various domains, instigating profound disruptions with its content generation capabilities. From creating realistic images, composing music, writing articles, to designing pharmaceutical drugs, the capabilities of generative AI are proving to be game-changing.

One of the most prominent applications of this technology lies in the field of Natural Language Processing (NLP) with chatbots standing at the epicenter of this revolution. Chatbots, powered by generative AI, are redefining human-computer interaction. They are no longer limited to predefined templates and rules but can now understand context, generate human-like responses, and learn from past interactions, thereby enriching the quality of engagement. They're central to improving customer service, personalizing user experiences, automating tasks, and transforming business processes across sectors. Consequently, generative AI's role in chatbots' advancement is significant, marking a defining moment in the evolution of NLP and its applications.

---

## The Cognitive Tapestry: Unraveling Human Memory

Memory is a multifaceted and ever-evolving process that involves the continuous _encoding_, _storage_, and _retrieval_ of information. It plays a crucial role in our everyday experiences, aiding our comprehension of the world and facilitating our interactions within it. Understanding the formation and transition of memories, especially from short-term to long-term, is a captivating subject of investigation in the fields of Neuroscience and Cognitive Science. Although many aspects of this mechanism remain subjects of exploration, substantial progress has been made in identifying the critical brain regions involved in these processes.

![Anatomy of Human Brain](https://miro.medium.com/1*qRVc8e5VTH_7JAtFr4drFQ.png)

Research has identified three key brain regions that play a significant role in the formation, storage, and retrieval of memories: the **prefrontal cortex**, the **hippocampus**, and the **temporal lobe**.

The _prefrontal cortex_ (shown in Figure A above), located at the front of the brain, is crucial for short-term memory. It is responsible for temporarily holding, usually around 20 to 30 seconds and manipulating information, allowing us to perform tasks requiring immediate recall and attention. It's this fleeting retention that allows us to carry out everyday tasks. When you're engaged in a conversation, baking, or even ordering at a restaurant, you're continuously using your short-term memory. It's worth noting that the information in short-term memory is susceptible to interference. New data entering short-term memory quickly displace the old ones, and similar items in the environment can also interfere with these transient memories.

The _hippocampus_ (shown in the Figure A above), a small sea horse shaped structure deep within the brain, is essential for the formation of long-term memories. It acts as a gateway, consolidating and organizing information from short-term memory into long-lasting representations. Imagine you're taking a stroll through your childhood neighborhood, filled with familiar sights and sounds. As you walk past a house, a flood of memories rushes back. You remember the laughter of your childhood friends playing in the backyard, the scent of freshly cut grass, and the warm feeling of the sun on your face. These memories are stored in your long-term memory, which allows you to recall events and experiences from the distant past. Long-term memory is like a vast library of your life's experiences, knowledge, and skills. It's the repository where information is stored for an extended period, ranging from days to years or even a lifetime. Damage to the hippocampus can result in severe difficulties in forming new memories and remembering past events.

The _temporal lobe_ (illustrated in Figure B above), particularly the medial temporal lobe, works as a bridge between short-term and long-term memory. It interacts with the prefrontal cortex and hippocampus, aiding in the encoding and retrieval of memories. This region plays a crucial role in memory consolidation, where newly formed memories become more stable and integrated into the brain's existing knowledge network.

---

## I. Building Cognitive Chatbot Models

![Photo by [Mohamed Nohassi](https://unsplash.com/@coopery?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)](https://miro.medium.com/0*jM_iRNnN3r4nbxdn)

In the pursuit of designing conversation systems that mimic human cognition, the integration of _memory management_ becomes paramount. This article outlines the key requirements for such systems, including the incorporation of a consolidation bridge akin to the temporal lobe's function and the utilization of memory stores/databases for emulating the prefrontal cortex and hippocampus of our brain.

In order to facilitate immediate context processing, it is important that our chatbot models seamlessly integrate with a short-term memory store/database. This memory store will serve as a repository for retaining and recalling recent information exchanged during a conversation. By effectively capturing and organizing all utterances and turn-related data throughout the ongoing session, our model can emulate the functioning of the prefrontal cortex in the human brain. For the purpose of this article, we will be utilizing **[Amazon DynamoDB](https://aws.amazon.com/dynamodb/)** as the preferred choice for implementing this memory store solution.

Complementing the function of short-term memory stores, long-term memory play a vital role in preserving and consolidating past conversations and user preferences. In addition, these stores also serve as repositories for encoded domain-specific knowledge, facilitating seamless retrieval and enhancing the RAG process. This particular functionality mirrors the remarkable ability of the hippocampus in humans to store and retain long-term memories, empowering the chatbot to develop a personalized understanding of each individual user and maintain a persistent and enriched knowledge base. For further information on retrieval augmented generation (RAG), we invite you to explore our previous article [here](https://medium.com/@shankar.arunp/augmenting-large-language-models-with-verified-information-sources-leveraging-aws-sagemaker-and-f6be17fb10a8). To facilitate RAG and serve as our long-term memory store, we will be utilizing **[Amazon OpenSearch Service](https://aws.amazon.com/opensearch-service/)**.

To create a comprehensive conversational experience, we must incorporate a pipeline that emulates the temporal lobe's functionality within our chatbot models. The temporal lobe acts as the bridge between short-term and long-term memories, seamlessly integrating and organizing information over time. By emulating this mechanism, our models should ensure a smooth flow of knowledge and context, enhancing their ability to understand and respond to conversations with human-like cognitive capabilities. For this purpose, we will employ a pipeline that incorporates **[DynamoDB Streams](https://docs.aws.amazon.com/amazondynamodb/latest/developerguide/Streams.html)** coupled with **[AWS Lambda](https://aws.amazon.com/lambda/)**, which listens for the event when a conversation session concludes. This pipeline will not only be responsible for encoding the conversation but also for consolidating the information and facilitating its transition from short-term to long-term memory.

The integration of memory management into conversational AI systems is a critical step towards achieving human-like cognitive behavior. By leveraging short-term and long-term memory stores/databases, along with a pipeline that replicates the function of the temporal lobe, we can enable chatbot models to process, retain, and recall information in a manner similar to the human brain. This setup opens up new possibilities for creating intelligent and engaging conversational experiences, pushing the boundaries of AI-driven interactions with users.

---

## II. Core Architecture Overview

Our chatbot's cognitive architecture (shown below) encompasses several components, and we will provide a high-level overview of its design. To achieve efficient memory management, we leverage services such as SageMaker JumpStart, DynamoDB, and OpenSearch, along with a Lambda function.

At the core of the chatbot's cognition, LLMs (text generation and embedding models) play vital roles. The language model enables the chatbot to understand and generate human-like responses by processing and interpreting natural language inputs. Text embedding model, on the other hand, provide efficient representation of text data, enabling semantic understanding and similarity analysis.

Let's now delve into the steps involved in implementing and integrating the components of the chatbot's cognitive architecture at a high-level.

![High-level Cognitive Architecture](https://miro.medium.com/1*j0ZPgaCjpGFUceZSmNpVcw.png)

1. _New Session Initialization:_ Start a new session by generating a unique session ID. Store each conversation turn data (query and bot response) and its timestamp in the DynamoDB `conversations` table, mapped to the session ID.

2. _Session Completion:_ Record the start and end time of the session, calculate the duration, and count the number of turns. Update the `sessions` table in DynamoDB with the session's metadata, including the start and end times, duration, and number of turns.

3. _Event Capture:_ Utilize DynamoDB Streams to capture the session completion `update` event. Configure a Lambda function as a trigger for the DynamoDB Streams `update` event.

4. _Lambda Function Execution:_ When the Lambda function is triggered, retrieve the multi-turn conversation data using the session ID. Assemble the retrieved data into a context that is ordered chronologically.

5. _Context Summarization:_ Pass the assembled context to an endpoint of a SageMaker text generation model. Generate a concise summary for the multi-turn conversation thread using the SageMaker model.

6. _Context Encoding:_ Take the generated summary and pass it to a SageMaker embedding model endpoint. Encode the summary to create an embedding vector representation for the conversation summary.

7. _Storage of Conversation Data:_ Push the embedding vector, original summary, and relevant metadata (such as session ID and end time) to an OpenSearch index.

By following this process, the chatbot efficiently manages short-term memory using DynamoDB while utilizing OpenSearch as a long-term memory store/database for conversation summaries and relevant metadata.

---

## III. Configuring AWS Services

### SageMaker JumpStart

[JumpStart](https://docs.aws.amazon.com/sagemaker/latest/dg/studio-jumpstart.html) is a feature of Amazon SageMaker that provides developers with the ability to build, train, and deploy machine learning (ML) models quickly. The JumpStart Foundation Model Hub hosts a diverse collection of advanced LLMs contributed by prominent entities in the field. This includes esteemed open-source communities, AWS, and esteemed AWS partners like AI21, Stability.AI, Cohere, and various others. These language models enable a wide array of applications, from generating human-like text to answering questions about a given text, translating languages, summarizing long documents, and more.

For our chatbot application, we have opted to utilize two distinct models from the JumpStart's model hub. Our first selection is an embedding model `GPT-J 6B` to be used with the Retriever Augmented Generation (RAG) system. This is a 6 billion parameter, transformer-based model from [Hugging Face](https://huggingface.co/EleutherAI/gpt-j-6B) without a text generation model head. It takes a text string as input and produces an embedding vector with `4096` dimensions. While the published transformer-based model produces an embedding for every token in the input sequence, this model uses _mean pooling_, or an element-wise average, to aggregate these token embeddings into a sequence embedding. The RAG combines the benefits of retrieval-based and generative pre-training for machine learning, allowing us to extract precise information from the vast text corpus of your domain-specific data.

As the primary LLM in our cognitive architecture, we have chosen to employ the `FLAN T5 XXL`, a cutting-edge transformer-based model, is widely known for its exceptional capabilities in text generation tasks. It will serve as the backbone for our chatbot, enabling it to generate human-like, contextually aware responses.

To integrate these models into our system, we will follow the steps outlined in notebooks `[01-deploy-text-embedding-model.ipynb](https://github.com/arunprsh/aws-sagemaker-chatbot-memory/blob/main/01-deploy-text-embedding-model.ipynb)` and `[02-deploy-text-generation-model.ipynb](https://github.com/arunprsh/aws-sagemaker-chatbot-memory/blob/main/02-deploy-text-generation-model.ipynb)` in the sample [Github repository](https://github.com/arunprsh/aws-sagemaker-chatbot-memory) that accompanies this article. These notebooks will walk us through the creation of SageMaker endpoints. SageMaker endpoints are interfaces that allow for the deployment of models in a secure and scalable manner. They serve as conduits for application requests and responses, making them an essential component of our machine learning-powered chatbot. Ensuring these endpoints are set up correctly is pivotal to delivering a seamless and effective user interaction experience with our chatbot.

### Amazon OpenSearch Service

In our [previous article](https://medium.com/@shankar.arunp/augmenting-large-language-models-with-verified-information-sources-leveraging-aws-sagemaker-and-f6be17fb10a8) , we thoroughly discussed the concept of Retrieval Augmented Generation (RAG) with LLMs. We elucidated the process with an example, explaining how we encoded legal documents into embeddings and indexed them using Amazon OpenSearch Service. This technique was subsequently utilized to provide comprehensive answers to legal questions.

Amazon OpenSearch Service is a fully managed service that makes it easy to deploy, secure, and operate search and analytics suite, at scale. The service is well-suited for tasks such as application search, log analytics, and real-time monitoring, as it enables real-time searching, analysis, and visualization of data.

In the present article, we will extend our exploration and discuss the incorporation of the RAG setup as the long-term memory. The initial phase of the process aligns with the instructions laid out in our previous article. You are required to set up OpenSearch and generate embeddings and indices for your closed-domain data. Follow the detailed instructions and the code samples covered in our previous article.

However, for our current setup, we will need an additional step - the creation of an index for our conversation data. The fundamental idea here is that once a session is concluded, we aim to capture the multi-turn conversation, summarize it, encode it, and then transmit it to OpenSearch for indexing. To facilitate a smooth operation of this step, we have provided a notebook `[04-create-os-index.ipynb](https://github.com/arunprsh/aws-sagemaker-chatbot-memory/blob/main/04-create-os-index.ipynb)` in the example repository linked with this article.

Through the amalgamation of RAG, LLMs, and Amazon OpenSearch Service, we can build an effective long-term memory model that can be leveraged for complex, domain-specific question-answering tasks. By persistently storing and retrieving conversational data, the system continually evolves, enabling it to provide increasingly accurate and context-relevant responses.

### Amazon DynamoDB Tables

We chose DynamoDB as the storage solution for our multi-turn chatbot's conversations and sessions data because it perfectly aligned with our requirements. The decision was driven by several key factors. Firstly, DynamoDB's remarkable scalability was a crucial aspect, as it allowed us to effortlessly handle the constantly increasing volume of conversations and sessions. Secondly, its low latency ensured that retrieving and storing chatbot data was lightning-fast, creating a seamless user experience. Another reason behind our choice was DynamoDB's automatic scaling capabilities, which eliminated the need for manual provisioning and allowed us to optimize resource allocation efficiently.

For our architectural setup, we initiate the process by creating the essential tables for storing conversation and session data. We establish two tables, namely `conversations` and `sessions`, which will serve as the foundation for our data storage. To conveniently execute this task, you can utilize the notebook named `[03-create-dynamodb-tables.ipynb](https://github.com/arunprsh/aws-sagemaker-chatbot-memory/blob/main/03-create-dynamodb-tables.ipynb)`. Running this notebook will generate the tables with their respective partition and sort keys as shown below.

![DynamoDB Tables](https://miro.medium.com/1*40BocA4mVdAkvIeVvbG9fQ.png)

The figure below illustrates how a multi-turn conversation is segmented into individual turns and stored as a flattened tabular structure within the `conversations` table:

!["conversations" table at the end of a session](https://miro.medium.com/1*3mbyYp1Ulj4aBXHdbTthiw.png)

The `sessions` table (shown below) on the other hand serves as a record of conversation sessions, containing essential details such as the session ID, start and end times and number of turns in the conversation. It is important to note that the fields `end_time` and `num_turns` are initially set to `null` and `0`, respectively. These fields are intended to capture the session's end time and the number of turns within the session, but they remain unpopulated until the session reaches its conclusion. For our purposes, we define the conclusion of a session as the point when a user performs an action to create a new session, thereby signifying the conclusion of the current session. By adhering to this logic, we ensure accurate tracking of session data within our system.

![`"sessions" table at the start of a session`](https://miro.medium.com/1*nYwVg5hpW3sXadytuZJLrw.png)

!["sessions" table at the end of a session](https://miro.medium.com/1*IfhLAb9v8DRpN98tfmmhfQ.png)

### DynamoDB Streams

Next, we need to create a consolidation pipeline that bridges the information between our short-term and long-term memory storages, similar to the way the temporal lobe functions in the human brain, as we discussed in the previous sections. In our case, we aim to bridge the gap between DynamoDB and OpenSearch. We will use DynamoDB Streams for this purpose.

DynamoDB Streams is a feature of Amazon DynamoDB that provides a time-ordered sequence of item-level changes in a DynamoDB table. It captures a log of all modifications made to the table, including inserts, updates, and deletions, and stores them in a stream. This stream can be used to trigger and process events in real-time, enabling applications to react to changes in the database. DynamoDB Streams is commonly used for use cases such as maintaining replica tables, updating search indexes, and triggering downstream processes or notifications based on changes in the data.

To enable DynamoDB Streams for our scenario, we need to select the `sessions` table and enable the stream, as shown below. By doing so, we will be able to capture the item-level update events on our sessions table whenever a conversation session concludes.

![](https://miro.medium.com/1*TjnwE5AIShzY-Somq4oGHQ.png)

To capture the complete item as it appears after being updated, select the _New image_ option for the _View type_ as shown below. This ensures that we capture the entire updated item in its entirety.

### Amazon DynamoDB Streams

![](https://miro.medium.com/1*oqKrH79nGt0kvWZtSP7tgQ.png)

### AWS Lambda

As a secondary component of our memory consolidation pipeline, we need to perform downstream actions on the captured update events from DynamoDB Streams. To accomplish this, we utilize a Lambda function. AWS Lambda is a serverless compute service provided by Amazon Web Services. It allows you to run your code without managing servers, automatically scaling based on incoming requests. With Lambda, you pay only for the compute time consumed by your code, making it a cost-effective and efficient solution for event-driven applications.

Next, we will create the trigger to connect the DynamoDB stream with the Lambda function. To do this, we need to create the function from the Lambda console or using AWS SDK. The Lambda handler code is provided as a Python script `[05-lambda-handler.py](https://github.com/arunprsh/aws-sagemaker-chatbot-memory/blob/main/05-lambda-handler.py)` in the shared Github samples supplementing this article. An excerpt of the handler code is shown below.

```python
from boto3.dynamodb.conditions import Key
from requests.auth import HTTPBasicAuth
import requests
import boto3
import json
import os

...

# Create service clients
dynamodb = boto3.resource('dynamodb')
sagemaker_runtime = boto3.client('sagemaker-runtime')

# Reference SageMaker JumpStart endpoints
domain_endpoint = os.environ['ES_ENDPOINT']
domain_index = os.environ['ES_INDEX_NAME']

# Reference Amazon OpenSearch endpoint 
URL = f'{domain_endpoint}/{domain_index}'

...

def lambda_handler(event: dict, context) -> None:
    for record in event['Records']:
        if record['eventName'] == 'MODIFY':
            session_item = record['dynamodb']['NewImage']
            session_id = session_item['session_id']['S']
            end_time = session_item['end_time']['N']

            # Query the conversations table
            conversation_turns = query_conversations_table(session_id)

            # Flatten the conversation turns into a dict
            flattened_conversations = flatten_conversations(conversation_turns)
            # print(flattened_conversations)

            summary = summarize_conversations(flattened_conversations)

            # Encode the dict into an embedding
            embedding = encode_conversations(summary)

            # Write the embedding to Elasticsearch
            write_to_elasticsearch(session_id, embedding, end_time, summary)

            print(f"Session {session_id} was persisted to long term memory")
...
```

To make things work smoothly for the handler, we must make sure to set the correct environment variables.

![AWS Lambda: Environment Variables](https://miro.medium.com/1*wEXzSYQIqmowBwQWPKZyiQ.png)

Additionally, it is crucial to ensure that the Lambda function created has the necessary IAM role permissions to interact with DynamoDB tables, SageMaker endpoints, and OpenSearch. For our prototype, we have granted full access, but for a production environment, it is important to enable only the required actions.

![](https://miro.medium.com/1*gq0aDUEZsf_HlpL9Co5-Bw.png)

After successfully creating the Lambda function with the correct code and role access privileges, the next step is to establish a connection to the previously setup DynamoDB Stream. This can be achieved by creating a trigger and selecting the created Lambda function, as illustrated in the provided screenshots.

![](https://miro.medium.com/1*L7xEiAZ5IFcOnv9vLtyXxA.png)

![](https://miro.medium.com/1*OREFPH2uAh8em2VgTHMB6g.png)

Returning to the Lambda function page, you will find the DynamoDB trigger listed under the Configurations → Triggers section for the function, as depicted in the provided screenshots.

![](https://miro.medium.com/1*v_sgnTZn4Jlz_7SXEAUbxQ.png)

![](https://miro.medium.com/1*rEn3VvMKJQJDKQi5Mkjrbg.png)

Let's delve into the lambda handler case and understand how the memory consolidation process is facilitated here. The purpose of the lambda function is to summarize and store conversation data into a long-term storage system. Let's examine our handler code and the steps involved in this process.

The `lambda_handler` function starts by iterating through the event records to identify any modifications in the DynamoDB table. When a modification event is detected, relevant data such as the session ID and end time are extracted from the event. Subsequently, the code queries the conversations table using the session ID to retrieve the conversation turns associated with that particular session.

To summarize the conversation, the code flattens the conversation turns into a single string representation. Unnecessary information is removed, and user and bot messages are concatenated into a cohesive conversation text. The flattened conversation is then passed to the summarize function call, which utilizes a text generation model to generate a concise summary of the conversation.

The generated summary is encoded into an embedding using the text embedding model. The embedding is a numerical representation of the text summary, enabling further analysis or search operations. Finally, the code writes the session ID, embedding, end time, and conversation summary to an OpenSearch cluster, ensuring long-term storage and easy retrieval of the summarized conversation data.

In the field of LLMs, techniques have been explored to condense information for easier accessibility during query time. One such approach involves using an LLM to provide summaries of documents or conversations, and storing these summaries instead of the actual documents. This approach reduces noise, improves search results, and mitigates concerns about prompt token limitations.

Contextual summarization can be performed either at ingestion time or during query time. Conducting contextual compression during query time allows for more guided extraction of relevant context based on the provided question. However, heavy workloads during query time can negatively impact user latency. Therefore, it is recommended to offload as much of the workload to ingestion time as possible to enhance latency and avoid runtime issues, which is precisely what is done in this case. This technique of summarizing conversation history is applied to avoid encountering token limit problems, ensuring efficient storage and retrieval of information.

Additionally, the steps happening inside the Lambda function can be tracked by examining the [CloudWatch](https://aws.amazon.com/cloudwatch/) logs (shown below).

![](https://miro.medium.com/1*tgj0jcqXwsrwY1jmXbinbw.png)

---

## IV Building an AWS-backed Chatbot UI with Streamlit

Streamlit is an open-source Python library that simplifies the process of building interactive web applications for data science and machine learning projects. With Streamlit, developers can create and deploy user-friendly apps with minimal effort. Although not specifically designed for chatbot development, Streamlit can be used effectively to create chatbot interfaces.

In the previous sections, we learned how to configure the necessary AWS components required to enable our cognitive architecture with memory stores. This backend setup forms the foundation of our chat application. Now, let's explore how to utilize Streamlit to seamlessly create the frontend UI for our application. To begin, we need to set up the Streamlit application. You can find all the code required for this setup in our sample [GitHub repository](https://github.com/arunprsh/aws-sagemaker-chatbot-memory/tree/main/chatbot-app).

The provided code is a simplified implementation of an AI assistant powered by AWS services. The app features a chat interface where users can input their queries, with a carefully designed layout and configuration to enhance usability. A sidebar is included, allowing users to control the number of turns to remember and providing options to create new sessions or retrieve chat history from older sessions. These features empower users to customize the app's behavior and conveniently access previous interactions. Overall, the integration of Streamlit ensures an intuitive and visually appealing interface, delivering a seamless and professional user experience.

![](https://miro.medium.com/1*cfaJ-ROOA2nn0HnxojjuYw.png)

The chatbot consists of three main modules: `llm.py`, `retrieve.py`, and `app.py`. The `llm` module plays a crucial role in detecting the intent of incoming user queries or utterances. Each query is classified into one of three categories: chit chat utterances that require short-term memory, long-term memory queries that retrieve information from a closed domain-specific data indexed in OpenSearch, or long-term memory queries that retrieve past conversations.

Intent detection can be implemented in various ways. One approach is to use predefined patterns (see below). For example, if a query starts with patterns like `\past` or `/past`, it indicates a request for past conversations. Similarly, if it starts with `\verified` or `/verified`, it implies a desire to obtain answers from a verified source in long-term memory. Queries without such indicators default to regular chat that leverages the llm directly.

```python
def detect_task(query: str) -> str:
    if query.startswith('\\verified') or query.startswith('/verified'):
        return 'LTM VERIFIED SOURCES'
    elif query.startswith('\\past') or query.startswith('/past'):
        return 'LTM PAST CONVERSATIONS'
    else:
        return 'STM CHAT'
```

Another approach to intent classification is to have the chatbot itself perform the classification based on a few-shot prompt (as shown below).

```python
def detect_task(query: str) -> str:
    prompt = f"""
    Given a QUERY for the user, classify the INTENT behind it into one of the following classes:
    LTM VERIFIED SOURCES or LTM PAST CONVERSATIONS OR STM CHAT.
    
    QUERY => What is the process for filing a divorce in India?
    INTENT => LTM VERIFIED SOURCES
    QUERY => What is the UK law regarding intellectual property rights?
    INTENT => LTM VERIFIED SOURCES
    QUERY => Explain the concept of Habeas Corpus in Indian law.
    INTENT => LTM VERIFIED SOURCES
    QUERY => What are the rights of a tenant under UK law?
    INTENT => LTM VERIFIED SOURCES
    QUERY => What is the Indian law on cybercrime and data protection?
    INTENT => LTM VERIFIED SOURCES
    QUERY => What did we discuss yesterday?
    INTENT => LTM PAST CONVERSATIONS
    QUERY => Can you recall our conversation about climate change?
    INTENT => LTM PAST CONVERSATIONS
    QUERY => What was the solution to the math problem we were working on?
    INTENT => LTM PAST CONVERSATIONS
    QUERY => Can you remind me of the book recommendations you gave in our previous chat?
    INTENT => LTM PAST CONVERSATIONS
    QUERY => What were the details of the Indian law case study we discussed earlier?
    INTENT => LTM PAST CONVERSATIONS
    QUERY => How is the weather today?
    INTENT => STM CHAT
    QUERY => Where is Miami located?.
    INTENT => STM CHAT
    QUERY => What is your name?
    INTENT => STM CHAT
    QUERY => Hello!
    INTENT => STM CHAT
    QUERY => Hi!
    INTENT => STM CHAT
    QUERY => {query}
    INTENT =>
"""
```

Once the task or intent type is classified, the `llm` module generates an appropriate response using the context retrieved from the retrieve module. The `llm` module utilizes the SageMaker JumpStart's text generation model, specifically the `FLAN-T5-XXL` model, to perform various tasks based on user queries. It offers functions for summarizing passages, collating answers, and generating dialogue responses.

The `retrieve` module demonstrates the integration of OpenSearch and Amazon SageMaker to retrieve relevant information based on user queries. The module includes functions to encode a query using a text embedding model deployed on SageMaker, generate an OpenSearch query based on the encoded query, and retrieve top matching passages or past conversations from OpenSearch.

Finally, the `app` module acts as the driver for the chat utility, tying everything together by utilizing the `llm` and `retrieve` modules. It also includes functions for session management. Overall, the chatbot's architecture and modules are designed to effectively detect user intent, generate appropriate responses using text generation models, retrieve information from both short-term and long-term memory, and provide a seamless chat experience through the `app` module.

In our chatbot application, we have implemented a sliding window approach to manage the chat's past history. This approach allows us to retain only the `top N` turns in the history. Additionally, we have the option to enforce a constraint where the past history is clipped to a fixed value, ensuring it does not exceed a specific number of tokens bound to the LLM.

To provide a user-friendly experience, we have included a configuration option in the sidebar. This element allows users to set the maximum number of turns to remember for the past history. By adjusting this setting, users can control the amount of context retained by the chatbot.

The implementation follows a simple and straightforward logic, as demonstrated in the code snippet below:

```python
def transform_ddb_past_history(history: list, num_turns: int) -> str:
    past_hist = []
    for turn in history:
        me_utterance = turn['Me']
        bot_utterance = turn['AI']
        past_hist.append(f'Me: {me_utterance}')
        past_hist.append(f'AI: {bot_utterance}')
    past_hist = past_hist[-num_turns*2:]
    past_hist_str = '\n'.join(past_hist)
    return past_hist_str
```

Alternatively, you can also leverage the power of [LangChain](https://python.langchain.com/docs/get_started/introduction.html) to efficiently manage and manipulate previous chat history. LangChain provides a variety of strategies for applications like chatbots that rely on remembering past interactions.

---

## V Dissecting the Conversation Flow

Let us delve into the dynamics of conversation within the chatbot framework, specifically focusing on the chitchat intent and its treatment of direct queries unrelated to legal matters, as outlined in our setup. Such queries fall under the `STM CHAT` classification, implying that they can be promptly stored in short-term memory, namely DynamoDB. Moreover, these queries are directly forwarded to the text generation models for inference, as depicted in the aforementioned diagram.

![](https://miro.medium.com/1*kyFM-9yrlo7oIKkox6jxNw.png)

In situations where a query necessitates retrieving information from verified sources or past conversations, the chatbot relies on retrieval or semantic search mechanisms over encoded vectors (embeddings) of specialized documents or previous dialogues. Irrespective of the specific case, the query undergoes encoding and is utilized for semantic search to identify relevant documents or conversations. Subsequently, re-ranking takes place based on the similarity score, employing Euclidean distance for document search and a combination of timestamp and similarity score for conversations. The diagram below illustrates how the chatbot incorporates both the text generation and embedding models to handle these intent types, namely `LTM VERIFIED SOURCES` and `LTM PAST CONVERSATIONS`. Both scenarios necessitate accessing long-term memory for effective functioning.

![](https://miro.medium.com/1*B5o5ZalyxYHc-vl9j4hQhw.png)

Accessing long-term memory in AI systems can effectively follow the RAG pattern, as we extensively discussed in our previous article. This approach plays a crucial role in reducing the likelihood of AI hallucination and ensures that responses are factually accurate. By leveraging specialized domain knowledge, the RAG process optimizes the AI's capability to retrieve relevant information, formulate an accurate response, and subsequently generate comprehensive and coherent answers. This methodology underscores the importance of long-term memory in creating reliable and robust AI models.

![](https://miro.medium.com/0*1V2VkbOdm-Om9VYm.png)

Now, let's delve into a few exemplary dialogues involving our chatbot. The following interactions demonstrate the chatbot's adeptness in harnessing its long-term memory - a vector store (OpenSearch) containing verified documents from a custom domain - to accurately respond to inquiries. Leveraging semantic search, the bot enhances the text generation process to yield factually correct and validated responses. Notably, it ranks these responses based on relevance and provides pointers to the original legal document ID and precise passage from which the information was derived. This particular conversation illuminates the topic of _capital punishment_ in India.

![](https://miro.medium.com/1*PJnMdK5TmKQ6zYUgZ_ABWA.png)

![](https://miro.medium.com/1*KtM2NbKgbMMOXiLUD90xrg.png)

Here, we present another illustrative example where our chatbot proficiently applies the RAG pattern to address queries related to _court defamation_ law in the United Kingdom.

![](https://miro.medium.com/1*gpMl0IYmiguOASlQpNBuyA.png)

Recalling information from past conversations operates on similar principles as we discussed earlier, with one subtle yet significant difference. In this process, we re-rank the results not only based on relevance but also chronologically, thus integrating a time-based dimension into the ranking metric. Let's illustrate this through a dialogue where we retrieve previous discussions on the subject of capital punishment. This example will draw upon the conversations we examined earlier, thereby underlining the power of temporal relevance in enhancing the depth and accuracy of AI responses.

![](https://miro.medium.com/1*v6WLLkTLvz_IbCwyPVWJwg.png)

![](https://miro.medium.com/1*Foo_ylxPbhR5efOSJoY9RQ.png)

## VI Beyond the Horizon: Areas for Improvement and Reflections

So far, we have examined a range of scenarios commonly encountered in conversations. We have explored techniques for handling casual exchanges or "chit chat," utilizing the LLM directly to address open-domain questions, employing RAG approach for closed-domain inquiries, and retrieving previous conversations. Now, we will delve into more intricate scenarios and explore potential avenues for enhancing the bot's capabilities.

### Handling Stacked Turns

In scenarios where the exchange of turns deviates from the conventional back-and-forth pattern of conversation, a complex situation can arise when a human user inputs multiple separate queries or instructions in rapid succession, without waiting for a response in between. This unique form of interaction lacks a universally accepted term, as it breaks away from the traditional turn-taking model. To handle this, a system can be implemented to wait and receive all of the user's stacked inputs before generating a response. The system can then consolidate the inputs by either merging them into a single consolidated input for response generation, or generating individual responses for each query and later consolidating them. These mechanisms ensure that the context of the conversation is maintained and that appropriate responses are generated, enhancing the overall conversational experience for the user.

### **Routing to Human Agents**

Another scenario that is quite popular in the enterprise setting, particularly among customer support agents, involves implementing logic to facilitate the seamless transition of queries from a bot to a human agent. This logic comes into play when the bot is unable to retrieve relevant information or validate an answer based on verified sources, thereby necessitating the involvement of a human agent. This functionality can take various forms, including allowing users to directly communicate with a human agent if they so desire. Achieving this requires the ability to accurately sense the user's intent and ensuring a smooth and efficient transfer of complex or sensitive queries from the bot to human agents, ultimately enhancing the overall user experience. Importantly, this process does not terminate the session but rather redirects it to a human agent, ensuring continuity in the user's interaction.

### Addressing Inconclusive Sessions

Frequently, conversations are left in a state of flux, presenting unique challenges in managing chat sessions that did not reach a definitive conclusion. Discerning such inconclusive sessions and crafting strategies to handle them is an important aspect of the AI conversation domain. Moreover, it's crucial to appropriately categorize these incomplete conversations, ensuring their persistence in both short-term and long-term memory systems. While it's essential to store them in the immediate data repository for immediate follow-ups, the decision to encode and transition these to the long-term memory depends on multiple factors, including relevance, potential for future user interaction, and data storage considerations. This duality of memory systems enables a comprehensive approach to managing and learning from abandoned dialogues.

### Dealing with Connectivity Issues

A variant to the above scenario is to address disruptions caused by loss of connectivity during a conversation. A reliable solution for this scenario involves establishing a persistent connection between the chat UI and the server via various backend strategies. The server can store and regularly save the conversation history, safeguarding all dialogue exchanges. Cookies can be utilized to store session IDs or conversation IDs, enabling seamless retrieval of the user's conversation history when resuming the conversation after a disruption. These cookies can also temporarily store the user's recent inputs, allowing retrieval and transmission of the data to the server once connectivity is restored. This approach guarantees that no user input is lost during connectivity disruptions.

### Sentiment Tracking

Understanding the sentiment behind a user's utterance holds utmost importance in facilitating meaningful conversations. By incorporating sentiment tracking, chatbots can leverage this information to enhance their responses and effectively handle user interactions. Beyond merely discerning user intent, tracking the sentiment of user utterances post-bot response is vital. This process can be efficiently executed through the utilization of the LLM itself. This enables the chatbot to handle various scenarios adeptly, such as consecutive negative sentiments, which may necessitate human intervention to ensure optimal conversation flow.

### Managing Long Intervals Between Turns

Another important aspect of chatbot design involves addressing scenarios where there is a significant delay between user turns. This situation can occur when users are distracted during the conversation or are often searching for information, requiring the chatbot to maintain coherence and context in the conversation. By leveraging the LLM directly together with additional time keeping modules, chatbots can effectively manage these delays, ensuring a seamless conversational experience. This scenario is closely related to the inconclusive session scenario discussed earlier, highlighting the need for comprehensive conversation management.

Designing chatbot systems involves considering numerous unique scenarios and corner cases that necessitate the implementation of external modules and components. By supporting these additional scenarios, chatbots can become more dynamic and be better prepared for enterprise adoption, capable of handling real-world practical challenges.

## VII Final Remarks

Augmenting chatbots with memory is an important advancement in artificial intelligence. AWS plays a key role in enabling the development of advanced cognitive architectures for businesses. These chatbots can remember past conversations, access reliable sources, and provide accurate responses. With real-time adaptability and personalized interactions, chatbots with short-term memory improve user experiences and customer satisfaction. Moreover, chatbots with long-term memory become virtual knowledge repositories, enhancing decision-making and providing accurate information. Embracing augmented chatbots with memory, powered by AWS, helps organizations achieve higher productivity and efficiency in customer service.

In the future, chatbots will continue to evolve, incorporating advanced capabilities such as connecting to internal systems, making API calls via internet, and capturing entities and factual knowledge into their long-term memory. These chatbots will go beyond simple conversations and tap into various forms of long-term memory, such as knowledge graphs, to store and retrieve information.

By leveraging their extensive memory banks, chatbots will be able to combine past knowledge with newly available facts, generating innovative ideas and insights. They will actively seek out new information, update their knowledge repositories, and refine their decision-making processes.

Ultimately, the synergy between memory, connectivity, and learning in chatbots will drive organizations towards unparalleled productivity, efficiency, and customer satisfaction. With the limitless potential of augmented chatbots on the horizon, the future holds boundless opportunities for transforming the way we interact, learn, and innovate!

---

Thank you for taking the time to read and engage with this article. Your support in the form of following me and clapping the article is highly valued and appreciated. If you have any queries or doubts about the content of this article or the shared notebooks, please do not hesitate to reach out to me via email at _[arunprsh@amazon.com](mailto:arunprsh@amazon.com)_ or _[shankar.arunp@gmail.com](mailto:shankar.arunp@gmail.com)_. You can also connect me on [https://www.linkedin.com/in/arunprasath-shankar/](https://www.linkedin.com/in/arunprasath-shankar/)

I welcome any feedback or suggestions you may have. If you are an individual passionate about ML on scale, NLP/NLU, and interested in collaboration, I would be delighted to connect with you. Additionally, If you are an individual or part of a startup, or enterprise looking to gain insights on Amazon Sagemaker and its applications in NLP/ML, I would be happy to assist you. Do not hesitate to reach out to me.