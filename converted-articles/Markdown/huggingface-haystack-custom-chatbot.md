![](https://miro.medium.com/1*U7EUlePDqjUa2js11eeNxg.png)

# Build A Chatbot Conversational App with Haystack & HuggingFace

### ChatGPT and HuggingChat are both web-based Conversational Apps consisting of a UI, conversational memory and access to a LLM. Here I show how you can create your own conversational app using open-source technology.

_I'm currently the Chief Evangelist @ [HumanFirst](https://www.humanfirst.ai/). I explore and write about all things at the intersection of AI and language; ranging from LLMs, Chatbots, Voicebots, Development Frameworks, Data-Centric latent spaces and more._

_In the coming week I aim to compile a matrix with the most notable LLM-based development frameworks for conversational applications._

This article covers a simple notebook example on how to build a conversational app which has memory. There are two main components used for this demo, HuggingFace and Haystack.

Via _Hugging Face_, [hosted Inference API](https://huggingface.co/docs/api-inference/index)s can be used to access Large Language Models using simple HTTP requests.

You don't need to download models, perform any fine-tuning or training. All you need is an API Key from HuggingFace, as seen below:

![](https://miro.medium.com/1*TFjQjs2CBAf16US3TcEqfA.png)

Haystack is an open-source, pro-code framework to build Autonomous Agents, prompt pipelines, search tools and more.

The conversational app demo shown below will make use of three nodes; `PromptNode`, `ConversationalAgent` & `ConversationSummaryMemory`.

## PromptNode

The [PromptNode](https://docs.haystack.deepset.ai/docs/prompt_node) is initialised with three parameters, `model_name`, `api_key`, and `max_length` to manage the model output.

```python
from haystack.nodes import PromptNode

model_name = "OpenAssistant/oasst-sft-1-pythia-12b"
prompt_node = PromptNode(model_name, api_key=model_api_key, max_length=256)
```

Here is the simplest implementation of the PromptNode:

```python
pip install --upgrade pip
pip install farm-haystack[colab]

from haystack.nodes import PromptNode
prompt_node = PromptNode()
prompt_node("What is the capital of Germany?")
```

And the output:

```plaintext
['berlin']
```

## ConversationSummaryMemory

Conversation memory is important for conversational apps to have a human-like element.

Follow-up questions can be asked which reference previous conversational context in an implicit fashion.

[ConversationSummaryMemory](https://docs.haystack.deepset.ai/docs/agent#conversational-agent-memory) is used to save space and also LLM tokens.

The summary has a brief overview of the conversation history and will be updated as the conversation continues.

_Implementing Conversation Summary Memory:_

```python
from haystack.agents.memory import ConversationSummaryMemory

summary_memory = ConversationSummaryMemory(prompt_node)
```

## ConversationalAgent

And lastly, the conversational [agent](https://cobusgreyling.medium.com/agents-llms-multihop-question-answering-ca6521227b6c):

```python
from haystack.agents.conversational import ConversationalAgent

conversational_agent = ConversationalAgent(prompt_node=prompt_node, memory=summary_memory)
```

A conversational agent is an agent holding conversational memory. To read more about agents, you can refer to this [article](https://cobusgreyling.medium.com/llm-apps-2dc9c6ac7ebd).

Here is the complete code for the Conversational Agent:

```python
pip install --upgrade pip
pip install farm-haystack[colab]

from getpass import getpass
model_api_key = getpass("Enter model provider API key:")

from haystack.nodes import PromptNode

model_name = "OpenAssistant/oasst-sft-1-pythia-12b"
prompt_node = PromptNode(model_name, api_key=model_api_key, max_length=256)

from haystack.agents.memory import ConversationSummaryMemory
summary_memory = ConversationSummaryMemory(prompt_node)

from haystack.agents.conversational import ConversationalAgent
conversational_agent = ConversationalAgent(prompt_node=prompt_node, memory=summary_memory)

conversational_agent.run("What are the five biggest countries in Africa?")
conversational_agent.run("What is the  main languages spoken in these countries?")
conversational_agent.run("Are any of the counries land-locked?")

print(conversational_agent.memory.load())
```

---

_**⭐️ Please follow me on [LinkedIn](https://www.linkedin.com/in/cobusgreyling/) for updates on LLMs ⭐️**_

![](https://miro.medium.com/1*lIm_TXh6TC9uGn63lOjZtQ.png)

_I'm currently the [Chief Evangelist](https://www.linkedin.com/in/cobusgreyling) @ [HumanFirst](https://www.humanfirst.ai). I explore and write about all things at the intersection of AI and language; ranging from LLMs, Chatbots, Voicebots, Development Frameworks, Data-Centric latent spaces and more._

> [**NLU design tooling**](https://www.humanfirst.ai)

![](https://miro.medium.com/1*qPfFI9uFl04n1ZUywxH38w.png)

![[https://www.linkedin.com/in/cobusgreyling](https://www.linkedin.com/in/cobusgreyling)](https://miro.medium.com/1*mwQw4LOeZdWG1AD8RDheXw.jpeg)

![](https://miro.medium.com/1*qPfFI9uFl04n1ZUywxH38w.png)

> [**Get an email whenever Cobus Greyling publishes.**](https://cobusgreyling.medium.com/subscribe)

> [**COBUS GREYLING**](https://www.cobusgreyling.com)

![](https://miro.medium.com/1*qPfFI9uFl04n1ZUywxH38w.png)

> [**Building a Conversational Chat App | Haystack**](https://haystack.deepset.ai/tutorials/24_building_chat_app)

> [**ChatGPT APIs & Managing Conversation Context Memory**](https://cobusgreyling.medium.com/chatgpt-apis-managing-conversation-context-memory-8b100dfe544a)

> [**Build Your Own ChatGPT or HuggingChat**](https://cobusgreyling.medium.com/build-your-own-chatgpt-or-huggingchat-876d01b1ef4a)