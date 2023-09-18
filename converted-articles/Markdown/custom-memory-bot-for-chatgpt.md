# Custom Memory for ChatGPT API

### A Gentle Introduction to LangChain Memory Types

![Self-made gif.](https://miro.medium.com/1*B275_3l9j_mw4-W9G7ZCqA.gif)

If you have ever used the OpenAI API, I am sure you have noticed the catch.

_Got it?_

_Right!_ Every time you call the ChatGPT API, the model has no memory of the previous requests you have made. In other words: **each API call is a standalone interaction**.

And that is definitely annoying when you need to perform follow-up interactions with the model. A chatbot is the golden example where follow-up interactions are needed.

In this article, we will explore how to give memory to ChatGPT when using the OpenAI API, so that it remembers our previous interactions.

## Warm-Up!

Let's perform some interactions with the model so that we experience this default no-memory phenomenon:

```makefile
prompt = "My name is Andrea"
response = chatgpt_call(prompt)
print(response)

# Output: Nice to meet you, Andrea! How can I assist you today?
```

But when asked a follow-up question:

```makefile
prompt = "Do you remember my name?"
response = chatgpt_call(prompt)
print(response)

# Output: I'm sorry, as an AI language model, I don't have the ability 
# to remember specific information about individual users.
```

_Right,_ so in fact the model does not remember my name even though it was given on the first interaction.

**Note:** The method `chatgpt_call()` is just a wrapper around the OpenAI API. We already gave a shot on how easily call GPT models at [ChatGPT API Calls: A Gentle Introduction](https://medium.com/forcodesake/chatgpt-api-calls-introduction-chatgpt3-chatgpt4-ai-d19b79c49cc5) in case you want to check it out!

Some people normally work around this memoryless situation by pre-feeding the previous conversation history to the model every time they do a new API call. Nevertheless, this practice is not cost-optimized and it has certainly a limit for long conversations.

In order to create a memory for ChatGPT so that it is aware of the previous interactions, we will be using the popular `langchain` framework. This framework allows you to easily manage the ChatGPT conversation history and optimize it by choosing the right memory type for your application.

## **LangChain Framework**

The `langchain` framework's **purpose is to assist developers when building applications powered by Large Language Models (LLMs).**

![Self-made screenshot from the official LangChain [GitHub repository](https://github.com/hwchase17/langchain).](https://miro.medium.com/1*6NnR65TjZ5F3mfwiGLXiGw.png)

According to their [GitHub description](https://github.com/hwchase17/langchain):

> Large language models (LLMs) are emerging as a transformative technology, enabling developers to build applications that they previously could not. However, using these LLMs in isolation is often insufficient for creating a truly powerful app - the real power comes when you can combine them with other sources of computation or knowledge.

> This library aims to assist in the development of those types of applications.

They claim that building an application by only using LLMs may be insufficient. We found that too when doing follow-up interactions with the model by using the OpenAI API only.

### Framework Setup

Getting the `langchain` library up and running in Python is simple. As for any other Python library, we can install it with `pip`:

```typescript
pip install langchain
```

LangChain calls the OpenAI API behind the scenes. Therefore, it is necessary to set your OpenAI API key as an environment variable called `OPENAI_API_KEY`. Check out [A Step-by-Step Guide to Getting Your API Key](https://medium.com/forcodesake/a-step-by-step-guide-to-getting-your-api-key-2f6ee1d3e197) if you need some guidance for getting your OpenAI key.

### LangChain: Basic Calls

Let's start by setting up a basic API call to ChatGPT using LangChain.

This task is pretty straightforward since the module `langchain.llms` already provides an `OpenAI()` method for this purpose:

```python
# Loads OpenAI key from the environment
from langchain.llms import OpenAI
chatgpt = OpenAI()
```

Once the desired model is loaded, we need to start the so-called _conversation chain_. LangChain also provides a module for that purpose:

```python
from langchain.chains import ConversationChain
conversation = ConversationChain(llm=chatgpt)
```

Let's define the conversation with `verbose=True` to observe the reasoning process of the model.

Finally, `langchain` provides a `.predict()` method to send your desired prompt to ChatGPT and get its completion back. _Let's try it!_

```python
conversation.predict(input="Hello, we are ForCode'Sake! A Medium publication with the objective of democratizing the knowledge of data!")
```

_Let's do a follow-up interaction!_

```python
conversation.predict(input="Do you remember our name?")

# Output: " Hi there! It's great to meet you. 
# I'm an AI that specializes in data analysis. 
# I'm excited to hear more about your mission in democratizing data knowledge.
# What inspired you to do this?"
```

We can see that **the model is capable of handling follow-up interactions without problems when using `langchain`**.

## LangChain Memory Types

As we have observed, LangChain conversation chains already keep track of the `.predict` calls for a declared `conversation`. However, **the default conversation chain stores each and every interaction we have had with the model**.

As we have briefly discussed at the beginning of the article, storing all the interactions with the model can **quickly escalate to a considerable amount of tokens to process every time we prompt the model**. It is essential to bear in mind that ChatGPT has a token limit per interaction.

In addition, the **ChatGPT usage cost also depends on the number of tokens**. Processing all the conversation history in each new interaction is likely to be expensive over time.

To overcome these limitations, `langchain` implements different types of memories to use in your application.

_Let's explore them!_

### #1. Complete Interactions

Although the default behavior of LangChain is to store all the past interactions, this memory type can be explicitly declared. It is the so-called `ConversationBufferMemory`, and it simply fills a buffer with all our previous interactions:

```python
from langchain.memory import ConversationBufferMemory
memory=ConversationBufferMemory()
```

Declaring the memory type allows us to have some additional control over the ChatGPT memory. For example, we can check the buffer content at any time with `memory.buffer` or `memory.load_memory_variables({})`.

In addition, we can add extra information to the buffer without doing a real interaction with the model:

```css
memory.save_context({"input": "Hi"},
                    {"output": "What's up"})
memory.save_context({"input": "Not much, just hanging"},
                    {"output": "Cool"})
```

If you do not need to manipulate the buffer in your application, you might be good to go with the default memory and no explicit declaration. **Although I really recommend it for debugging purposes!**

### #2. Interactions within a window

One less costly alternative is storing only a certain amount of previous interactions (`k`) with the model. That is the so-called _window_ of interaction.

When conversations grow big enough, it might be sufficient for your application that the model only remembers the most recent interactions. For those cases, the `ConversationBufferWindowMemory` module is available.

_Let's explore its behavior!_

Firstly, we need to load the `llm` model and the new type of `memory`. In this case, we are setting `k=1` which means that only the previous iteration will be kept in memory:

```python
# OpenAI key from environment
from langchain.llms import OpenAI
llm = OpenAI()
```

```java
from langchain.memory import ConversationBufferWindowMemory
memory = ConversationBufferWindowMemory(k=1)
```

Secondly, let's add some context to our conversation as shown in the previous section:

```python
memory.save_context({"input": "Hi"},
                    {"output": "What's up"})
memory.save_context({"input": "Not much, just hanging"},
                    {"output": "Cool"})
```

Although we have stored two interactions in our conversation history, due to the fact that we have set `k=1`, the model will only remember the last interaction `{"input": "Not much, just hanging"},
 {"output": "Cool"}`.

To prove it, let's check what the model has in memory:

```python
memory.load_memory_variables({})

# Output: {'history': 'Human: Not much, just hanging\nAI: Cool'}
```

We can further prove it by asking a follow-up question setting `verbose=True`, so that we can observe the stored interactions:

```python
conversation.predict(input="Can you tell me a joke?")
```

And the verbose output is the following:
<script src="https://gist.github.com/aandvalenzuela/0b75634f6407495de1370cf5cfe337d8.js"></script>
As we can observe, **the model only remembers the previous interaction**.

### #3. Summary of the interactions

I am sure you are now thinking that **completely deleting old interactions with the model might be a bit risky for some applications**.

Let's imagine a customer service chatbot that asks to the user its contract number in the first place. The model must not forget this information, no matter which interaction number it has.

For that purpose, there is a memory type that uses the model itself to generate a summary of the previous interactions. Therefore, the model only stores a summary of the conversation in memory.

This optimized memory type is the so-called `ConversationSummaryBufferMemory`. It also allows to store the complete most recent interactions up to a maximum number of tokens (given by `max_token_limit`) together with the summary of the previous ones.

_Let's observe this memory behavior in practice!_

```javascript
from langchain.memory import ConversationSummaryBufferMemory
```

Let's create a conversation with quite some content so that we can explore the summary capabilities:

```python
schedule = "There is a meeting at 8am with your product team. \
You will need your powerpoint presentation prepared. \
9am-12pm have time to work on your LangChain \
project which will go quickly because Langchain is such a powerful tool. \
At Noon, lunch at the Italian restaurant with a customer who is driving \
from over an hour away to meet you to understand the latest in AI. \
Be sure to bring your laptop to show the latest LLM demo."

memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=100)
memory.save_context({"input": "Hello"}, {"output": "What's up"})
memory.save_context({"input": "Not much, just hanging"},
                    {"output": "Cool"})
memory.save_context({"input": "What is on the schedule today?"}, 
                    {"output": f"{schedule}"})
```

Now, when checking the memory content with `memory.load_memory_variables({})` , we will see the actual summary of our interactions:

```json
{
  'history': 'System: 
  \nThe human greets the AI and asks what is on the schedule for the day. 
  The AI responds with "Cool".\n
  AI: There is a meeting at 8am with your product team. 
  You will need your powerpoint presentation prepared. 
  9am-12pm have time to work on your LangChain project which will go quickly because Langchain is such a powerful tool. 
  At Noon, lunch at the italian resturant with a customer who is driving from over an hour away to meet you to understand the latest in AI. 
  Be sure to bring your laptop to show the latest LLM demo.'
}
```

_The summary sounds nice, isn't it?_

Let's perform a new interaction!

```python
llm = OpenAI()

conversation = ConversationChain(
    llm=llm, 
    memory = memory,
    verbose=True
)
```

And the verbose output looks as follows:
<script src="https://gist.github.com/aandvalenzuela/162243fbca72b86439be4e9597dfd1b0.js"></script>
As we can observe from the example, **this memory type allows the model to keep important information, while reducing the irrelevant information** and, therefore, the amount of used tokens in each new interaction.

## Summary

In this article, we have seen **different ways to create a memory for our GPT-powered application** depending on our needs.

By **using the LangChain framework instead of bare API calls to the OpenAI API**, we get rid of simple problems such as making the model aware of the previous interactions.

Despite the fact that the default memory type of LangChain might be already enough for your application, I really encourage you to estimate the average length of your conversations. It is a nice exercise to compare the average number of tokes used - and therefore the cost! - with the usage of the summary memory. **You can get full model performance at a minimal cost!**

It seems to me that the LangChain framework has a lot to give us regarding GPT models. _Have you already discovered another handy functionality?_

---

That is all! Many thanks for reading!

I hope this article helps you when **building ChatGPT applications!**

You can also subscribe to my **[Newsletter](https://medium.com/@andvalenzuela/subscribe)** to stay tuned for new content. **Especially**, **if you are interested in articles about ChatGPT**:

> [**ChatGPT Moderation API: Input/Output Control**](https://towardsdatascience.com/chatgpt-moderation-api-input-output-artificial-intelligence-chatgpt3-data-4754389ec9c8)

> [**Unleashing the ChatGPT Tokenizer**](https://towardsdatascience.com/chatgpt-tokenizer-chatgpt3-chatgpt4-artificial-intelligence-python-ai-27f78906ea54)

> [**Mastering ChatGPT: Effective Summarization with LLMs**](https://towardsdatascience.com/chatgpt-summarization-llms-chatgpt3-chatgpt4-artificial-intelligence-16cf0e3625ce)

Also towards a **responsible AI**:

> [**What ChatGPT Knows about You: OpenAI's Journey Towards Data Privacy**](https://towardsdatascience.com/what-chatgpt-knows-about-you-openai-towards-data-privacy-science-ai-b0fa2376a5f6)