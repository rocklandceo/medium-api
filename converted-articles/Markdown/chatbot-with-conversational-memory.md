# How to build a Chatbot with ChatGPT API and a Conversational Memory in Python

**🧠 Memory Bot 🤖 - An easy up-to-date implementation of ChatGPT API, the GPT-3.5-Turbo model, with LangChain AI's 🦜 - ConversationChain memory module with Streamlit front-end.**

👨🏾 ‍💻  G[itHub ](https://github.com/avrabyt)⭐️| 🐦  T[witter ](https://twitter.com/avra_b)| 📹  Y[ouTube ](https://www.youtube.com/@Avra_b)| ☕️ B[uyMeaCoffee ](https://www.buymeacoffee.com/AvraCodes)| K[o-fi💜 ](https://ko-fi.com/avrabyt)

![](https://miro.medium.com/1*zCdaOlocbFcZY60GxcrhtA.png)

## Introduction

With the emergence of Large Language Models (LLMs), AI technologies have advanced to a level where humans can converse with chatbots in a way that resembles human conversation. In my opinion, chatbots are poised to become an essential component of our daily lives for a wide range of problem-solving tasks. We will soon encounter chatbots in various domains, including customer service and personal assistance.

!["Conversation is food for the soul. It nourishes our spirits and helps us to grow." - John Templeton |Photo by [Etienne Boulanger](https://unsplash.com/@etienneblg?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)](https://miro.medium.com/0*EiU1SREAmC7shEy7)

Let me highlight the relevance of this blog post, by addressing the important _**context**_ in our day-to-day _**conversation**_. Conversations are natural ways for humans to communicate and exchange informations. In conversations, we humans rely on our _**memory**_ to remember what has been previously discussed (i.e. the **context**), and to use that information to generate relevant responses. Likewise, instead of humans if we now include chatbots with whom we would like to converse, _**the ability to remember the context of the conversation is important for providing a seamless and natural conversational experience**._

_"In a world where you can be anything, be kind. And one of the simplest ways to do that is through conversation" - Karamo Brown_

Now, imagine a chatbot that is _**stateless**_, i.e. the chatbot treats each incoming query/input from the user independently - and forgets about the past conversations or context ( in simpler terms they lack the memory ). I'm certain, we all are used to such AI assistants or chatbots.I would refer to them here as _**traditional**_ chatbots.

![You definitely know the context here ....😉  |Photo by [Tron Le](https://unsplash.com/@tronle_sg?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)](https://miro.medium.com/0*Rytnxznp3bt9nbaA)

A major drawback of traditional chatbots is that they can't provide a seamless and natural conversational experience for users. Since they don't remember the context of the conversation, users often have to repeat themselves or provide additional information that they've already shared. Another issue can sometimes be irrelevant or "off-topic". Without such abilities, it's more difficult for these chatbots to generate coherent and relevant responses based on what has been discussed. This can lead to frustrating and a less satisfying user experience.

![Traditional chatbots looses the context . For instance, does not remember the name of the 'country' that was mentioned in the past context | Image by Author](https://miro.medium.com/1*g4vyQTiQ9T4-_GE_16U2_g.png)

_I've a [blog post](https://medium.com/@avra42/build-your-own-chatbot-with-openai-gpt-3-and-streamlit-6f1330876846) and [YouTube video](https://youtu.be/BHwVRI9N8B0) explaining how to build such **traditional or simple Chatbot**. Here's a quick recap and [live app to try.](https://next.databutton.com/v/lgzxq112/Traditional_Chat_Bot)_

![A quick demo of a **traditional** **chat bot** 🤖 which I built and demostrated previously. The bot lacks the memory and looses the conversational context.](https://miro.medium.com/1*-c-2eJi_qR8gm6V5j9QMrw.gif)

_**However, in this blog post, we will be introducing our chatbot that overcomes the limitations of traditional chatbots. Our chatbot will have the ability to remember the context of the conversation, making it a more natural and seamless experience for the users.**_ We like to call it the "**MemoryBot**" 🧠 🤖

![](https://miro.medium.com/1*j04XeiXI9jiF_Fgj-iqt4g.png)

## Building the 🧠 Memory Bot 🤖

The following resources will be instrumental in our development,

- **[OpenAI](https://openai.com)** is a research organization that aims to create advanced artificial intelligence in a safe and beneficial way. They have developed several large language models (LLMs) like GPT-3, which are capable of understanding and generating human-like language. These models are trained on vast amounts of data and can perform tasks such as language translation, summarization, question answering, and more. The models offered can be accessed via API keys. In order to create one, please follow my other blog posts and tutorial videos (refer to the related blog section below ). Open AI also provides a Python package to work with. **For installation use,** `pip install openai`

- **[LangChain](https://langchain.readthedocs.io/en/latest/index.html)** is a Python library that provides a standard interface for memory and a collection of memory implementations for chatbots. It also includes examples of chains/agents that use memory, making it easy for developers to incorporate conversational memory into their chatbots using LangChain.LangChain's memory module for chatbots is designed to enable conversational memory for large language models (LLMs). **For installation use,** `pip install langchain`

- **[Streamlit](https://streamlit.io)** is an open-source app framework for building data science and machine learning web applications. It allows developers to create interactive web applications with simple Python scripts. **For installation use,** `pip install streamlit`

- **[DataButton](https://www.databutton.io/)** is an online workspace for creating full-stack web apps in Python. From writing Python scripts to building a web app in Streamlit framework and finally to deployment in the server— all come in a single workspace. _**Moreover, you can skip the above packages installation steps. Instead, directly plug-in those package name in the configuration space which DataButton provides and leave the rest on DataButton to handle !**_ You can gain free access to their tools by signing up and start building one for yourself.

### Workflow

**Model**: We will be using the very latest, ChatGPT API, the GPT-3.5-Turbo large language model which OpenAI offers - that can understand as well as generate natural language or code. _**As Open AI claims it is one of the most capable and cost-effective models they offer at this moment in the GPT3.5 family.**_ ( read more [here](https://platform.openai.com/docs/models/gpt-3-5) )

![](https://miro.medium.com/1*nfpdZevVlSmSUjqKv7pSbw.png)

**[ConversationChain](https://langchain.readthedocs.io/en/latest/modules/memory/getting_started.html#using-in-a-chain) and [Memory](https://langchain.readthedocs.io/en/latest/modules/memory/key_concepts.html#key-concepts):** One of the key core components which LangChain provides us with are - chains. [Please refer to my earlier blog post to have a detailed understanding](https://medium.com/@avra42/getting-started-with-langchain-a-powerful-tool-for-working-with-large-language-models-286419ba0842) on how it works and one of its use-cases in integrating LLMs.

We will use a combination of **chains**: `ConversationChain` (has a simple type of memory that remembers all previous inputs/outputs and adds them to the context that is passed) and **memory** comprising of (a) buffer, which can accept the _n_ number of user interactions as context (b) summary, that can summarize the past conversations. At times both (a) and (b) can be included together as a memory.

We will try to implement a relatively sophisticated memory called ["Entity Memory"](https://langchain.readthedocs.io/en/latest/modules/memory/key_concepts.html#entity-memory) , compared to other available memory available in this module. EntityMemory is best defined in LangChain AI's official docs,
_"A more complex form of memory is remembering information about specific entities in the conversation. This is a more direct and organized way of remembering information over time. Putting it in a more structured form also has the benefit of allowing easy inspection of what is known about specific entities"_

**Front-end development:** To build the chatbot, we'll be using the online DataButton platform which has a in-built code editor ( _IDE_ ), a package plus configuration maintenance environment, alongside with a space to view the development in real-time ( i.e. _localhost_ ). Since DataButton utilizes the Streamlit framework, the code can be written with simple Streamlit syntax.

![](https://miro.medium.com/1*GvitPIDZKylwKbpbdU4wiQ.png)

Alternatively, the entire front-end process can also be developed locally via typical Streamlit-Python Web app development workflow which I've discussed several times over my YouTube / blog posts tutorial. Briefly, it follow,

- Writing and testing the code locally in the computer

- Adding the dependencies as `requirements.txt` file

- Pushing to the GitHub and deployment over the [Streamlit cloud](https://streamlit.io/cloud)

Please refer to my other Streamlit-based blog posts and YouTube tutorials.

![](https://miro.medium.com/1*o1nb8D-2qpBn15DUf54kPA.png)

Moreover, both the above-mentioned methods, at this moment allows free-hosting of web apps. Please refer to the respective official websites for further details.

## The Code

We will now move to the main section of developing our Memory Bot with very few lines of python syntax.

- We will start with **importing necessary libraries** ,

```javascript

import streamlit as st
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationEntityMemory
from langchain.chains.conversation.prompt import ENTITY_MEMORY_CONVERSATION_TEMPLATE
from langchain.llms import OpenAI
```

- Followed by , **setting up the Streamlit page configuration**. Not critical, but can be a nice UI add-on to the Web App. ( r[efer to the doc](https://docs.streamlit.io/library/api-reference/utilities/st.set_page_config) )

```bash
st.set_page_config(page_title='🧠MemoryBot🤖', layout='wide')
```

- **Initialize session states**. One of t**he critical steps** - since the conversation between the user input, as well as the memory of '_chains of thoughts_' needs to be stored at every reruns of the app

Session state is useful to store or cache variables to avoid loss of assigned variables during default workflow/rerun of the Streamlit web app. I've discussed this in my previous [blog posts](https://medium.com/dev-genius/streamlit-python-tips-how-to-avoid-your-app-from-rerunning-on-every-widget-click-cae99c5189eb) and [video](https://youtu.be/dPdB7zyGttg) as well - do refer to them. Also( [refer to the official doc](https://docs.streamlit.io/library/api-reference/session-state) ).

```bash
if "generated" not in st.session_state:
    st.session_state["generated"] = []
if "past" not in st.session_state:
    st.session_state["past"] = []
if "input" not in st.session_state:
    st.session_state["input"] = ""
if "stored_session" not in st.session_state:
    st.session_state["stored_session"] = []
```

- **We'll define a function to get the user input.** Typically not necessary to wrap within a function, but why not ...

```python
def get_text():
    """
    Get the user input text.
    Returns:
        (str): The text entered by the user
    """
    input_text = st.text_input("You: ", st.session_state["input"], key="input",
                            placeholder="Your AI assistant here! Ask me anything ...", 
                            label_visibility='hidden')
    return input_text
```

- **Additional feature : Start a new chat,** at times - we might want our Memory Bot to erase its memory / context of the conversation and start a new one. This function can be super useful in such circumstances,

```python
def new_chat():
    """
    Clears session state and starts a new chat.
    """
    save = []
    for i in range(len(st.session_state['generated'])-1, -1, -1):
        save.append("User:" + st.session_state["past"][i])
        save.append("Bot:" + st.session_state["generated"][i])        
    st.session_state["stored_session"].append(save)
    st.session_state["generated"] = []
    st.session_state["past"] = []
    st.session_state["input"] = ""
    st.session_state.entity_memory.store = {}
    st.session_state.entity_memory.buffer.clear()
```

- **Some config for a user to play with:** Options to preview the buffer and the memory. Also changing to different GPT-3 offered models.

```python
with st.sidebar.expander(" 🛠 ️ Settings ", expanded=False):
    # Option to preview memory store
    if st.checkbox("Preview memory store"):
        st.write(st.session_state.entity_memory.store)
    # Option to preview memory buffer
    if st.checkbox("Preview memory buffer"):
        st.write(st.session_state.entity_memory.buffer)
    MODEL = st.selectbox(label='Model', options=['gpt-3.5-turbo','text-davinci-003','text-davinci-002','code-davinci-002'])
    K = st.number_input(' (#)Summary of prompts to consider',min_value=3,max_value=1000)

```

- **Set up the App Layout and widget to accept secret API key**

```python
# Set up the Streamlit app layout
st.title("🧠 Memory Bot 🤖")
st.markdown(
        ''' 
        > :black[**A Chatbot that remembers,**  *powered by -  [LangChain]('https://langchain.readthedocs.io/en/latest/modules/memory.html#memory') + 
        [OpenAI]('https://platform.openai.com/docs/models/gpt-3-5') + 
        [Streamlit]('https://streamlit.io') + [DataButton](https://www.databutton.io/)*]
        ''')
# st.markdown(" > Powered by -  🦜 LangChain + OpenAI + Streamlit")

# Ask the user to enter their OpenAI API key
API_O = st.sidebar.text_input(":blue[Enter Your OPENAI API-KEY :]", 
                placeholder="Paste your OpenAI API key here (sk-...)",
                type="password") # Session state storage would be ideal
```

**Creating key Objects:** This is a very crucial part of the code

- **Open AI Instance** needs to be created which will be later called

- **ConversationEntityMemory** is stored as session state

- **ConversationChain** is initiated.

Storing the Memory as Session State is pivotal otherwise the memory will get lost during the app re-run. A perfect example to use Session State while using Streamlit.

```python
if API_O:
    # Create an OpenAI instance
    llm = OpenAI(temperature=0,
                openai_api_key=API_O, 
                model_name=MODEL, 
                verbose=False) 


    # Create a ConversationEntityMemory object if not already created
    if 'entity_memory' not in st.session_state:
            st.session_state.entity_memory = ConversationEntityMemory(llm=llm, k=K )
        
        # Create the ConversationChain object with the specified configuration
    Conversation = ConversationChain(
            llm=llm, 
            prompt=ENTITY_MEMORY_CONVERSATION_TEMPLATE,
            memory=st.session_state.entity_memory
        )  
else:
    st.markdown(''' 
        ```
        - 1. Enter API Key + Hit enter 🔐  

        - 2. Ask anything via the text input widget

        Your API-key is not stored in any form by this app. However, for transparency ensure to delete your API once used.
        ```
        
        ''')
    st.sidebar.warning('API key required to try this app.The API key is not stored in any form.')
    # st.sidebar.info("Your API-key is not stored in any form by this app. However, for transparency ensure to delete your API once used.")
```

- **Implementing a Button to Clear the memory** and calling the `new_chat()` function which we wrote about earlier,

```bash
st.sidebar.button("New Chat", on_click = new_chat, type='primary')
```

- **Get the user INPUT and RUN the chain. Also, store them** - that can be dumped in the future in a chat conversation format.

```lua
user_input = get_text()
if user_input:
    output = Conversation.run(input=user_input)  
    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)
```

- **Display the conversation history using an expander, and allow the user to download it.**

```python
# Allow to download as well
download_str = []
# Display the conversation history using an expander, and allow the user to download it
with st.expander("Conversation", expanded=True):
    for i in range(len(st.session_state['generated'])-1, -1, -1):
        st.info(st.session_state["past"][i],icon="🧐")
        st.success(st.session_state["generated"][i], icon="🤖")
        download_str.append(st.session_state["past"][i])
        download_str.append(st.session_state["generated"][i])
    
    # Can throw error - requires fix
    download_str = '\n'.join(download_str)
    if download_str:
        st.download_button('Download',download_str)
```

- **Additional features ( _not well tested ..._)**

```python
# Display stored conversation sessions in the sidebar
for i, sublist in enumerate(st.session_state.stored_session):
        with st.sidebar.expander(label= f"Conversation-Session:{i}"):
            st.write(sublist)

# Allow the user to clear all stored conversation sessions
if st.session_state.stored_session:   
    if st.sidebar.checkbox("Clear-all"):
        del st.session_state.stored_session
```

We have built the Memory Bot app ✅

**How does the app look now ? - a quick demo,**

![A quick demo of the app we build. What's new here - The MemoryBot remembers my name and country of living, which is already an added feature compared to traditional chatbots. You can also download the conversation and start a new topic. This app can be extended further. Play around with it and its settings. Have fun!](https://miro.medium.com/1*WZWR-lX88mJWQMrMQa1BYA.gif)

We can deploy our app from the local host to the DataButton server, using the publish page button (alternatively, you can also push to GitHub and serve in Streamlit Cloud ). A unique link will be generated which can be shared with anyone globally. For instance, I've deployed the Web App already in the DataButton server ( link to the [live app](https://next.databutton.com/v/lgzxq112/Memory_Bot) ).

_Refer to my YouTube video on this very similar aspect and live code each steps with me in 15 mins,_

<iframe src="https://cdn.embedly.com/widgets/media.html?src=https%3A%2F%2Fwww.youtube.com%2Fembed%2FcHjlperESbg%3Ffeature%3Doembed&display_name=YouTube&url=https%3A%2F%2Fwww.youtube.com%2Fwatch%3Fv%3DcHjlperESbg&image=https%3A%2F%2Fi.ytimg.com%2Fvi%2FcHjlperESbg%2Fhqdefault.jpg&key=a19fcc184b9711e1b4764040d3dc5c07&type=text%2Fhtml&schema=youtube" title="" height="480" width="854"></iframe>

## Conclusion

We have successfully built a Memory Bot that is well aware of the conversations and context and also provides real human-like interactions. I strongly feel this memory bot can be further personalized with our own datasets and extended with more features. Soon, I'll be coming with a new blog post and a video tutorial to explore LLM with front-end implementation.

👨🏾 ‍💻  G[itHub ](https://github.com/avrabyt)⭐️| 🐦  T[witter ](https://twitter.com/avra_b)| 📹  Y[ouTube ](https://www.youtube.com/@Avra_b)| ☕️ B[uyMeaCoffee ](https://www.buymeacoffee.com/AvraCodes)| K[o-fi💜 ](https://ko-fi.com/avrabyt)

_**Hi there! I'm always on the lookout for sponsorship, affiliate links, and writing/coding gigs to keep broadening my online content. Any support, feedback and suggestions are very much appreciated! Interested? Drop an email here: avrab.yt@gmail.com**_

_[Also consider becoming my Patreon Member ? - you'll get access to exclusive content, codes, or videos beforehand, one-to-one web app development / relevant discussion, live-chat with me on specific videos and other perks. ( FYI : Basic Tier is 50% cheaper than ChatGPT/monthly with benefits which an AI can't help with 😉  )](https://patreon.com/user?u=82100262&utm_medium=clipboard_copy&utm_source=copyLink&utm_campaign=creatorshare_creator&utm_content=join_link)_

---

### Related Blogs

1. [Getting started with LangChain - A powerful tool for working with Large Language Models](https://medium.com/@avra42/getting-started-with-langchain-a-powerful-tool-for-working-with-large-language-models-286419ba0842)

2. [Summarizing Scientific Articles with OpenAI ✨ and Streamlit](https://medium.com/@avra42/summarizing-scientific-articles-with-openai-and-streamlit-fdee12aa1a2b?source=rss-bf79cad6afa1------2)

3. [Build Your Own Chatbot with openAI GPT-3 and Streamlit](https://medium.com/@avra42/build-your-own-chatbot-with-openai-gpt-3-and-streamlit-6f1330876846?source=rss-bf79cad6afa1------2)

4. [How to ‘stream' output in ChatGPT style while using openAI Completion method](https://medium.com/@avra42/how-to-stream-output-in-chatgpt-style-while-using-openai-completion-method-b90331c15e85)

5. [ChatGPT helped me to built this Data Science Web App using Streamlit-Python](https://medium.com/@avra42/chatgpt-build-this-data-science-web-app-using-streamlit-python-25acca3cecd4?source=rss-bf79cad6afa1------2)

### Recommended YouTube Playlists

1. [OpenAI - Streamlit Web Apps](https://youtube.com/playlist?list=PLqQrRCH56DH82KNwvlWpgh3YJXu461q69)

2. [Streamlit-Python-Tutorials](https://youtube.com/playlist?list=PLqQrRCH56DH8JSoGC3hsciV-dQhgFGS1K)

### Links, references, and credits

1. LangChain Docs : [https://langchain.readthedocs.io/en/latest/index.html](https://langchain.readthedocs.io/en/latest/index.html)

2. LangChain GitHub Repo : [https://github.com/hwchase17/langchain](https://github.com/hwchase17/langchain)

3. Streamlit : [https://streamlit.io/](https://streamlit.io/)

4. DataButton : [https://www.databutton.io/](https://www.databutton.io/)

5. [Open AI document](https://platform.openai.com/docs/api-reference/completions/create#completions/create-stream)

6. [Open AI GPT 3.5 Model family](https://platform.openai.com/docs/models/gpt-3-5)

---

### Thank you for your time in reading this post! Make sure to leave your feedback and comments. See you in the next blog, stay tuned 🤖