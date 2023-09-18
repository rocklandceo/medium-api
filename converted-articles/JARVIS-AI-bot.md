### Artificial Intelligence | GPT-3

# Bring Tony Stark's JARVIS to Life: Build a Personal AI Assistant with Python, React, and GPT-3

### From Idea to Reality: How I Built My Own Personal AI Assistant Web App

![[https://labs.openai.com/s/OpIpRPXhHAB77gn6EVF0FOeJ](https://labs.openai.com/s/OpIpRPXhHAB77gn6EVF0FOeJ)](https://miro.medium.com/1*YECgOJd1rsiH0v3_Bhb4vg.png)

Do you ever wish you could have a personal AI assistant like Tony Stark's J.A.R.V.I.S. or Iron Man' F.R.I.D.A.Y to help you with your tasks, answer your questions, and keep you company?

Well, you are in luck because I have turned this sci-fi dream into a reality! That is at least some parts of it.

In this article, I will show you how to build your very own AI assistant using Python FastAPI, ReactJS, and the powerful GPT-3 language model.

You will see a live demo of the AI assistant in action and explore the system design and architecture. I will break down how the backend and frontend work, so you will have a good understanding of what's going on under the hood.

I will also provide you with the open-source Github code to get you started.

So, whether you are a seasoned developer or a curious beginner, you can build your very own AI assistant with ease.

---

## **From Idea to Reality: Turning a Dreamy AI Assistant into a Tangible Web App!**

In my previous article, _[Creating Your Own AI-Powered Second Brain](https://medium.com/gitconnected/creating-your-own-ai-powered-second-brain-a-guide-with-python-and-chatgpt-f5547ef7e136)_, I explored how to create an AI-powered second brain using Python and ChatGPT. This second brain was able to remember and organize information based on context data provided by the user.

It was a successful proof-of-concept. Looking at the reading stats and engagement, it seems like you folks found it interesting too.

In this post, we take things to the next level by actually building a personal AI assistant that you can talk to, listen to, and ask questions of - all in natural language.

Plus, with the power of GPT-3 and web scraping, this AI assistant can deliver even more insights and answers beyond the user-provided context data.

So, are you ready to revolutionize the way you work and live? Let's get started.

---

## **The Real Deal: Witness the AI Assistant in Action**

It's a little tricky to showcase the power of the assistant here on Medium.

Let me explain what are the steps involved first, and then I will leave you with a GIF and a YouTube video to see the full thing in action. Here are the steps:

1. Hover your mouse on the "Say Something" button

2. Recording starts

3. Speak your question towards the microphone

4. Move your mouse cursor away from the button

5. Backend magic happens!

6. Your AI assistant speaks the answer to you through the speakers

7. You also get a text transcript of the question/answer in the UI

To get the best experience, I would recommend checking the 50 seconds [YouTube video below](https://youtu.be/ncTRdQPs0Ug).

<iframe src="https://cdn.embedly.com/widgets/media.html?src=https%3A%2F%2Fwww.youtube.com%2Fembed%2FncTRdQPs0Ug&display_name=YouTube&url=https%3A%2F%2Fwww.youtube.com%2Fwatch%3Fv%3DncTRdQPs0Ug&image=http%3A%2F%2Fi.ytimg.com%2Fvi%2FncTRdQPs0Ug%2Fhqdefault.jpg&key=a19fcc184b9711e1b4764040d3dc5c07&type=text%2Fhtml&schema=youtube" title="" height="480" width="854"></iframe>

Given the audio/video nature of the feature, a GIF won't really do it much justice, but here's one if you don't prefer YouTube:

![Author's generated GIF to showcase the personal assistant web app's functionality](https://miro.medium.com/1*2eKVrYZ7rCo6R0Gj6otDyg.gif)

---

## **Behind the Scenes: A Look at the System Design and Architecture of the AI Assistant**

Now, let's move our attention to the technical details.

If I had to break down the system into multiple components, this is how it would look like:

- GPT-3 as the Large Language Model (LLM)

- Llama-Index to vectorize context data and pass it to GPT-3

- Python FastAPI server to interact with the trained LLM model

- ReactJS & ChakraUI to build a frontend UI

- Webkit SpeechRecognition library for voice input

- Webkit SpeechSynthesisUtterance library for text to speech

If you put all these together, this is how the system looks.

![Author's created system design diagram on Miro](https://miro.medium.com/1*4F_y6YDpHNtpkh5UddHxJQ.png)

Read the system design diagram from left to right, top to bottom.

Now that you have a bigger picture idea about how the system works, let's zoom into both the frontend and backend separately to get a deeper understanding.

---

## **The Backend: Learn How the Python FastAPI and GPT-3 Powers the AI Assistant**

In the last few months, ChatGPT has taken over the world, quite literally.

You can ask it to do your homework, prepare your presentations, write your SQL queries, help you write code, generate realistic images and videos - the list goes on and on.

Even though it can do all these different things, it still struggles when you ask it questions about your life - what did you eat yesterday? Who did you meet last week? Did you buy your medicines?

ChatGPT cannot answer these because it does not have any visibility into your personal life.

You need to give it your personal data for it to be able to help you. That's where the following comes in:

- A text file with your journal entries

- Llama Index to read this text file, vectorize the data, and pass it as context to GPT-3

A combination of these two gives GPT-3 what it needs to answer any question you have about your personal life.

Of course, it's not magic. You need the data to be existing in the journal to begin with, for GPT-3 to be able to help you.

The first step is to train the GPT-3 model on this data. Next, you save the trained model on your server.

```python
from llama_index import GPTSimpleVectorIndex, SimpleDirectoryReader

# Load data from the journal text file
documents = SimpleDirectoryReader("./data").load_data()

# Create a simple vector index
index = GPTSimpleVectorIndex(documents)
index.save_to_disk("generated_index.json")

# Create an infinite loop asking for user input and then breaking out of the loop when the response is empty
while True:
    query = input("Ask a question: ")
    if not query:
        print("Goodbye")
        break
    # query the index with the question and print the result
    result = index.query(query)
    print(result)
```

Now, you build a very simple FastAPI endpoint to interact with this saved model. The endpoint logic is straightforward:

1. Pass the user question coming from the web app

2. Ask the question to the saved GPT-3 model

3. Return the answer to the client in JSON

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from llama_index import GPTSimpleVectorIndex


app = FastAPI()

# Define allowed origins
origins = [
    "http://localhost:3000",
    "http://localhost:5000",
    "http://localhost:8000",
    "http://localhost:8080",
]
# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/answers")
async def get_answer(question: str):
    index = GPTSimpleVectorIndex.load_from_disk("generated_index.json")
    answer = index.query(question)

    return {"answer": answer.response}
```

After this, the ball is in the client's court to present the data back to the user. Let's see how that's done.

---

## **The Frontend: Discover How ReactJS Brings the AI Assistant to Life with a Stunning UI**

The user interacts with the web app.

The web app has fundamentally 4 jobs:

1. Take the user's question through the microphone and turn it into text

2. Pass the question to the server through an API call

3. Transform the answer coming from the server from text to speech, and produce output through the user's speakers

4. Show the transcript to the user when doing speech-to-text and text-to-speech

```javascript
import React, { useState, useEffect } from "react";
import { Button, VStack, Center, Heading, Box, Text } from "@chakra-ui/react";

function App() {
  const [transcript, setTranscript] = useState("");
  const [answer, setAnswer] = useState("");
  const [isRecording, setIsRecording] = useState(false);
  const [buttonText, setButtonText] = useState("Say Something");
  const [recognitionInstance, setRecognitionInstance] = useState(null);

  useEffect(() => {
    const recognition = new window.webkitSpeechRecognition();
    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.lang = "en-US";

    recognition.onresult = (event) => {
      let interimTranscript = "";
      let finalTranscript = "";

      for (let i = event.resultIndex; i < event.results.length; i++) {
        const transcript = event.results[i][0].transcript;
        if (event.results[i].isFinal) {
          finalTranscript += transcript + " ";
        } else {
          interimTranscript += transcript;
        }
      }

      setTranscript(finalTranscript);
    };

    setRecognitionInstance(recognition);
  }, []);

  const recordAudio = () => {
    setAnswer("");
    setButtonText("Recording...");
    setIsRecording(!isRecording);
    recognitionInstance.start();
  };

  const stopAudio = async () => {
    setButtonText("Say Something");
    setIsRecording(!isRecording);
    recognitionInstance.stop();

    const response = await fetch(
      `http://127.0.0.1:8000/answers?question=${transcript}`
    );
    const data = await response.json();
    setAnswer(data["answer"]);

    const utterance = new SpeechSynthesisUtterance(data["answer"]);

    window.speechSynthesis.speak(utterance);
  };

  return (
    <Box
      bg="black"
      h="100vh"
      display="flex"
      justifyContent="center"
      alignItems="center"
      padding="20px"
    >
      <Center>
        <VStack spacing={12}>
          <Heading color="red.500" fontSize="8xl">
            👋  I am your personal assistant 🤖
          </Heading>
          <Button
            colorScheme="red"
            width="300px"
            height="150px"
            onMouseOver={recordAudio}
            onMouseLeave={stopAudio}
            fontSize="3xl"
          >
            {buttonText}
          </Button>
          {transcript && (
            <Text color="whiteAlpha.500" fontSize="2xl">
              Question: {transcript}
            </Text>
          )}
          {answer && (
            <Text color="white" fontSize="3xl">
              <b>Answer:</b> {answer}
            </Text>
          )}
        </VStack>
      </Center>
    </Box>
  );
}

export default App;
```

There are some really fancy libraries you can use to generate amazing AI voices.

I kept things very simple and used Webkit libraries that are baked into the browser.

---

## **Give it a Go: Get Started with Your Own Personal AI Assistant Today!**

If you have reached this far into the article, thank you so much for reading.

I hope you found this valuable and insightful.

I open-sourced the on my personal **[GitHub repo](https://github.com/irtiza07/personal-assistant-ai-www)**. If you know your way around code, I would highly suggest cloning it and getting started with your own personal AI assistant!

I will close it with a pro tip: To make the assistant most helpful, export data from your task manager and calendar, and put it in your text file. I trained my model on data from ClickUp and Google Calendar. It was insanely useful!

Super excited to hear from you :) Thank you for reading.

If you enjoyed it, please consider clapping and following me on Medium.

---

## Level Up Coding

Thanks for being a part of our community! Before you go:

- 👏  Clap for the story and follow the author 👉 

- 📰  View more content in the [Level Up Coding publication](https://levelup.gitconnected.com/?utm_source=pub&utm_medium=post)

- 💰  Free coding interview course ⇒ [View Course](https://skilled.dev/?utm_source=luc&utm_medium=article)

- 🔔  Follow us: [Twitter](https://twitter.com/gitconnected) | [LinkedIn](https://www.linkedin.com/company/gitconnected) | [Newsletter](https://newsletter.levelup.dev)

🚀👉  J**[oin the Level Up talent collective and find an amazing job](https://jobs.levelup.dev/talent/welcome?referral=true)**