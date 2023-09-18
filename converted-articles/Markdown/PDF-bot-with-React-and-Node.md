# Building a PDF Chatbot with React and Node

![](https://miro.medium.com/1*M7OT2mkks6TBkjD_KfaoIg.jpeg)

Hi EveryOne, How are you ? 🖐

In this time, I am going to let you through the process of building a PDF chatbot using React and Node. We'll be using the GPT-3 API to interact with the chatbot and generate responses.

### Prerequisites

Before we get started, make sure you have the following:

- Node.js installed on your machine

- A text editor (such as VS Code)

- An OpenAI API key

### Setting up the Project

To get started, create a new directory for your project and navigate to it in your terminal. Then, run the following command to initialize a new Node.js project:

```bash
npm init -y  
```

Next, install the necessary dependencies:

```bash
npm install express openai pdfkit
```

We'll be using the Express framework for our server, the OpenAI SDK for interacting with the GPT-3 API, and the PDFKit library for generating PDFs.

### Creating the Server

Create a new file called `server.js` in your project directory. First, import the necessary modules:

```javascript
const express = require('express');
const openai = require('openai');
const PDFDocument = require('pdfkit');
```

Next, create a new instance of the Express app:

```cpp
const app = express();
```

Now, create a new instance of the OpenAI API client:

```csharp
const client = new openai.OpenAI('<YOUR_API_KEY>');
```

Replace `<YOUR_API_KEY>` with your own OpenAI API key.

Create a route that listens for POST requests to `/chat`:

```dart
app.post('/chat', async (req, res) => {
  const { message } = req.body;

  const response = await generateResponse(message);

  res.json({ response });
});
```

This route takes a `message` parameter from the request body and sends it to GPT-3 for completion. It then returns the completed text as a JSON response.

Create a function that sends a prompt to GPT-3 and returns its response:

```php
async function generateResponse(prompt) {
  const response = await client.completions.create({
    engine: 'davinci',
    prompt,
    maxTokens: 1024,
    n: 1,
    stop: '\n',
  });

  return response.choices[0].text.trim();
}
```

This function takes a `prompt` argument and sends it to GPT-3 for completion. It returns the completed text as a string.

### Creating the PDF

Create a new route that listens for GET requests to `/pdf`:

```dart
app.get('/pdf', async (req, res) => {
  const { message } = req.query;

  const response = await generateResponse(message);

  const doc = new PDFDocument();

  doc.pipe(res);

  doc.fontSize(16).text(response);

  doc.end();
});
```

This route takes a `message` query parameter and sends it to GPT-3 for completion. It then generates a PDF document with the completed text and streams it back to the client.

### Creating the Frontend

Create a new file called `App.js` in your project directory. First, import the necessary modules:

```javascript
import React, { useState } from 'react';
import axios from 'axios';
```

Next, create a new component that renders a form for sending messages to the server:

```javascript
function App() {
  const [message, setMessage] = useState('');
  const [response, setResponse] = useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();

    const { data } = await axios.post('/chat', { message });

    setResponse(data.response);
  };

  return (
    <div>
      <form onSubmit={handleSubmit}>
        <input type="text" value={message} onChange={(e) => setMessage(e.target.value)} />
        <button type="submit">Send</button>
      </form>

      {response && (
        <div>
          <p>{response}</p>
          <a href={`/pdf?message=${encodeURIComponent(response)}`} target="_blank" rel="noopener noreferrer">Download PDF</a>
        </div>
      )}
    </div>
  );
}
```

This component renders a form with an input field for sending messages to the server. When the form is submitted, it sends a POST request to `/chat` with the message as the request body. It then displays the response from the server and a link to download a PDF of the response.

### Running the Server

To run the server, start it by running the following command in your terminal:

```typescript
node server.js
```

This will start the server and listen for requests on port 3000.

### Running the Frontend

To run the frontend, create a new file called `index.html` in your project directory with the following contents:

```xml
<!DOCTYPE html>
<html>
  <head>
    <title>PDF Chatbot</title>
  </head>
  <body>
    <div id="root"></div>

    <script src="https://unpkg.com/react/umd/react.production.min.js"></script>
    <script src="https://unpkg.com/react-dom/umd/react-dom.production.min.js"></script>
    <script src="https://unpkg.com/axios/dist/axios.min.js"></script>
    <script src="App.js"></script>

    <script>
      ReactDOM.render(React.createElement(App), document.getElementById('root'));
    </script>
  </body>
</html>
```

This file loads the necessary dependencies and renders the `App` component.

To view the frontend, open `index.html` in your web browser.

---

### Conclusion

In this blog, we showed you how to build a PDF chatbot using React and Node. With GPT-3's powerful natural language processing capabilities, you can create chatbots that can hold conversations and generate PDFs with ease.

I hope it helps your work 😘