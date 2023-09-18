# All You Need to Build Your Custom Chatbot with Next.js, OpenAI, and Supabase: Part II

![Photo by [Jason Leung](https://unsplash.com/@ninjason?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)](https://miro.medium.com/0*7gxbVtRa4G3OEkoM)

Hey there, friends! Do you recall our fascinating conversation in [Part I](https://medium.com/@abdelfattah.sekak/all-you-need-to-build-your-custom-chatbot-with-nextjs-openai-and-supabase-part-i-26158cf688fa) about crafting custom chatbots? If you enjoyed that, you're in for a treat! Today, we're going to breathe life into that chatbot using the incredible power of OpenAI embeddings. Together, we'll delve into the process of constructing a robust and scalable search solution using PostgreSQL for storing our valuable embeddings. Plus, we'll guide you through setting up an API that interacts seamlessly with our search system. So, are you ready to embark on this exhilarating adventure? Let's dive right in and make some magic happen!

## II. Storing and Managing Embeddings in PostgreSQL

You might be wondering how we're going to implement all this magic in PostgreSQL, right? Worry not, this is where **pgvector** comes in. Pgvector is an amazing extension for PostgreSQL that allows you to store and query vector embeddings within your database. I've got your back; let's set it up together, step by step!

### Set up supabase account

Excited to explore Supabase, the open-source Firebase alternative? Well, your adventure begins right here, right now! Head over to the Supabase website and hit the ‘Start your Project' button. Sign up using GitHub, GitLab, or Bitbucket, or use your favorite email.

Next, create a new project. Supabase will then provide you with a secure URL and API key - your secret tools for accessing your backend. Voilà! You're now part of the Supabase community

![Supabase - Project creation](https://miro.medium.com/1*3aqp6tp1naHlWKrBcvtYRA.png)

![Supabase - Project keys](https://miro.medium.com/1*zFkMRslABXR1nIfj8K9x_A.png)

### Enable the Vector Extension

First things first, we need to enable the Vector extension in our environment. If you happen to use Supabase, it's as easy as clicking a few buttons on the web portal! Simply head over to **Database** → **Extensions**, and you're all set. But if you're on SQL, just run this nifty command:

```cpp
create extension vector;
```

![](https://miro.medium.com/1*6kRg7y9elqTi4GUYYB2fJw.png)

### Create a Table for Your Documents and Embeddings

Next up, we're going to create a cozy home for our documents and embeddings. Here's a simple recipe to cook up that table:

```sql
create table documents (
  id bigserial primary key,
  content text,
  embedding vector(1536)
);
```

![](https://miro.medium.com/1*PxK5s3i5cZjgQ6GlzbZcfQ.png)

Pgvector introduces an incredibly versatile new data type called `vector`. In the code above, we create a column named `embedding` with this vector data type. Our dear OpenAI's `text-embedding-ada-002` model outputs 1536 dimensions, so that's the number we're using for our vector size. We also have a text column named `content` to store the original document text that generated the embedding. However, you can always customize and store a reference to a document instead, depending on your needs.

## III. Creating a Similarity Search Function

With our table set up, let's create a search function to find similar documents using embeddings in PostgreSQL. This is the key to unlocking the powerful search capabilities we've been chasing!

```sql
create or replace function match_documents (
  query_embedding vector(1536),
  match_threshold float,
  match_count int
)
returns table (
  id bigint,
  content text,
  similarity float
)
language sql stable
as $$
select
documents.id,
documents.content,
1 - (documents.embedding <=> query_embedding) as similarity
from documents
where 1 - (documents.embedding <=> query_embedding) > match_threshold
order by similarity desc
limit match_count;
$$;
```

![](https://miro.medium.com/1*9QwYuPnq1knw7udDNWMvdg.png)

## IV. Generating Embeddings with JavaScript

To generate embeddings for our documents and store them in our PostgreSQL database, we'll use JavaScript to interact with the OpenAI API and our PostgreSQL client. But first, let's install the necessary dependencies with the following command:

```scss
npm install @supabase/supabase-js langchain puppeteer
```

Next, add your Supabase keys to the `.env.local` file:

```ini
OPENAI_API_KEY=your_api_key_here
SUPABASE_URL=your_supabase_url_here
SUPABASE_SECRET_KEY=your_supabase_secret_key_here
```

### Configuring Supabase Client Instance

To configure your Supabase client instance, include the following snippet in `utils/supabase.ts`:

```typescript
// utils/supabase.ts

import { createClient } from "@supabase/supabase-js";

export const supabase = createClient(
  process.env.SUPABASE_URL!,
  process.env.SUPABASE_SECRET_KEY!,
  {
    auth: {
      persistSession: false,
    },
  }
);
```

Now, let's create a function to scrape data from a website using langchain's utility functions. In our example, we'll use my finance blog, [www.swstock.com](http://www.swstock.com), to collect data and train the chatbot to answer questions about the content.

```typescript
// scripts/getDocuments.ts

import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { convert } from "html-to-text";
import { PuppeteerWebBaseLoader } from "langchain/document_loaders/web/puppeteer";

const splitter = new RecursiveCharacterTextSplitter({
  chunkSize: 2000,
  chunkOverlap: 100,
});

const URL_EXAMPLE = "https://www.swstock.com";

export const getDocuments = async () => {
  const loader = new PuppeteerWebBaseLoader(URL_EXAMPLE);
  console.info(`Getting URL from ${URL_EXAMPLE}`);
  const docs = await loader.load();
  return await splitter.createDocuments(
    docs.map((doc) => {
      const text = convert(doc.pageContent).replace(/\n/g, " ");
      return text;
    })
  );
};
```

### Generate Embeddings with OpenAI and Store in PostgreSQL

Next, let's generate some embeddings with OpenAI's glorious API and store them in our PostgreSQL database.

```javascript
// scripts/generateEmbeddings.ts

import { getDocuments } from "./getDocuments";
import { openai } from "../utils/openai";
import { supabase } from "../utils/supabase";

export async function generateEmbeddings() {
  const documents = await getDocuments(); // Your custom function to load docs
  for (const document of documents) {
    // OpenAI recommends replacing newlines with spaces for best results
    const input = document.pageContent.replace(/\n/g, " ");
    try {
      const embeddingResponse = await (
        await openai.createEmbedding({
          model: "text-embedding-ada-002",
          input,
        })
      ).json();
      const [{ embedding }] = embeddingResponse.data;
      // In production we should handle possible errors
      await supabase.from("documents").insert({
        content: input,
        embedding,
      });
    } catch (error) {
      console.error(error);
    }
  }
}

generateEmbeddings();
```

To start the process, set up this function to run with Node.js:

```undefined
npm install node-ts
```

Now, add the following command to your `package.json` scripts section:

```rust
// package.json
....
"scripts": {
"dev": "next dev",
"build": "next build",
"start": "next start",
"lint": "next lint",
"generate-embeddings": "ts-node --esm ./scripts/generateEmbeddings.ts"
},
...
```

Ready for action? Run the following command in your terminal:

```undefined
EXPORT OPENAI_API_KEY=your_api_key_here
EXPORT SUPABASE_URL=your_supabase_url_here
EXPORT SUPABASE_SECRET_KEY=your_supabase_secret_key_here
npm run generate-embeddings
```

## V. Creating a Search Function

We're almost there! Now that we've got our embeddings stored, let's create a function that performs a similarity search against your pre-processed embeddings and injects similar content in response to the user's query.

```typescript
// app/api/chat/injectCustomData.ts

import { openai } from "@/utils/openai";
import { supabase } from "@/utils/supabase";
import { ChatCompletionRequestMessage } from "openai-edge";

export const injectCustomData = async (
  messages: ChatCompletionRequestMessage[]
) => {
  const lastMessage = messages.pop();
  if (!lastMessage) {
    return messages;
  }
  const input = lastMessage.content;
  const embeddingResponse = await (
    await openai.createEmbedding({
      model: "text-embedding-ada-002",
      input: input!,
    })
  ).json();
  const [{ embedding }] = embeddingResponse.data;
  const { data: documents } = await supabase.rpc("match_documents", {
    query_embedding: embedding,
    match_threshold: 0.78, // Choose an appropriate threshold for your data
    match_count: 10, // Choose the number of matches
  });
  let contextText = "";
  for (let i = 0; i < documents.length; i++) {
    const document = documents[i];
    const content = document.content;
    contextText += `${content.trim()}---\n`;
  }
  const prompt = `
        You are a representative that is very helpful when it comes to talking about SW Stock! Only ever answer
        truthfully and be as helpful as you can!"
        Context: ${contextText}
        Question: """
        ${input}
        """
        Answer as simple text:
      `;
  return [
    ...messages,
    {
      role: "user",
      content: prompt,
    },
  ] as ChatCompletionRequestMessage[];
};
```

## VI. Modifying Route Handler to Send Requests to OpenAI with Custom Data

Finally, let's tweak our route handler to send requests with our custom data to OpenAI.

```javascript
// app/api/chat/route.ts

import { openai } from "@/utils/openai";
import { OpenAIStream, StreamingTextResponse } from "ai";
import { injectCustomData } from "./injectCustomData";

export const runtime = "edge";

export async function POST(req: Request) {
  // Extract the `messages` from the body of the request
  const { messages } = await req.json();
  const messagesWithCustomData = await injectCustomData(messages);
  // Request the OpenAI API for the response based on the prompt
  const response = await openai.createChatCompletion({
    model: "gpt-3.5-turbo",
    stream: true,
    messages: messagesWithCustomData,
  });

  // Convert the response into a friendly text-stream
  const stream = OpenAIStream(response);

  // Respond with the stream
  return new StreamingTextResponse(stream);
}
```

![](https://miro.medium.com/1*oIKsqjZahZTCr1C0xsrH5g.png)

And there you have it! You now wield the power of a custom chatbot powered by OpenAI embeddings and PostgreSQL. Be ready to dazzle your users with advanced chat capabilities and an exceptional chatting experience. Happy chatting!

## Git Repository

Interested in diving further into the code that brings our custom chatbot to life? No problem!

You can access the complete source code for our custom chatbot on GitHub

> [**GitHub - abdelfattah-sekak/custom-chabot**](https://github.com/abdelfattah-sekak/custom-chabot)

This repository allows you to delve into each line of code at your own pace, giving you a deeper understanding of how the chatbot comes together. This hands-on examination will further enhance your knowledge acquired from our tutorial.

Remember, GitHub is a great treasure trove of learning material. So, go ahead and explore the repository, clone it, run it locally, or even contribute by making it even better! Collaboration is the key to growth in the open-source community.

_**If you find yourself with any questions or simply wish to chat, don't hesitate to share your thoughts below or reach out to me directly at:**_

**Email** : abdelfattah.sekak@gmail.com

**LinkedIn** : [https://www.linkedin.com/in/abdelfattah-sekak-760847141/](https://www.linkedin.com/in/abdelfattah-sekak-760847141/)