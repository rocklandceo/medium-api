# All You Need to Build Your Custom Chatbot with Next.js, OpenAI, and Supabase: Part I

![Photo by [Jason Leung](https://unsplash.com/@ninjason?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)](https://miro.medium.com/0*I1bDRecwAKUyNCVG)

Ever thought about having a powerful yet easy-to-build chatbot that could be trained with your own data for your business? Get excited, because that's exactly what we'll cover in this Part I of our two-part series on how to create a custom chatbot in Next.js using OpenAI and Supabase! In Part II, we'll add depth to the chatbot by utilizing embeddings.

Vercel AI SDK offers an extraordinary set of tools to build AI-powered user interfaces using Next.js and OpenAI's API. In this educational blog post, we'll explore how to set up AI and OpenAI-Edge in your Next.js application.

## Prerequisites

Before jumping into action, ensure you have the following:

- Node.js 18+ installed on your local development machine.

- An OpenAI API key. You can get one by signing up on the [OpenAI website](https://beta.openai.com/signup/).

## Creating a Next.js Application

Let's kick off our journey by creating a brand-new Next.js application. Run the following command in your terminal. This command generates a new directory called `my-ai-app` and sets up a basic Next.js application inside it:

```lua
npx create-next-app custom-chatgpt --tailwind
```

Now, navigate to the shiny, newly created directory:

```bash
cd custom-chatgpt
```

## Installing Dependencies

To prep our environment, install the AI and OpenAI-Edge dependencies. OpenAI-Edge is preferred over the official OpenAI SDK since it's compatible with Vercel Edge Functions:

```typescript
npm install ai openai-edge
```

## Configuring the OpenAI API Key

Time to connect your application to the OpenAI engine. Create a `.env.local` file in your project root and add your OpenAI API key. This key allows your application to communicate with the OpenAI service:

```ini
OPENAI_API_KEY=your_api_key_here
```

## Setting Up OpenAI

We have to set up a route handler in Next.js for managing OpenAI. Create a file called `utils/openai.ts` and import the essential components:

```javascript
// utils/openai.ts

import { Configuration, OpenAIApi } from "openai-edge";

const config = new Configuration({
  apiKey: process.env.OPENAI_API_KEY,
});

export const openai = new OpenAIApi(config);import { Configuration, OpenAIApi } from "openai-edge";const config = new Configuration({ apiKey: process.env.OPENAI_API_KEY,});export const openai = new OpenAIApi(config);
```

## Creating an API Route

Our next step is to create a Next.js route handler in a new file called `app/api/chat/route.ts`. This handler employs the Edge Runtime to generate chat messages through OpenAI, passing them back to Next.js:

```javascript

// app/api/chat/route.ts

import { openai } from "@/utils/openai";
import { OpenAIStream, StreamingTextResponse } from "ai";
export const runtime = "edge";

export async function POST(req: Request) {
  // Extract the `messages` from the body of the request
  const { messages } = await req.json();
  // Request the OpenAI API for the response based on the prompt
  const response = await openai.createChatCompletion({
    model: "gpt-3.5-turbo",
    stream: true,
    messages: messages,
  });

  // Convert the response into a friendly text-stream
  const stream = OpenAIStream(response);

  // Respond with the stream
  return new StreamingTextResponse(stream);
}
```

## Creating UI Components

The AI SDK incorporates hooks like `useChat` and `useCompletion` for your client components. To use the edge runtime directly, we will create a wrapper with a React Server Component:

In `app/page.tsx` on the server component side, implement the Chat component and export the runtime as 'edge':

```javascript
// app/page.tsx

import Chat from "./chat";

export const runtime = "edge";

export default function Page() {
  return (
    <div className="h-full w-full p-8 flex">
      <Chat />
    </div>
  );
}
```

Next, in the client-side `app/chat.tsx`, embrace the 'use client' directive and the `useChat` hook from 'ai/react'. This hook fuels dynamic chat functionality:

```javascript
// app/chat.tsx

"use client";
import { useChat } from "ai/react";

export default function MyComponent() {
  const { messages, input, handleInputChange, handleSubmit } = useChat({
    api: "/api/chat",
  });

  return (
    <div className="bg-gray-100 p-4 rounded shadow-md m-auto w-full">
      <ul className="space-y-2">
        {messages.map((m, index) => (
          <li key={index} className="p-2 rounded bg-white shadow text-gray-700">
            <span className="font-semibold">
              {m.role === "user" ? "User: " : "AI Assistant: "}
            </span>
            {m.content}
          </li>
        ))}
      </ul>

      <form onSubmit={handleSubmit} className="mt-4">
        <label className="block text-gray-700 text-sm font-bold mb-2">
          Write here ...
          <input
            value={input}
            onChange={handleInputChange}
            className="w-full p-2 mt-1 rounded border shadow-sm"
          />
        </label>
        <button
          type="submit"
          className="mt-2 bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded focus:outline-none focus:shadow-outline"
        >
          Submit
        </button>
      </form>
    </div>
  );
}
```

With this assembly, the server-side Page function returns the Chat component configured with edge runtime. In contrast, the client-side component harnesses the `useChat` hook for dynamic chat functionality.

## Running the Application

You're only one step away from having your chatbot up and running! Start your application with this command:

```typescript
npm run dev
```

![](https://miro.medium.com/1*7JnAOP5ZCuGEoTEsk7-shA.png)

Your AI-powered chatbot is now live! Thanks to AI and OpenAI-Edge, creating user interfaces that make a real impact has never been simpler. Now, you can develop and deploy AI-based user interfaces in your Next.js applications.

Feel inspired? Great! Now, go forth and unleash the power of AI to create engaging and intelligent chatbots for your business. And don't forget to stay tuned for Part II, where we'll amp up our chatbot with embeddings!

You can access the complete source code for our custom chatbot on GitHub

### GitHub Repo

> [**GitHub - abdelfattah-sekak/custom-chabot**](https://github.com/abdelfattah-sekak/custom-chabot)

### Part II

> [**All You Need to Build Your Custom Chatbot with Next.js, OpenAI, and Supabase : Part II**](https://medium.com/@abdelfattah.sekak/all-you-need-to-build-your-custom-chatbot-with-nextjs-openai-and-supabase-part-ii-7e4270cb5ddf)

_**If you find yourself with any questions or simply wish to chat, don't hesitate to share your thoughts below or reach out to me directly at:**_

**Email** : abdelfattah.sekak@gmail.com

**LinkedIn** : [https://www.linkedin.com/in/abdelfattah-sekak-760847141/](https://www.linkedin.com/in/abdelfattah-sekak-760847141/)

## In Plain English

_Thank you for being a part of our community! Before you go:_

- _Be sure to **clap** and **follow** the writer! 👏_ 

- _You can find even more content at **[PlainEnglish.io](https://plainenglish.io/) 🚀**_ 

- _Sign up for our **[free weekly newsletter](http://newsletter.plainenglish.io/)**. 🗞 ️_

- _Follow us on **[Twitter](https://twitter.com/inPlainEngHQ)**_, _**[LinkedIn](https://www.linkedin.com/company/inplainenglish/)**_, _**[YouTube](https://www.youtube.com/channel/UCtipWUghju290NWcn8jhyAw)**_, and _**[Discord](https://discord.gg/GtDtUAvyhW).**_