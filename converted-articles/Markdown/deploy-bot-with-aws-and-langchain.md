# Deploying an AI Q/A Bot on AWS with LangchainJS

### Quick guide to create a company Q/A bot with custom data in 8 minutes leveraging OpenAI, Langchain and Serverless

![The various tools we'll use to deploy a NodeJS script with Langchain to AWS](https://miro.medium.com/1*CJkzGPFzcgrhXkW1tkPFDg.png)

I've got the AI bug, so I'm finding all the ways we can build a Langchain NodeJS backend. If you're new to Langchain look [here](https://www.youtube.com/watch?v=nE2skSRWTTs), this video is a great introduction. But essentially, it will allow us to build AI apps faster.

My aim is to quickly develop Q/A bots that will use information from text files to answer questions. The application should be reusable with any text file. I'll use OpenAI for both the embeddings and LLM model. Langchain is used to simplify the process. AWS seems like a good choice to deploy to as it has a generous free tier. I'm using the Serverless Framework which will simplify infrastructure management letting us deploy faster.

If you follow along you should be done within a few minutes. The end result will be two endpoints. The first endpoint will convert text files into embeddings.

```bash
curl -X POST "https://YOUR_AWS_POST_URL_HERE/dev/process" \
     -H "Content-Type: application/json" \
     -H "x-api-key: YOUR_API_KEY_HERE" \
     -d '{ "bucketName": "my-langchain-bucket", "key": "customer_service_questions.txt" }'
```

```json
{
  "message": "Embeddings generated and uploaded successfully",
  "bucket": "my-langchain-bucket",
  "directory": "embeddings"
}
```

The second endpoint will let us ask questions using those embeddings. It will also allow us to send in history with the request.

```bash
curl -X POST "https://YOUR_AWS_POST_URL_HERE/dev/question" \
     -H "Content-Type: application/json" \
     -H "x-api-key: YOUR_API_KEY_HERE" \
     -d '{ "question": "can I pay with paypal?", "chatHistory": "", "bucketName": "my-langchain-bucket" }'
```

```json
{
    "text": "Yes, you can pay with PayPal. You can find this option on the payment section during checkout."
}
```

I've added in the option of sending a system and prompt template with the requests. Please see the documentation on how to best structure your calls [here.](https://github.com/ilsilfverskiold/langchainjs-aws-service) This will allow you to change its personality a bit.

These endpoints will work with any files so they can be reusable. We can connect it to a React chatbot using [this](https://github.com/ilsilfverskiold/react-langchain-chatbot) repository at the end so we can interact with it using a nice UI.

![](https://miro.medium.com/1*6EOeYKajH6Wu_uwJJ8UxCQ.gif)

You can also obviously connect it to Slack or Discord or so. This is just an example.

The repository we'll work with you'll find [here](https://github.com/ilsilfverskiold/langchainjs-aws-service). This assumes you have a well structured .txt file. If you are working with other files rather than a .txt file, you'll need to tweak the code a bit. The file I'll be working with to test [[this](https://github.com/ilsilfverskiold/langchainjs-aws-service)](https://github.com/ilsilfverskiold/langchainjs-aws-service/blob/main/customer_service_questions.txt) is this customer service questions file. If you are working with a URL, Cloudflare Workers may be faster for you. See this repository.

I had to struggle a bit with the new AWS SDK v3 that comes with NodeJS 18 but this script should work well, so you're welcome to grab it. On a side note, there may be easier ways to do this but it was a good exercise nonetheless.

### Sections

1. Langchain & OpenAI Embeddings

2. Setup & Deployment

3. Results

If you want to build directly scroll down to **Setup & Deployment.**

## Langchain & OpenAI Embeddings

This article will use the Serverless Framework to create two HTTP Post requests with functions that will allow you to do the following.

1. Transform text files to embeddings that will allow you to query them.

2. Send in a question and return a response based on a text file that you've created embeddings for.

### Embeddings

The whole internet has information about embeddings at this point. So it should be easy to find more information on how embeddings work.

But essentially, we're splitting up text into chunks and then setting embeddings using OpenAI's Ada-002 to each of those chunks of text. Embeddings are arrays filled with vectors representing the semantic content of the text. With these embeddings in place, we can then compare a given question to a piece of text using cosine similarity. This will allow us to search content and decide which piece of text is more likely to be relevant.

We use OpenAI both for our embeddings and for the LLM model here but Langchain simplifies the process.

### Creating an Embedding Store with a Text File

The first process of this is to split document into chunks, set the embeddings of each chunk with OpenAI and create a vector store. I.e. instead of having a large piece of text we'll now have several documents with their own set of embeddings.

We do this with node-faiss via Langchain.

This will create two files that we store to S3 in AWS.

![The first part of this process is turning our .text files to a vector store that we store in S3](https://miro.medium.com/1*wXY8LArY8uRDbzQo4EY7Lg.png)

### Searching Content Based On Prompt

The second part of this is to load the vector store when we receive a question (i.e. a prompt) and search the vector store for the piece of text that is most similar from the entire text we used at the start.

This will give us context.

![When we receive a prompt we'll use it to search the vector store to get out a piece of text from the text file that is most likely to be relevant (we'll call this result "context")](https://miro.medium.com/1*75NS33bBuQiYDAzFaOvq0w.png)

### Using Context with Prompt

When we have a piece of text that is most likely to be relevant to the question asked, we'll use it to call the LLM (OpenAI's GPT-Turbo-3.5 or GPT-4 in this case) along with potential history to receive a response.

![We use context along with the prompt and history to receive a response back from the LLM (in this case OpenAI's GPT-3.5)](https://miro.medium.com/1*pA9vjwH9ABfZiR1o5nXEbQ.png)

If that explained it, you can look into the script files [processEmbeddings.mjs ](https://github.com/ilsilfverskiold/langchainjs-aws-service/blob/main/processEmbeddings.mjs)and [processQuestion.mjs](https://github.com/ilsilfverskiold/langchainjs-aws-service/blob/main/processQuestion.mjs) to see how this is done.

To clarify a bit more, see the demonstration below.

```json
// prompt
"question": "can I pay with paypal?"

// this piece of text returns with the highest score from the text file
"context": "a: You can pay using common methods like Visa, Mastercard, and PayPal. There are also local payment options based on your shipping country which you can find on the payment section during checkout. Once approved, the total amount gets deducted from your account. All transactions are secured and managed by Adyen international payment services, which ensure safe and encrypted connections."

// using context, the prompt in a request to the LLM this is the response we get
"response": "Yes, you can pay with PayPal. You can find this option on the payment section during checkout."
```

The user can keep asking it questions and it will answer based on this file.

I know, you would have thought it was more magical than this. But no, this is how we make these Q/A bots.

It is quite efficient though.

## Setup & Deployment

To do this we'll use Serverless to deploy our application on AWS. You'll need an AWS account for this.

If you don't have an [AWS account](https://aws.amazon.com/), go and sign up for one. Make sure you enable MFA as well. Technically you should now create an admin user and work from there but that is up to you. Maybe create a billing alert under Billing → Budgets though.

### Create Your IAM User

This will probably be the most tedious part as AWS is big on security and you need to navigate with care.

Navigate to **IAM.** Create a new user and name it whatever you'd like. It will **not** need access to the management console.

Under permissions, you are looking for `attach policies directly` and then create policy.

![](https://miro.medium.com/1*-6-vuHq_TJZtduT-Vks-lQ.png)

Under specify permissions choose JSON and paste in the permissions below.

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "cloudformation:*",
                "lambda:*",
                "apigateway:*",
                "logs:*",
                "s3:*",
                "iam:GetRole",
                "iam:CreateRole",
                "iam:PassRole",
                "iam:PutRolePolicy",
                "iam:AttachRolePolicy"
            ],
            "Resource": "*"
        }
    ]
}
```

This is a broad set of permissions which you should always be cautious of. But Serverless will need quite a few for this to go smoothly. You can try to look through the Serverless documentation to see if you can set a more granular set of permissions, otherwise we'll carry on.

Name the policy something and click **Save**. I named mine **`serverless`**.

You may need to recreate the IAM user to see the policy but you should find it if you reload the page.

![](https://miro.medium.com/1*FL7htpa3f7SFjlX_W89XRw.png)

Create the user and then click on it to find where you can generate an access key.

Choose **Local Code** when they ask you what you are intending to use it for and download the .csv file. We'll need these credentials later for Serverless to create our application.

It is worth noting that AWS recommends the use of an IDE which supports the AWS Toolkit enabling authentication through IAM Identity Center. Visual Studio Code should have an extension you can use. This allows you to authenticate without having to hardcode or expose your access keys, which can improve security. For this exercise I won't be doing this.

Don't log out of the AWS console yet, we'll add the .txt file we're using to S3 directly.

### Add Your Text File to S3

What we're doing here is trying to set up a bot that will answer questions from a file. This file should be well structured.

What I did for this article, was get some inspiration online and use an arbitrary company name called Socky. I wonder where you think I got more inspiration from? Hehe. You can steal this document as a [test file](https://github.com/ilsilfverskiold/langchainjs-aws-service/blob/main/customer_service_questions.txt).

If you've opened it, look at how it is structured.

Make sure you keep a similar structure or you will have to tweak the script a bit so it will get processed correctly.

```plaintext
q: how to contact customer support?
a: If you can't find the answers you're looking for in our Help Centre, you can always contact our Customer Support. You can reach out to us via the Socky Chatbot or by sending us a direct message on our official social media pages like Facebook, Instagram, or Twitter. Please note that we don't offer phone support. Although our main support language is English, we can also assist in German and Dutch. Remember to send only one query at a time and always include your order number.

q: why can't I place my order or checkout on your website?
a: There can be several reasons for this issue. Make sure to remove any special characters from checkout fields and refresh your browser. It might also help to try from a different device or clear cookies and cache. Ensure that you're using the correct regional website based on your shipping country. If none of these steps help, please contact our support.

q: can I cancel my order or make changes to it after it has been placed?
a: We process and pack orders quickly, so making changes can be challenging. If you need to make any adjustments, please reach out to us immediately.
```

A tip is to use ChatGPT, especially with Code Interpreter, to help set up the structure for you.

Now navigate to S3 within the AWS console and create a new bucket. Call it **my-langchain-bucket**. Remember the region but leave everything as is otherwise. I've set **eu-central-1** and this is hard coded into both my nodejs lambda functions. If you choose another region make sure you tweak the code before you deploy or it won't be able to access the bucket.

Go into your bucket and upload your .txt file.

Then create a folder called `embeddings`

Look at what my bucket looks like below. The embeddings folder should be empty, we'll store our embeddings store in there.

![](https://miro.medium.com/1*fLZzBdVMWlHZt9dSqfri3Q.png)

I did it this way, but for the future it will probably better to let the script handle it.

We're done here though.

### Set Up Your Local Environment

Set up a new folder somewhere on your computer. I'll call my folder aws-langchain.

```bash
mkdir aws-langchain
```

Go into the folder.

```bash
cd aws-langchain
```

Clone the repository.

```bash
git clone https://github.com/ilsilfverskiold/langchain-embeddings-serverless.git
```

You can look at the files directly in Github, it's just a few scripts and a YAML file that we'll be using to deploy to Serverless. You can also copy those files directly rather than cloning the repository but this will be faster.

If you're done cloning it. Go into the folder.

```bash
cd langchain-embeddings-serverless
```

Open up your code in a code editor. I'm using Visual Studio Code and have the `code .` command enabled so I can just do this.

```bash
code .
```

Tada! Here are the files.

![](https://miro.medium.com/1*E7FV6Ir-obSqmeC5CSvm8w.png)

Let's do this!

First check your nodejs version.

```typescript
node -v
```

It should say 18x. If not make sure you upgrade. I use nvm. You can do too but you'll need to configure it.

```bash
nvm use 18
```

Then install all the dependencies.

```typescript
npm install
```

If you haven't installed the Serverless framework globally you can do so as well.

```typescript
npm install -g serverless
```

Go get an OpenAI API key. If you don't know what I am talking about go to [platform.openai.com](https://platform.openai.com/account/api-keys) and navigate to API keys. You're correct thinking you'll need to have tokens or a debit card added. This is the only thing that will cost us for this application.

Then add it in like so in the terminal.

```bash
export OPENAI_API_KEY="apikeyhere"
```

The environment variable is set only for the current shell session so if you exit the window you'll need to set it again.

We can't forget to set our AWS credentials, otherwise Serverless won't be able to deploy.

Remember that CSV file I made you download. Use it for this command in your terminal.

```bash
serverless config credentials --provider aws --key YOUR_AWS_KEY --secret YOUR_AWS_SECRET
```

Now you can actually deploy from here but let's just go through a bit of the serverless.yml file.

### The Serverless Framework

Go to the serverless.yml file or look at the YAML file below. Here is the file that will let us deploy via Serverless.

```yaml
service: langchain-service

provider:
  name: aws
  runtime: nodejs18.x
  region: eu-central-1
  environment:
    OPENAI_API_KEY: ${env:OPENAI_API_KEY}
  iamRoleStatements:
    - Effect: Allow
      Action:
        - s3:GetObject
        - s3:PutObject
      Resource: "arn:aws:s3:::my-langchain-bucket/*"
  apiGateway:
    apiKeys:
      - langchainAPIKey
    usagePlan:
      quota:
        limit: 1000
        offset: 2
        period: MONTH
      throttle:
        burstLimit: 10
        rateLimit: 1

functions:
  processFile:
    handler: processEmbeddingsHandler.handler
    timeout: 15
    events:
      - http:
          path: process
          method: post
          private: true

  processQuestion:
    handler: processQuestionHandler.handler
    timeout: 15
    events:
      - http:
          path: question
          method: post
          cors: true
          private: true
```

This file will create two Lambda functions with a NodeJS 18 runtime and attach a POST endpoint to both with API Gateway. We're also setting a usage plan associated with our endpoints as well as an API key. We'll receive our endpoints and api key once the application has been deployed.

Remember what we named our bucket in S3? my-langchain-bucket. This script will allow the lambdas to access this bucket. If you named it something else, you need to tweak this file.

```yaml
Resource: "arn:aws:s3:::my-langchain-bucket/*"
```

### Deploy Your Application

Let's go ahead and deploy this application now.

```bash
serverless deploy
```

This can take a few minutes.

After it has completed you should have two HTTP Post routes in your terminal presented along with an API key.

![](https://miro.medium.com/1*A1okJlRnxDNKdm4fpOe1WA.png)

If you try this out and get issues with node-faiss you need to do a manual fix here.

I found a solution through [ewfian](https://github.com/hwchase17/langchainjs/issues/1930#issuecomment-1646500643). I've added it to the serverless.yml file but it didn't go through properly for me so I added this layer directly to the created lambdas in the AWS console.

To do this go to Lambda in AWS find Layers. Add the zip file found under /layer in the directory as a layer.

![](https://miro.medium.com/1*mCp5JxTzCRgxMu4a3hEdZw.png)

![I have already created the layer before here but you'll need to create a new one.](https://miro.medium.com/1*xJNuJBZHUrv_FEPR5pHUUQ.png)

Once you've created the layer with the zip file, locate the two functions we've created with Serverless.

![](https://miro.medium.com/1*tXPB3NxR-S5Iz45m8OVy6Q.png)

Open them up one by one and scroll down until you see Layers.

![](https://miro.medium.com/1*NF_dyism32rnBy8nPIzctQ.png)

Add the node-faiss layer you've created.

You can log out of the console now.

## The Results

### Test Your API Routes

Now we first have to create embeddings with our text file. We'll use the first route to do this.

![](https://miro.medium.com/1*jr7DQzo8lwGqEjCi7SK_uA.png)

This route is obviously letting us set any .txt file we have as long as it is in a new bucket. The idea is that we can reuse it.

However, you'll have to tweak the lambda roles to allow you to access that new bucket as well.

The next route allows us to ask a question with the file itself using an LLM, OpenAI's GPT-3.5 in this case.

![](https://miro.medium.com/1*V7PH3F_EUk0O1VaM1dhtEA.png)

If you set chatHistory as an empty string the function will grab context from the file, if you set it with history it will use the history and the document as context.

See the lambda function [processQuestion.mjs](https://github.com/ilsilfverskiold/langchainjs-aws-service/blob/main/processQuestion.mjs) to work with the chain I've set up.

### Add it to a React Application

Now, I like to have a pretty end result and I have a [react-chatbot](https://github.com/ilsilfverskiold/react-langchain-chatbot) repository I've used as a Langchain playground that is easy enough to clone and simply power up. To use an AWS route like the one we've set up look at option 4.

Add your credentials in an .env file by following the [README file](https://github.com/ilsilfverskiold/react-langchain-chatbot) instructions once you've cloned it.

The end result of a minute or two of work is this.

![](https://miro.medium.com/1*6EOeYKajH6Wu_uwJJ8UxCQ.gif)

I've since improved the code which allows you to send in a prompt and system template so your bot can have a bit of a personality. I think it is good to use for actual stuff now.

See the full Github documentation [here](https://github.com/ilsilfverskiold/langchainjs-aws-service) on how to structure your calls.

Fun stuff.

❤