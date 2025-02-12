# Your Own Personal ChatGPT

### How you can fine-tune OpenAI's GPT-3.5 Turbo model to perform new tasks using your custom data

![**"a minimalist painting of a cute robot in an art class,"** Midjourney Image, Edited by Author](https://miro.medium.com/1*j85YSivk7_4XXVeCWpU7BQ.jpeg)

I was excited when I got an email from OpenAI announcing the ability to fine-tune ChatGPT. The update came in response to the requests from developers and businesses looking to customize the model to better cater to their specific requirements. By leveraging this fine-tuning, it's now possible to improve steerability, achieve more consistent output formatting, and establish a desired custom tone. Another noteworthy aspect is that users can send shorter prompts without any notable dip in performance.

Here's what OpenAI says on their development blog [1].

> This update gives developers the ability to customize models that perform better for their use cases and run these custom models at scale. Early tests have shown a fine-tuned version of GPT-3.5 Turbo can match, or even outperform, base GPT-4-level capabilities on certain narrow tasks. As with all our APIs, data sent in and out of the fine-tuning API is owned by the customer and is not used by OpenAI, or any other organization, to train other models. - Andrew Peng, et al., OpenAI

In this article, I'll demonstrate how I used text from my Medium articles as training and test data to convert plain text into Markdown format automatically. Before I describe the experiment, I'll give you a little background on ChatGPT.

## Background

The AI model known as ChatGPT was introduced in November of 2022 [2]. It was the first public chatbot that OpenAI released, and I have written about it several times [here on Medium](https://robgon.medium.com/list/chatgpt-d71f5c6d0f10). The model works as well as a general chatbot but has a few limitations. For example, its training cutoff was September 2021, so it doesn't directly know about any new information since then. Using browser plug-ins to fetch and augment the model's data is possible, but this is currently slow and cumbersome.

A better way to infuse new information and skills into ChatGPT is using OpenAI's [fine-tuning API](https://platform.openai.com/docs/guides/fine-tuning). Fine-tuning ChatGPT through the API offers improved results compared to regular prompting. It enables training on more examples than a prompt can handle, leads to shorter prompts that save tokens, and results in quicker response times. There is a cost to using this, however.

### Pricing

Here are prices for using the various models from [OpenAI's pricing page](https://openai.com/pricing), in US dollars.

![**Prices for Using [OpenAI](https://openai.com/pricing)'s Models,** Source OpenAI](https://miro.medium.com/1*ue70uDwklx-my-4p3jrFeA.png)

The term "token" refers to the number of word parts used for prompts and results, where 750 words can be represented with about 1,000 tokens. The term "context" refers to the total number of tokens used for an interaction's inputs and results. The chart shows that the tokens for the input words (i.e., the prompt) cost less than those for the output words (i.e., the results.) Using GPT-4 costs more than using GPT-3.5 Turbo. A fine-tuned version of GPT-3.5 Turbo is somewhere between the two other models. However, it would be worth the cost if a fine-tuned GPT-3.5 Turbo can outperform GPT-4. We'll see if it's true in my experiment.

Note that these prices are subject to change over time and may be different across locales and currencies. Any changes will impact the limitations and trade-offs discussed above, so please check your local prices before making any decisions.

### Automatically Formatting Documents

To kick things off, I ran an experiment to see if the GPT models could be trained to render text files to Markdown format automatically. This lightweight markup language specifies text formats like headings, blockquotes, code blocks, etc.

![**Components for Formatting Text Files**, Diagram by Author](https://miro.medium.com/1*KCaLOhxonHtMsr-YD3lOaw.png)

To test the models, I downloaded 36 of my articles on Medium and converted the HTML files to Markdown and plain text format using the Beautiful Soup [library](https://code.launchpad.net/beautifulsoup/). The text files are used as inputs, and the Markdown files are used as outputs for the training and testing. I used 32 articles for training and three for testing. I tested three variants of OpenAI's language models, GPT-4, GPT-3.5 Turbo, and the fine-tuned GPT-3.5 Turbo, and collected the results.

Here's what one of the articles looks like in Markdown format, before and after rendering.

### Markdown Format

![](https://miro.medium.com/1*cuY4iPg9J2hDrNGDfPGGYQ.png)

![**Example Article Showing Markdown Format** (left), **and Rendered** (right), Images by Author](https://miro.medium.com/1*tKsCVTXwPwfKHJe_HbRg2g.png)

You can see the Markdown format characters on the left in dark red, like # for Heading 1 and ## for Heading 2. On the right, you can see how the file is rendered, showing formatting like the block quote at the end.

## Fine-tuning GPT-3.5 Turbo

Fine-tuning the model was easy. I followed the instructions from OpenAI [here](https://platform.openai.com/docs/guides/fine-tuning). The first step was to gather my training and validation data into JSON files. I divided my articles into four sections to keep within the 4K token limit for GPT-3.5 Turbo. Each entry in the file had a message with an optional system prompt, a user message, and an expected response from the assistant. Here is an example entry from my training file.

```json
{"messages": [
{"role": "system", "content": "You render plain text into markdown format."},
{"role": "user", "content": "MachineRay: Using AI to Create Abstract Art\nHow I trained a GAN using public domain paintings\nRobert A. Gonsalves\nTowards Data Science\nAug 3, 2020\nMachineRay - https://medium.com/towards-data-science/machineray-using-ai-to-create-abstract-art-39829438076a\nFor the past three months, I have been exploring the latest techniques in Artificial Intelligence (AI) and Machine Learning (ML) to create abstract art. During my investigation, I learned that three things are needed to create abstract paintings: (A) source images, (B) an ML model, and (C) a lot of time to train the model on a high-end GPU. Before I discuss my work, let\u2019s take a look at some prior research.\nThis is the first part of my series of articles on how AI can be used for creative endeavors. The second part is on how to use ML to generate plots for new stories, available  here .\nBackground\nArtificial Neural Networks\nWarren McCulloch and Walter Pitts created a computational model for Neural Networks (NNs) back in 1943[1]. Their work led to research of both the biological processing in brains and the use of NNs for AI. Richard Nagyfi discusses the differences between Artificial Neural Networks (ANNs) and biological brains in this  post . He describes an apt analogy that I will summarize here:  ANNs are to brains as planes are to birds . Although the development of these technologies was inspired by biology, the actual implementations are very different!\nBoth ANNs and biological brains learn from external stimuli to understand things and predict outcomes. One of the key differences is that ANNs work with floating-point numbers and not just binary firing of neurons.  With ANNs it\u2019s numbers in and numbers out.\nThe diagram below shows the structure of a typical ANN. The inputs on the left are the numerical values that contain the incoming stimuli. The input layer is connected to one or more hidden layers that contain the memory of prior learning. The output layer, in this case just one number, is connected to each of the nodes in the hidden layer.\nEach of the internal arrows represents numerical weights that are used as multipliers to modify the numbers in the layers as they get processed in the network from left to right. The system is trained with a dataset of input values and expected output values. The weights are initially set to random values. For the training process, the system runs through the training set multiple times, adjusting the weights to achieve the expected outputs. Eventually, the system will not only predict the outputs correctly from the training set, but it will also be able to predict outputs for unseen input values. This is the essence of Machine Learning (ML).  The intelligence is in the weights . A more detailed discussion of the training process for ANNs can be found in Conor McDonald\u2019s post,  here .\nGenerative Adversarial Networks\nIn 2014, Ian Goodfellow and seven coauthors at the Universit\u00e9 de Montr\u00e9al presented a  paper  on Generative Adversarial Networks (GANs)[2].  They came up with a way to train two ANNs that effectively compete with each other to create content like photos, songs, prose, and yes, paintings.  The first ANN is called the Generator and the second is called the Discriminator. The Generator is trying to create realistic output, in this case, a color painting. The Discriminator is trying to discern real paintings from the training set as opposed to fake paintings from the generator. Here\u2019s what a GAN architecture looks like.\nA series of random noise is fed into the Generator, which then uses its trained weights to generate the resultant output, in this case, a color image. The Discriminator is trained by alternating between processing real paintings, with an expected output of 1 and fake paintings, with an expected output of -1. After each painting is sent to the Discriminator, it sends back detailed feedback about why the painting is not real, and the Generator adjusts its weights with this new knowledge to try and do better the next time.  The two networks in the GAN are effectively trained together in an adversarial fashion . The Generator gets better at trying to pass off a fake image as real, and the Discriminator gets better at determining which input is real, and which is fake. Eventually, the Generator gets pretty good at generating realistic-looking images. You can read more about GANs, and the math they use, in Shweta Goyal\u2019s post  here .\nImproved GANs for Large Images\nAlthough the basic GAN described above works well with small images (i.e. 64x64 pixels), there are issues with larger images (i.e. 1024x1024 pixels). The basic GAN architecture has difficulty converging on good results for large images due to the unstructured nature of the pixels. It can\u2019t see the forest from the trees. Researchers at NVIDIA developed a series of improved methods that allow for the training of GANs with larger images. The first is called \u201c Progressive Growing of GANs \u201d [3].\n"},
{"role": "assistant", "content": "# MachineRay: Using AI to Create Abstract Art\n## How I trained a GAN using public domain paintings\nRobert A. Gonsalves\n</br>Aug 3, 2020</br>\nMachineRay - https://medium.com/towards-data-science/machineray-using-ai-to-create-abstract-art-39829438076a</br></br>\nFor the past three months, I have been exploring the latest techniques in Artificial Intelligence (AI) and Machine Learning (ML) to create abstract art. During my investigation, I learned that three things are needed to create abstract paintings: (A) source images, (B) an ML model, and (C) a lot of time to train the model on a high-end GPU. Before I discuss my work, let\u2019s take a look at some prior research.\nThis is the first part of my series of articles on how AI can be used for creative endeavors. The second part is on how to use ML to generate plots for new stories, available  here .\n# Background\n## Artificial Neural Networks\nWarren McCulloch and Walter Pitts created a computational model for Neural Networks (NNs) back in 1943[1]. Their work led to research of both the biological processing in brains and the use of NNs for AI. Richard Nagyfi discusses the differences between Artificial Neural Networks (ANNs) and biological brains in this  post . He describes an apt analogy that I will summarize here:  ANNs are to brains as planes are to birds . Although the development of these technologies was inspired by biology, the actual implementations are very different!\nBoth ANNs and biological brains learn from external stimuli to understand things and predict outcomes. One of the key differences is that ANNs work with floating-point numbers and not just binary firing of neurons.  With ANNs it\u2019s numbers in and numbers out.\nThe diagram below shows the structure of a typical ANN. The inputs on the left are the numerical values that contain the incoming stimuli. The input layer is connected to one or more hidden layers that contain the memory of prior learning. The output layer, in this case just one number, is connected to each of the nodes in the hidden layer.\nEach of the internal arrows represents numerical weights that are used as multipliers to modify the numbers in the layers as they get processed in the network from left to right. The system is trained with a dataset of input values and expected output values. The weights are initially set to random values. For the training process, the system runs through the training set multiple times, adjusting the weights to achieve the expected outputs. Eventually, the system will not only predict the outputs correctly from the training set, but it will also be able to predict outputs for unseen input values. This is the essence of Machine Learning (ML).  The intelligence is in the weights . A more detailed discussion of the training process for ANNs can be found in Conor McDonald\u2019s post,  here .\n## Generative Adversarial Networks\nIn 2014, Ian Goodfellow and seven coauthors at the Universit\u00e9 de Montr\u00e9al presented a  paper  on Generative Adversarial Networks (GANs)[2].  They came up with a way to train two ANNs that effectively compete with each other to create content like photos, songs, prose, and yes, paintings.  The first ANN is called the Generator and the second is called the Discriminator. The Generator is trying to create realistic output, in this case, a color painting. The Discriminator is trying to discern real paintings from the training set as opposed to fake paintings from the generator. Here\u2019s what a GAN architecture looks like.\nA series of random noise is fed into the Generator, which then uses its trained weights to generate the resultant output, in this case, a color image. The Discriminator is trained by alternating between processing real paintings, with an expected output of 1 and fake paintings, with an expected output of -1. After each painting is sent to the Discriminator, it sends back detailed feedback about why the painting is not real, and the Generator adjusts its weights with this new knowledge to try and do better the next time.  The two networks in the GAN are effectively trained together in an adversarial fashion . The Generator gets better at trying to pass off a fake image as real, and the Discriminator gets better at determining which input is real, and which is fake. Eventually, the Generator gets pretty good at generating realistic-looking images. You can read more about GANs, and the math they use, in Shweta Goyal\u2019s post  here .\n## Improved GANs for Large Images\nAlthough the basic GAN described above works well with small images (i.e. 64x64 pixels), there are issues with larger images (i.e. 1024x1024 pixels). The basic GAN architecture has difficulty converging on good results for large images due to the unstructured nature of the pixels. It can\u2019t see the forest from the trees. Researchers at NVIDIA developed a series of improved methods that allow for the training of GANs with larger images. The first is called \u201c Progressive Growing of GANs \u201d [3].\n"}
]}
```

The system prompt is the same for every entry. The user message is plain text from one of my articles, and the assistant's response is the same text in Markdown format.

As mentioned, OpenAI does not use any data submitted to and generated by their API to train models or improve their service. However, this is different for the interactive version of their service. More info is available [here](https://help.openai.com/en/articles/5722486-how-your-data-is-used-to-improve-model-performance).

### Checking the Data

Next, I ran the check_file() function from OpenAI on the training and testing files. Here are the results from the check.

![**Output from check_file**, Image by Author](https://miro.medium.com/1*WDdOp8NLZR-1e-3RQthhwQ.png)

You can see that I used 132 examples for training, which is 32 articles with 4 sections each. The script showed me if there were any errors and calculated the total number of tokens, which I used to estimate the cost of training.

### Run the Training

Once I had my training and testing files set up and checked, I uploaded the files to my account.

```python
openai.File.create(file=open("training.jsonl", "rb"),purpose='fine-tune')
openai.File.create(file=open("testing.jsonl", "rb"),purpose='fine-tune')
```

I then got the names of the files and ran this command to start the training.

```ini
results = openai.FineTuningJob.create(
  training_file="file-PsY5FuC4m4JzIOKKtB7cWDbz",
  validation_file="file-xq5M0Yy1CFIkKcHCOgSFOp40",
  suffix = "robgon_03",
  model="gpt-3.5-turbo")
```

The suffix is a way to provide some metadata so I can identify the fine-tuned model later. By default, the system ran for three epochs, which took about a half hour to finish. When the training was done, OpenAI sent me an email notification. Here's what the training and validation loss looked like throughout 384 training steps (32 articles * 4 sections * 3 epochs.)

![**Train and Validation Loss for Fine-tuning GPT-3.5 Turbo**, Chart by Author](https://miro.medium.com/1*U799Whzi23bd3MJnYbTGaA.png)

It did a good job reducing the loss throughout the training. The validation also seemed to follow a downward trend, so it didn't overfit the training data. Let's see the new model in action.

### Testing the Fine-tuned Model

I logged into my OpenAI account and navigated to the Playground to test the system. I chose my fine-tuned model, entered the system prompt, added a few lines from the head of this article, and hit the submit button.

![**OpenAI Playground Test of the Fine-tuned Model**, Screenshot by Author](https://miro.medium.com/1*U6gYD_nf0S-o32zGHjqUoQ.png)

It worked well! With a minimal prompt, it matched the formatting that I expected from the training.

### Comparative Evaluation

To compare the accuracy of the fine-tuned model with the standard models, I created a [longish prompt](https://gist.github.com/robgon-art/95ec5a7932d8276db9162737f363dcfc#file-prompt-py) to test GPT-4 and GPT-3.5 Turbo with one-shot inferencing. I gave one example of before and after text in the system prompt and the text to be formatted in the user prompt for these two models.

I then ran the 12 examples (3 articles with 4 sections) through the three models to get the results. To get a quantitative measure of accuracy, I calculated the Character Error Rate (CER) between the predicted formatted text and the original formatted text as a reference. In addition to the CER results, I report the duration of running the inferences and the costs for comparison.

![**Results of Formatting Text Documents with Various GPT Models**, Table by Author](https://miro.medium.com/1*UiaYVvGCi8-nsQLiBuI2CA.png)

You can see that the GPT-4 model had fewer errors than the stock GPT-3.5 Turbo model. However, formatting the three articles took a lot longer and had a much higher cost. The fine-tuned GPT-3.5 Turbo model had far fewer errors and ran much faster. However, the inferencing cost was in the middle and was burdened with the fine-tuning cost. I would have to run seven experiments (e.g., 21 formatted articles) to break even with the GPT-4 costs. But it may be worth it because of the low CER.

## Discussion

Although the fine-tuned model did a great job of automatically formatting text, I was curious to see if it "learned" anything about the source material it saw while fine-tuning. So I asked it this question.

> USER - Do you know about an article in Towards Data Science called Writing Songs with GPT-4: Part 1, Lyrics?

> ASSISTANT - Yes, I am familiar with the article you mentioned. It is titled "Writing Songs with GPT-4: Part 1, Lyrics" and it was published on Towards Data Science. This article explores the use of GPT-4, a language model developed by OpenAI, for generating song lyrics. It covers how to fine-tune the model and generate creative lyrics.

> USER - Who wrote the article?

> ASSISTANT - The article "Writing Songs with GPT-4: Part 1, Lyrics" was written by Ellie Chen.

OK, it is hallucinating here. First, I didn't fine-tune GPT-4 for this article. Second, the model thinks it was written by someone named Ellie Chen. So, the training data didn't sink in. As a next step, I could explore ways to inject new data into the GPT models during fine-tuning for later recall and use.

## Conclusion

OpenAI's introduction of fine-tuning capabilities for ChatGPT presents a new avenue for developers to adapt the model to specific tasks. This article explored the process and outcomes of using this feature, particularly in automatically converting plain text into Markdown format. Although the model worked well with this straightforward task, it didn't seem to comprehend and remember the new text it saw, so it can't reliably be used for generative queries about the new material. As AI technology progresses, the exploration of creating specialized models becomes a topic of interest, highlighting the importance of continued research and assessment.

## Source Code

The source code for this project is available on GitHub.

![**Creative Commons Attribution Sharealike**](https://miro.medium.com/0*NcYIOnH-GuX4Dz2-.png)

## Acknowledgments

I thank Jennifer Lim for reviewing the article and providing feedback.

## References

[1] A. Peng et al., [GPT-3.5 Turbo fine-tuning and API updates](https://openai.com/blog/gpt-3-5-turbo-fine-tuning-and-api-updates), 2023

[2] OpenAI, [Introducing ChatGPT](https://openai.com/blog/chatgpt), 2022