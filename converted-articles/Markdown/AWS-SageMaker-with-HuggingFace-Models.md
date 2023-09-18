# AWS SageMaker real-time endpoints with HuggingFace Embedding Models: A Guide for LLM Application

and how to ingest embedding data into vector database with real-time inference embedding endpoint with Langchain

I'm not entirely sure if I should officially label this as "part 2" of the series building a Chatbot app that's totally unique compared to 99% of the tutorials out there on the internet and is production-ready. 
The content we'll cover here sets the groundwork for the rest of our Chatbot journey. Even though I haven't explicitly labelled it as "part 2," I assure you it's a mandatory step. So, let's roll up our sleeves and get ready for some exciting progress!

> [**Zero to One: A Guide to Building a First PDF Chatbot with LangChain & LlamaIndex - Part 1**](https://medium.com/how-ai-built-this/zero-to-one-a-guide-to-building-a-first-pdf-chatbot-with-langchain-llamaindex-part-1-7d0e9c0d62f)

The concept of embedding plays a crucial role in any LLM Application. Whether you're creating a toy app, a PoC, or showcasing your product, this article can be beneficial. However, if your goal is to build a robust LLM app that can handle a large user base and scale effectively, this is precisely the article you need to focus on.

If you need a refreshment of choosing the embedding model, please refer to my previous article here:

> [**Choosing the Right Embedding Model: A Guide for LLM Applications**](https://medium.com/@ryanntk/choosing-the-right-embedding-model-a-guide-for-llm-applications-7a60180d28e3)

## Frustrations with Superficial Articles on the Search for an In-Depth Guide

Before I started, I want to highlight the frustration and the reason I want to write this article.

I've often come across articles with enticing titles like "Develop Your Own AI to Read Your Documents" or "Develop AI Applications with Langchain/Llamaindex." However, I've noticed that most of these articles tend to be quite basic, lacking in-depth discussions on the topics they promise to cover. It appears that many of these articles simply regurgitate information from official documents without providing any meaningful analysis or technical insights. This trend is also evident when searching for articles on Hugging Face's embedding.

![Photo by [Tim Gouw](https://unsplash.com/@punttim?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)](https://miro.medium.com/0*2bfq-L3nNBipaJS1)

Please understand that I do appreciate the value of these articles for individuals who are new to the subject matter and seek introductory information. However, as someone who delves into intricate technical details and craves in-depth exploration, I find these articles lacking in substance. It seems that the popularity of certain topics has led to an overwhelming number of articles covering the same information, resulting in a monotonous and uninteresting feed on platforms like Medium.

Personally, I have been searching for a comprehensive guide on deploying any Hugging Face's embedding model to AWS Sagemaker, but to no avail. Instead, I repeatedly encounter similar articles authored by different individuals, all offering a summary of the basic steps to deploy a Hugging Face embedding model to Sagemaker. The frustrating part is that they all seem to use the same Hugging Face model and even the same instance type.

What I truly desire is an in-depth, step-by-step tutorial that starts from scratch and guides me through the entire deployment process. Regrettably, I haven't been able to find such a guide. Consequently, I have taken it upon myself to write an article that may prove interesting and helpful to anyone else who has experienced this same struggle.

## On a Side Note

I've started the AI weekly newsletter to keep you updated on all the exciting happenings in the world of AI. It's a curated summary of engineer resources, AI startup spotlights, and the latest research from brilliant minds in the field, delivered straight to your inbox every week.

The best part? It's designed to be a quick read, taking you only about 5 minutes to get up to speed with the most important developments in AI. So, if you're eager to stay informed without spending too much time, this newsletter is perfect for you.

Check out my Substack and subscribe to stay in the loop with the latest AI trends and advancements

> [**AI Weekly | How AI Built This | Ryan Nguyen | Substack**](https://howaibuildthis.substack.com/s/ai-weekly)

## AWS Sagemaker

Over the past 8 years, I've had the pleasure of working extensively with a variety of AWS products. When it comes to developing prototypes or MVPs, AWS has always been my top choice. I've built and designed multiple data platforms for companies of all sizes, from startups to large corporations. AWS is like the OG in the game, but I'll be honest, sometimes I wish their services were a bit easier to use. In fact, there have been entire startups created solely to make AWS more user-friendly.

Having said that, AWS's machine learning platform isn't my personal favourite. It can be a bit tricky to navigate and often leads to confusion. On the other hand, in the world of data engineering, both in my current company and previous ones, we tend to lean towards Databricks for most of our development work. Databricks is a breeze to use and a great tool for working with data and developing data lakehouse platforms. It also provides an awesome platform for data scientists and machine learning enthusiasts to build and deploy models. The only downside is that Databricks can be a bit pricey, but hey, you get what you pay for, and Databricks delivers.

![](https://miro.medium.com/0*RG_jvnXtUJYHwfWf.jpg)

From a technical standpoint, AWS offers all the features and capabilities that Databricks provides, and in some aspects, even surpasses it. AWS stands out because its services are well-integrated and easily usable together. Building machine learning platforms or data platforms on top of AWS is a breeze, as you don't have to worry about the underlying infrastructure.

> Speaking honestly, AWS Sagemaker is an "okay" product.

While it can be difficult to use, and the notebook environment sucks, the overall functionality is acceptable. If your company heavily invests in AWS and utilizes it across all products, then it's highly likely you'll default to using AWS Sagemaker. As for us, and specifically for me, Sagemaker serves as the final step in the deployment process, thanks to its convenient endpoint deployment and seamless integration with API Gateway, Lambda, and ECS.

Recently, what piqued my interest and drew me back to working with Sagemaker is its partnership with Hugging Face. If you're reading this blog, you're probably a skilled engineer in the data space, be it a data engineer or a machine learning engineer, and I'm confident you're familiar with Hugging Face. Personally, I'm a huge fan of Hugging Face, and their collaboration with AWS has significantly bolstered the capabilities of Sagemaker.

![](https://miro.medium.com/0*UmZBTYI2Sc_qqQsY.png)

In this article, I won't dive into the nitty-gritty details of Sagemaker or how to use it. I also won't talk about building embedding models or using Hugging Face. If you're interested, I can cover those topics in future articles. For now, let's focus on the main topic: how to deploy any Hugging Face embedding model to AWS Sagemaker. It's going to be a fun and informative journey, so stick around!

## Prerequisite

1. You need to have an AWS account. If you don't already have one, then please create a new one.

2. I want to state again that this guide is not a tutorial on Sagemaker, it is also not a tutorial on how to use Hugging Face, nor one about how to build custom embedding.

3. In fact, you don't even need to know Sagemaker, just think about it as a platform where we are going to deploy our embedding model.

## Hugging Face on Amazon Sagemaker

If you are interested in how to deploy the Hugging Face model to Amazon Sagemaker, you can go through the official hugging face document here

> [**Hugging Face on Amazon SageMaker**](https://huggingface.co/docs/sagemaker/index)

In this post, we will cover one thing which is "**How to deploy any Hugging Face embedding model on AWS Sagemaker**".

Traditionally, the [SageMaker Hugging Face Inference Toolkit](https://github.com/aws/sagemaker-huggingface-inference-toolkit) supports the [pipeline feature](https://huggingface.co/transformers/main_classes/[pipelines](https://huggingface.co/transformers/main_classes/pipelines.html).html) from Transformers for zero-code deployment. This means you can run compatible Hugging Face Transformer models without providing pre- & post-processing code. Therefore we only need to provide an environment variable `HF_TASK` and `HF_MODEL_ID` when creating our endpoint and the Inference Toolkit will take care of it. This is a great feature if you are working with existing pipelines. To deploy any hugging face model on AWS Sagemaker, it is super simple this following code

```python
from sagemaker.huggingface.model import HuggingFaceModel

# Hub model configuration <https://huggingface.co/models>
hub = {
  'HF_MODEL_ID':'distilbert-base-uncased-distilled-squad', # model_id from hf.co/models
  'HF_TASK':'question-answering'                           # NLP task you want to use for predictions
}

# create Hugging Face Model Class
huggingface_model = HuggingFaceModel(
   env=hub,                                                # configuration for loading model from Hub
   role=role,                                              # IAM role with permissions to create an endpoint
   transformers_version="4.26",                             # Transformers version used
   pytorch_version="1.13",                                  # PyTorch version used
   py_version='py39',                                      # Python version used
)

# deploy model to SageMaker Inference
predictor = huggingface_model.deploy(
   initial_instance_count=1,
   instance_type="ml.m5.xlarge"
)

# example request: you always need to define "inputs"
data = {
"inputs": {
 "question": "What is used for inference?",
 "context": "My Name is Philipp and I live in Nuremberg. This model is used with sagemaker for inference."
 }
}

# request
predictor.predict(data)
```

- `HF_MODEL_ID` defines the model ID which is automatically loaded from [huggingface.co/models](http://huggingface.co/models) when you create a SageMaker endpoint. Access 10,000+ models on the 🤗 Hub through this environment variable.

- `HF_TASK` defines the task for the 🤗 Transformers p`ipeline.` A complete list of tasks can be found h[ere.](https://huggingface.co/docs/transformers/main_classes/pipelines)

But HF_TASK does not have an embedding task yet. Therefore, we will need to custom code to serve the embedding endpoint.

### What about embedding tasks?

By default, AWS Sagemaker officially supports these embedding models. It means if you want to use this model, you don't have to do anything extra. Just need to follow this notebook to deploy the supported embedding model.

But what if the official model is not the one you want? In fact, if you look at the hugging face's leaderboard, these embedding models are nowhere in the top rank of the data retrieval benchmark.

Luckily, the Hugging Face Inference Toolkit allows the user to override the default methods of the `HuggingFaceHandlerService`. You will need to create a folder named `code/` with an `inference.py` file in it. See [here](https://huggingface.co/docs/sagemaker/inference#create-a-model-artifact-for-deployment) for more details on how to archive your model artifacts. For example:

![](https://miro.medium.com/1*Wf8_7cdcL8WLEBQRsdhvGg.png)

The `inference.py` file contains your custom inference module and the `requirements.txt` file contains additional dependencies that should be added. The custom module can override the following methods:

- **`model_fn(model_dir)`** overrides the default method for loading a model. The return value `model` will be used in `predict` for predictions. `predict` receives argument the `model_dir`, the path to your unzipped `model.tar.gz`.

- **`transform_fn(model, data, content_type, accept_type)`** overrides the default transform function with your custom implementation. You will need to implement your own `preprocess`, `predict` and `postprocess` steps in the `transform_fn`. This method can't be combined with `input_fn`, `predict_fn` or `output_fn` mentioned below.

- **`input_fn(input_data, content_type)`** overrides the default method for preprocessing. The return value `data` will be used in `predict` for predictions. The inputs are:
`- input_data` is the raw body of your request.
`- content_type` is the content type from the request header.

- **`predict_fn(processed_data, model)`** overrides the default method for predictions. The return value `predictions` will be used in `postprocess`. The input is `processed_data`, the result from `preprocess`.

- **`output_fn(prediction, accept)`** overrides the default method for postprocessing. The return value `result` will be the response to your request (e.g.`JSON`). The inputs are:
`- predictions` is the result from `predict`.
`- accept` is the return accept type from the HTTP Request, e.g. `application/json`

## Development Environment

First, you need to create a custom IAM Role. You will need to use this role for most of your work so go ahead and create one.

```yaml
Role:
  Type: AWS::IAM::Role
  Properties:
    RoleName: SageMakerIAMRole
    Policies:
      - PolicyName: CustomerSagemakerPolicyAccess
        PolicyDocument:
          Version: 2012-10-17
          Statement:
            - Sid: ReadFromOpenSearch
              Effect: Allow
              Action:
                - "es:ESHttp*"
              Resource:
                - !Sub arn:aws:es:${AWS::Region}:${AWS::AccountId}:domain/*
            - Sid: ReadSecretFromSecretsManager
              Effect: Allow
              Action:
                - "secretsmanager:GetSecretValue"
              Resource: !Sub "arn:aws:secretsmanager:${AWS::Region}:${AWS::AccountId}:secret:*"
            - Sid: ReadWriteFromECR
              Effect: Allow
              Action:
                - "ecr:BatchGetImage"
                - "ecr:BatchCheckLayerAvailability"
                - "ecr:CompleteLayerUpload"
                - "ecr:DescribeImages"
                - "ecr:DescribeRepositories"
                - "ecr:GetDownloadUrlForLayer"
                - "ecr:InitiateLayerUpload"
                - "ecr:ListImages"
                - "ecr:PutImage"
                - "ecr:UploadLayerPart"
                - "ecr:CreateRepository"
                - "ecr:GetAuthorizationToken"
                - "ec2:DescribeAvailabilityZones"
              Resource: "*"
    ManagedPolicyArns:
      - arn:aws:iam::aws:policy/AmazonSageMakerFullAccess
      - arn:aws:iam::aws:policy/AWSCloudFormationReadOnlyAccess
      - arn:aws:iam::aws:policy/TranslateReadOnly
    AssumeRolePolicyDocument:
      Version: 2012-10-17
      Statement:
        - Effect: Allow
          Principal:
            Service:
            - sagemaker.amazonaws.com
          Action:
            - 'sts:AssumeRole'
```

If you want to use your local notebook, you need to note down the role name you've created above to manually set it in the code later.

**The next step is optional:** You can accomplish this guide with your local notebook environment and memory vector database with Chroma. However, if you want to have something serious, you can create a notebook instance along with the OpenSeach service.
With AWS Free Tier, you can have _**250 hours free of ml.t3.medium instance and up to 750 hours per month of a t2.small.search or t3.small.search instance**_

Here is yaml file, it basically created a notebook instance and OpenSearch instance.

```yaml
AWSTemplateFormatVersion: 2010-09-09
Description: Template to provision OpenSearch cluster and SageMaker Notebook for semantic search

Metadata:
  AWS::CloudFormation::Interface:
    ParameterGroups:
      - Label:
          default: Required Parameters
        Parameters:
          - OpenSearchUsername
          - OpenSearchPassword
          - OpenSearchIndexName
          - SageMakerNotebookName
    ParameterLabels:      
      OpenSearchUsername:
        default: OpenSearch cluster username
      OpenSearchPassword:
        default: OpenSearch cluster password
      OpenSearchIndexName:
        default: OpenSearch index name
      SageMakerNotebookName:
        default: Name of SageMaker Notebook Instance

Parameters:
  OpenSearchUsername:
    AllowedPattern: '^[a-zA-Z0-9]+$'
    Default: opensearchuser
    Description: User name for the account that will be added to the OpenSearch cluster.
    MaxLength: '25'
    MinLength: '5'
    Type: String
  OpenSearchPassword:
    AllowedPattern: '(?=^.{8,32}$)((?=.*\d)(?=.*[A-Z])(?=.*[a-z])|(?=.*\d)(?=.*[^A-Za-z0-9])(?=.*[a-z])|(?=.*[^A-Za-z0-9])(?=.*[A-Z])(?=.*[a-z])|(?=.*\d)(?=.*[A-Z])(?=.*[^A-Za-z0-9]))^.*'
    Description: Password for the account named above. Must be at least 8 characters containing letters, numbers and symbols
    MaxLength: '32'
    MinLength: '8'
    NoEcho: 'true'
    Type: String
  OpenSearchIndexName:
    Default: llm-app-index
    Type: String    
    Description: Name of the OpenSearch index for storing embeddings.
  SageMakerNotebookName:
    Default: llm-app-notebook
    Type: String
    Description: Enter name of SageMaker Notebook instance. The notebook name must _not_ already exist in your AWS account/region.
    MinLength: 1
    MaxLength: 63
    AllowedPattern: ^[a-z0-9](-*[a-z0-9])*
    ConstraintDescription: Must be lowercase or numbers with a length of 1-63 characters.
  SageMakerIAMRole:
    Description: Name of IAM role that will be created by this cloud formation template. The role name must _not_ already exist in your AWS account.
    Type: String
    Default: "SageMakerIAMRole"   

Resources:
# you can out the Role above here for the simple deployment
# Role:
# ....
  OpenSearchSecret:
    Type: AWS::SecretsManager::Secret
    Properties:
      Name: !Sub OpenSearchSecret-${AWS::StackName}
      Description: OpenSearch username and password
      SecretString: !Sub '{ "username" : "${OpenSearchUsername}", "password" : "${OpenSearchPassword}" }'

  NotebookInstance:
    Type: AWS::SageMaker::NotebookInstance
    Properties:
      NotebookInstanceName: !Ref SageMakerNotebookName
      InstanceType: ml.t3.medium
      RoleArn: !GetAtt Role.Arn

  OpenSearchServiceDomain:
  Type: AWS::OpenSearchService::Domain
  Properties:
    AccessPolicies:
      Version: 2012-10-17
      Statement:
        - Effect: Allow
          Principal:
            AWS: '*'
          Action: 'es:*'
          Resource: !Sub arn:aws:es:${AWS::Region}:${AWS::AccountId}:domain/*/*
    EngineVersion: 'OpenSearch_2.5'
    ClusterConfig:
      InstanceType: "t3.small.search"
    EBSOptions:
      EBSEnabled: True
      VolumeSize: 20
      VolumeType: 'gp3'
    AdvancedSecurityOptions:
      AnonymousAuthEnabled: False
      Enabled: True
      InternalUserDatabaseEnabled: True
      MasterUserOptions:
        MasterUserName: !Sub ${OpenSearchUsername}
        MasterUserPassword: !Sub ${OpenSearchPassword} 
    NodeToNodeEncryptionOptions:
      Enabled: True
    EncryptionAtRestOptions:
      Enabled: True
      KmsKeyId: alias/aws/es
    DomainEndpointOptions:
      EnforceHTTPS: True

Outputs:
  OpenSearchDomainEndpoint:
    Description: OpenSearch domain endpoint
    Value:
      'Fn::GetAtt':
        - OpenSearchServiceDomain
        - DomainEndpoint

  OpenSourceDomainArn:
    Description: OpenSearch domain ARN
    Value:
      'Fn::GetAtt':
        - OpenSearchServiceDomain
        - Arn

  OpenSearchDomainName:
    Description: OpenSearch domain name
    Value: !Ref OpenSearchServiceDomain

  Region:
    Description: Deployed Region
    Value: !Ref AWS::Region

  SageMakerNotebookURL:
    Description: SageMaker Notebook Instance
    Value: !Join
      - ''
      - - !Sub 'https://console.aws.amazon.com/sagemaker/home?region=${AWS::Region}#/notebook-instances/openNotebook/'
        - !GetAtt NotebookInstance.NotebookInstanceName
        - '?view=classic'

  OpenSearchSecret:
    Description: Name of the OpenSearch secret in Secrets Manager
    Value: !Ref OpenSearchSecret
```

For the rest of the article, I will use the notebook instance. You can do the same with your local notebook. Let's start with the first task.

**Install the necessary libraries**

```shell
!pip install langchain
!pip install opensearch-py
!pip install chromadb
```

**Install `git` and `git-lfs`**

```ruby
# For notebook instances (Amazon Linux)
!sudo yum update -y
!curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.rpm.sh | sudo bash
!sudo yum install git-lfs git -y
```

An alternative way to install **git-lfs** if the previous one does not work for you

```shell
!sudo yum install -y amazon-linux-extras
!sudo amazon-linux-extras install epel -y
!sudo yum-config-manager --enable epel
!sudo yum install git-lfs -y
```

**Set Permission.**

```python
import sagemaker
import boto3

sess = sagemaker.Session()
sagemaker_session_bucket = sess.default_bucket()

try:
    aws_role = sagemaker.get_execution_role()
except ValueError:
    iam = boto3.client('iam')
    aws_role= iam.get_role(RoleName='SageMakerIAMRole')['Role']['Arn']

aws_region = sess.region_name

print(f"sagemaker role arn: {role}")
print(f"sagemaker bucket: {sagemaker_session_bucket}")
print(f"sagemaker region: {aws_region}")
```

Now, it is all set. Next step, I will delve into how to deploy two different embedding models

## Example 1: [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)

This is the most basic embedding model that is offered by HuggingFace. Let's have a look at the documentation.

```python
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


# Sentences we want sentence embeddings for
sentences = ['This is an example sentence', 'Each sentence is converted']

# Load model from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

# Tokenize sentences
encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

# Compute token embeddings
with torch.no_grad():
    model_output = model(**encoded_input)

# Perform pooling
sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

# Normalize embeddings
sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

print("Sentence embeddings:")
print(sentence_embeddings)
```

What does this code do anyway? Well, this is a low-level embedding function that is done by all-MiniLM-L6-v2. If you are using Langchain, you may be familiar with

```java

from langchain.embeddings import HuggingFaceEmbeddings

model_id = "sentence-transformers/all-MiniLM-L6-v2"
embed_model = HuggingFaceEmbeddings(model_name=model_id)
```

Basically, _**HuggingFaceEmbedding**_ wraps all the necessary stuff away for you. To deploy these embedding inference models on Sagemaker, we need to unwrap those codes and put them into an **inference.py**

### Create a custom an **inference.py** script

```python
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] 
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def model_fn(model_dir):
  tokenizer = AutoTokenizer.from_pretrained(model_dir)
  model = AutoModel.from_pretrained(model_dir)
  return model, tokenizer

def predict_fn(data, model_and_tokenizer):
    # destruct model and tokenizer
    model, tokenizer = model_and_tokenizer

    # Tokenize sentences
    encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)

    # Perform pooling
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

    # Normalize embeddings
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

    # return dictonary, which will be json serializable
    return {"vector_embedding": sentence_embeddings.tolist()}
```

Please refer to **What about embedding tasks** section above for more information if you have forgotten what you have read already 😁

### **Create** model.tar.gz with inference script and model

```ini
repository = "sentence-transformers/all-MiniLM-L6-v2"
model_id=repository.split("/")[-1]
s3_location=f"s3://{sess.default_bucket()}/custom_inference/{model_id}/model.tar.gz"
```

```bash
## download the model from hf model
!git lfs install
!git clone https://huggingface.co/$repository

## copy inference.py and requirements.txt to code/ directory
## this is mandatory to have files under code/ folder
!cp -r inference.py $model_id/code/
!cp -r requirements.txt $model_id/code/

## create model.tar.gx archive with all the model artifacts
%cd $model_id
!tar zcvf model.tar.gz *

## upload to S3
!aws s3 cp model.tar.gz $s3_location
```

**Create a custom Huggingface Model**

```python
from sagemaker.huggingface.model import HuggingFaceModel

# create Hugging Face Model Class
huggingface_model = HuggingFaceModel(
   model_data=s3_location,       # path to your model and script
   role=role,                    # iam role with permissions to create an Endpoint
   transformers_version="4.26",  # transformers version used
   pytorch_version="1.13",        # pytorch version used
   py_version='py39',            # python version used
)

# deploy the endpoint endpoint
predictor = huggingface_model.deploy(
    initial_instance_count=1,
    instance_type="ml.m5.xlarge"
)
```

After this, you should have the embedding of real-time inference in the Sagemaker console.

---

Now, you may wonder, sentence-transformers/all-MiniLM-L6-v2 is already supported by AWS Sagemaer, why do we still need to go through this?

Well, example 1 is just a demonstration, I want to show example 1 because I want to show you the pattern on how to take any Huggingface model to Sagemaker not just only the official models.

As you will see, the beloved and almost champion in the embedding-retrieval **[hkunlp/instructor-large](https://huggingface.co/hkunlp/instructor-large)** will have a very different approach, but once you understand the logic behind it, you can onboard any models.
Let's try another one.

## Example 2: [hkunlp/instructor-large](https://huggingface.co/hkunlp/instructor-large)

If you look into the source code, the low-level embedding will be like this

```bash
!pip install InstructorEmbedding
```

```python
from InstructorEmbedding import INSTRUCTOR

model = INSTRUCTOR('hkunlp/instructor-large')
sentence = "3D ActionSLAM: wearable person tracking in multi-floor environments"
instruction = "Represent the Science title:"
embeddings = model.encode([[instruction,sentence]])
print(embeddings)
```

Compare to **sentence-transformers/all-MiniLM-L6-v2,** there is no _**mean_pooling function or AutoTokenizer or AutoModel**_. It is just INSTRUCTOR. 
So how do we incorporate to **inference.py**?

First, this model requires to install InstructorEmbedding, so we need to install this library to the inference endpoint.

You need to create a **requirements.txt** with this content.

```typescript
InstructorEmbedding
```

Modify the **inference.py** to follow instructor-embedding code.

```python
from InstructorEmbedding import INSTRUCTOR

def model_fn(model_dir):
  model = INSTRUCTOR(model_dir)
  return model

def predict_fn(data, model):
  embeddings = model.encode(data)
  return {"vector_embedding": sentence_embeddings.tolist()}
```

### Create model.tar.gz with inference script and model

```ini
repository = "hkunlp/instructor-large"
model_id=repository.split("/")[-1]
s3_location=f"s3://{sess.default_bucket()}/custom_inference/{model_id}/model.tar.gz"
```

```bash
## download the model from hf model
!git lfs install
!git clone https://huggingface.co/$repository
```

```bash
## copy inference.py and requirements.txt to code/ directory
## this is mandatory to have files under code/ folder
!cp -r inference.py $model_id/code/
!cp -r requirements.txt $model_id/code/

## create model.tar.gx archive with all the model artifacts
%cd $model_id
!tar zcvf model.tar.gz *

## upload to S3
!aws s3 cp model.tar.gz $s3_location
```

**Create a custom Huggingface Model**

```python
from sagemaker.huggingface.model import HuggingFaceModel
# create Hugging Face Model Class
huggingface_model = HuggingFaceModel(
   model_data=s3_location,       # path to your model and script
   role=role,                    # iam role with permissions to create an Endpoint
   transformers_version="4.26",  # transformers version used
   pytorch_version="1.13",        # pytorch version used
   py_version='py39',            # python version used
)
# deploy the endpoint endpoint
predictor = huggingface_model.deploy(
    initial_instance_count=1,
    instance_type="ml.m5.xlarge"
)
```

After this, you should have the embedding of real-time inference in the Sagemaker console.

Now, you have the instructor-large embedding inference endpoint running on Sagemaker, let's integrate it into Langchain.

## Langchain with Sagemaker Real-time embedding Endpoint

Langchain provides the `[SagemakerEndpointEmbeddings](https://notebooks.githubusercontent.com/view/ipynb?browser=chrome&bypass_fastly=true&color_mode=light&commit=1308b2c7f0b5c259e02a3a6573c1ca2e6f1c073f&device=unknown_device&docs_host=https%3A%2F%2Fdocs.github.com&enc_url=68747470733a2f2f7261772e67697468756275736572636f6e74656e742e636f6d2f6177732d73616d706c65732f6c6c6d2d617070732d776f726b73686f702f313330386232633766306235633235396530326133613635373363316361326536663163303733662f626c6f67732f7261672f646174615f696e67657374696f6e5f746f5f766563746f7264622e6970796e62&logged_in=true&nwo=aws-samples%2Fllm-apps-workshop&path=blogs%2Frag%2Fdata_ingestion_to_vectordb.ipynb&platform=windows&repository_id=629772172&repository_type=Repository&version=114)` class which is a wrapper around a functionality to talk to a Sagemaker Endpoint to generate embeddings. We will override the `embed_documents` function to define our own batching strategy for sending requests to the model (multiple requests are sent in one model invocation). Similarly, we extend the `ContentHandlerBase` class to provide implementation for two abstract methods which define how to process (encode/decode) the input data sent to the model and the output received from the model.

We finally create a `SagemakerEndpointEmbeddingsJumpStart` object that puts all this together and can now be used by langchain to talk to an LLM deployed as a Sagemaker Endpoint to generate embeddings.

```ruby
"""
Helper functions for using Samgemaker Endpoint via langchain
"""
import time
import json
from typing import List
from langchain.embeddings import SagemakerEndpointEmbeddings
from langchain.embeddings.sagemaker_endpoint import EmbeddingsContentHandler


# extend the SagemakerEndpointEmbeddings class from langchain to provide a custom embedding function
class SagemakerEndpointEmbeddingsJumpStart(SagemakerEndpointEmbeddings):
    def embed_documents(
        self, texts: List[str], chunk_size: int = 5
    ) -> List[List[float]]:
        """Compute doc embeddings using a SageMaker Inference Endpoint.

        Args:
            texts: The list of texts to embed.
            chunk_size: The chunk size defines how many input texts will
                be grouped together as request. If None, will use the
                chunk size specified by the class.

        Returns:
            List of embeddings, one for each text.
        """
        results = []
        _chunk_size = len(texts) if chunk_size > len(texts) else chunk_size
        st = time.time()
        for i in range(0, len(texts), _chunk_size):
            response = self._embedding_func(texts[i:i + _chunk_size])
            results.extend(response)
        time_taken = time.time() - st
        print(f"got results for {len(texts)} in {time_taken}s, length of embeddings list is {len(results)}")
        return results


# class for serializing/deserializing requests/responses to/from the embeddings model
class ContentHandler(EmbeddingsContentHandler):
    content_type = "application/json"
    accepts = "application/json"

    def transform_input(self, prompt: str, model_kwargs={}) -> bytes:
        input_str = json.dumps(prompt)
        return input_str.encode('utf-8') 

    def transform_output(self, output: bytes) -> str:
        response_json = json.loads(output.read().decode("utf-8"))
        embeddings = response_json["vector_embedding"]
        if len(embeddings) == 1:
            return [embeddings[0]]
        return embeddings

def create_sagemaker_embeddings_from_js_model(embeddings_model_endpoint_name: str, aws_region: str) -> SagemakerEndpointEmbeddingsJumpStart:
    # all set to create the objects for the ContentHandler and 
    # SagemakerEndpointEmbeddingsJumpStart classes
    content_handler = ContentHandler()

    # note the name of the LLM Sagemaker endpoint, this is the model that we would
    # be using for generating the embeddings
    embeddings = SagemakerEndpointEmbeddingsJumpStart( 
        endpoint_name=embeddings_model_endpoint_name,
        region_name=aws_region, 
        content_handler=content_handler
    )
    return embeddings
```

Init the wrap parameters:

```python
embeddings_model_endpoint_name = "<the name of your embedding endpoint model from Sagemaker>"
content_handler = ContentHandler()
embeddings_endpoint = create_sagemaker_embeddings_from_js_model(embeddings_model_endpoint_name, '<your-aws-region> (ap-southeast-2 for example)')
```

### Add Embedding Real-time Model to Chroma

```python
# import
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.chat_models import ChatOpenAI

# load the document and split it into chunks
loader = UnstructuredPDFLoader("./documents/Apple-10k-report.pdf")
documents = loader.load()
# split it into chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

# load it into Chroma
db = Chroma.from_documents(docs, embeddings_endpoint)

chat = ChatOpenAI(temperature=0.2, model_name='gpt-3.5-turbo', 
                                 openai_api_key="YOUR_API_KEY", 
                                 openai_organization="YOUR_ORGANIZATION_ID")
chain = RetrievalQA.from_chain_type(chat,chain_type="stuff", 
                             retriever=docsearch.as_retriever())
chain.run("What is revenue last year?")
```

### Add Embedding Real-time Model to OpenSearch Service

This is totally optional. If you have followed the yaml setup, you may already install the OpenSearch service. As mentioned, having Chroma in your local machine/notebook is fine for the experiment. If you need to bring your app to a high level of scale, then perhaps you need to either use a commercial vector database such as Pinecone or deploy your own database like Chroma on EC2 or AWS Managed OpenSearch service. Here is how to ingest data into OpenSearch with the real-time embedding endpoint.

```python
# import
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import OpenSearchVectorSearch

# load the document and split it into chunks
loader = UnstructuredPDFLoader("./documents/Apple-10k-report.pdf")
documents = loader.load()
# split it into chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

chat = ChatOpenAI(temperature=0.2, model_name='gpt-3.5-turbo', 
                                 openai_api_key="YOUR_API_KEY", 
                                 openai_organization="YOUR_ORGANIZATION_ID")

docsearch = OpenSearchVectorSearch(
    index_name="index-10k-report",
    embedding_function=embedding,
    opensearch_url= "<your_opensearch_url>",
    http_auth=("<your-opensearch-username>",'<your-opensearch-password>'),
    use_ssl=False,
    verify_certs=False,
    ssl_assert_hostname=False,
    ssl_show_warn=False    
)
docsearch.add_documents(documents=docs)

chain = RetrievalQA.from_chain_type(chat,chain_type="stuff", 
                             retriever=docsearch.as_retriever())
chain.run("What is revenue last year?")
```

## Wrap up

Wow, we've covered a lot in this post! As we've reached the end, I'm thinking about splitting the content into future posts to keep the journey going.

Deploying HuggingFace embedding models has been both challenging and incredibly rewarding. You should now have a good grasp of the basics and fundamentals needed to bring any HuggingFace embedding model to a real-time endpoint on Sagemaker.

I genuinely hope this article has been helpful to you in some way. If you found it valuable, a little clap would mean the world to me. Don't forget to follow me to stay updated on all the exciting content that's coming up!

If you have any questions? Feel free to leave a comment, and I promise to get back to you as soon as possible. Your feedback and involvement are incredibly valuable to me.

## Reference:

**OpenSearch**: [https://opensearch.org/blog/semantic-search-solutions/](https://opensearch.org/blog/semantic-search-solutions/)
**Chroma**: [https://github.com/chroma-core/chroma](https://github.com/chroma-core/chroma)

**Langchain + Sagemaker**: [https://python.langchain.com/docs/integrations/sagemaker_endpoint](https://python.langchain.com/docs/integrations/sagemaker_endpoint)

**Reach out to me on LinkedIn**: [https://www.linkedin.com/in/ryan-nguyen-abb844a4/](https://www.linkedin.com/in/ryan-nguyen-abb844a4/)
**Reach out to me on Twitter**: [https://twitter.com/kiennt_](https://twitter.com/kiennt_)

**Subscribe to my substack as I cover more ML in-depth:** [https://howaibuildthis.substack.com/](https://howaibuildthis.substack.com/)