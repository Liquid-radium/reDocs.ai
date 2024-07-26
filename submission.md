1. Understanding of codebase:
   Goal of project: automated documentation generation of a given codebase using GPT 3.5, FastAPI and the BERT model
   Process: uploading a zip file of the codebase allows for the generation of the documentation of the codebase
   Procedure by which the goal is achieved in brief (server):
   1. creates a folder and extracts the zip file which is uploaded
   2. reads the file and traverses it using the breadth first search algorithm, and processing of files takes place using the BERT model and tokenizer
   3. embeddings are created during traversal and clustering is performed upon these embeddings
   4. GPT is used to generate tests and documentation and mocking mechansim is used for the process
   5. routers are used for endpoint connections
   The client folder contains the code to implement these functions in the front end and display the code documentation

2. Explaining the machine learning aspects of algorithms used:
   1. Codebase traversal:
      - the codebase is traversed, snippets and embeddings are extracted using the process_files function
      - embeddings are created in the following manner: the codebase is cleaned up by removing characters, and tokenisation takes place, which are vectorised (converted into numbers which represent meaningful information about the word and are stored)
      - this is appended to the empty list code_for_gpt and embeddings respectively which can be used for further natural language processing tasks
      - the embeddings with maximum length is found out and the rest of the embeddings are padded for consistent input into a neural network, which in this case is GPT (improves effeciency). GPT makes use of transformer architecture, which is a special kind of neural network system.
      - depending on whether the mock_response is true or false, OpenAI API is called, in which the system prompt and used prompts are given to the GPT
      - the system prompt acts as the short statement which helps the GPT generate the required output based on the input prompt
    2. Code embeddings:
       - embeddings are created using BertTokenizer
       - tokens are created using the tokenize function and special tokens are added at the beginning and end of each token sequence for the following reasons: it marks the boundary for each sequence, and for attention mechanism (which is used in GPT) to anchor the embedding in place
       - these token sequences are then converted into tensors using pytorch, which is a compatible format to feed into a neural network. This is then fed into a neural network based model which returns the embeddings needed
       - the size and representation of the embeddings is handled using the embeddings function
    3. Handling large code files:
       - code refactoring helps handle large code files by enhancing the readability and retaining classes and functions which have higher significance in the overall block of code
       - code refactoring also looks for code smells (for example, repetitive blocks of cdoe performing the same functions, code blocks taking large computational power)
       - code refactoring is done using GPT in these steps: tokenisation, embeddings, attention mechanism, decoding
       - the encoder performs the task of tokenisation and creating embeddings, while the decoder contains transformer architecture which consists of multi-head attention mechanisms
       - these multi-head attention blocks calculate attention scores on the embeddings simultaneously thus enriching the model further than if a single head of attention was used
       - these are fed into a softmax function which calculates the probability of the following word and its meaning/context with respect to the previous blocks of code
       - the decoder then has options of new words to choose from depending on the probabilities and the temperature of the attention block
    4. Maintaining context with agglomerative clustering:
       - agglomerative clustering uses a bottom-up method of clustering which assumes all data point (code in this case) as single clusters, and combines them based on a metric of closeness (usually euclidean distance or cosine)
       - this process continues until all data is fit under a cluster (agglomaertive clustering does hard assignment). Some advantages of this are as follows
       - this is a method of unsupervised learning which increases effeciency of the transformer model in GPT while also reducing computation due to reduced features
       - clustering also helps in reducing overfitting, which is a common problem faced while making an accurate machine learning model
       - new features and context can be observed using the centroids or hierarchial information in the clusters by the transformers, which leads to higher accuracy
    5. Effecient document generation:
       - effecient document generation process is enhanced by the presence of code refactoring and testing
       - this is done using the GPT 3.5 model, using system and user prompts to perform the required tasks
       - a function called call_openai_api is defined which takes in text, system prompt and user prompt as parameters
       - roles are assigned to the user prompt and system prompt, and the response is limited to 4000 tokens, with temperature being 0.5 to allow for less probable words to be generated
       - two functions are used for code refactoring and test generation and the user prompt and system prompts are assigned using prompt engineering within the function depending on the task the function has to 
         perform
       - the call_openai_api function is called within these two functions to allow for generation of tests and code refactoring
       - code refactoring improves the quality of the document generated by enhancing readability for the transformer model which will generate the documentation for the original codebase
       - test generation and evaluation allows to understand the shortcomings of the model and provides a method of evaluation of the model
       - another function is written which generates the documentation using system and user prompts, having a maximum response length of 2000 tokens.

3. Tasks handled:

  
