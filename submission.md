1. Understanding of codebase:
   Goal of project: automated documentation generation of a given codebase using GPT 3.5, FastAPI and the BERT model
   Process: uploading a zip file of the codebase allows for the generation of the documentation of the codebase
   Procedure by which the goal is achieved in brief (server):
   1. creates a folder and extracts the zip file which is uploaded
   2. reads the file and traverses it using the breadth first search algorithm, and processing of files takes place using the BERT model and tokenizer
   3. embeddings are created during traversal and clustering is performed upon these embeddings
   4. GPT is used to generate tests and documentation and mocking mechansim is used for the process
   5. Document generation takes place using GPT 3.5 model and makes use of a system prompt and user prompt to do the same, similar to how code refactoring and test generation is done.
   6. routers are used for endpoint connections
   The client folder contains the code to implement these functions in the front end and display the end result 

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
  1. Dendrogram feature addition:
     - the linkage method from the heirarchy class was called to generate the dendrogram
     - the parameters passed were the list itself used for clustering, the metric of closeness, which was cosine as used in other areas of the codebase (during embeddings)
     - matplotlib was the library used for plotting the dendrogram and the plt.show() method was used for displaying the dendrogram
     - this functionality was added within the function clustering itself so that the dendrogram can be displayed along with the clustering done later when the function is called in the file file_traversal.py
     - the seaborn library could also be used for the purpose of plotting the dendrograms in this case
     - either method could be used for plotting with no advantage over the other
     - the code is written in the file convert_embeddings.py:
     - Z = hierarchy.linkage(list1, method='complete', metric='cosine')
     - 
       plt.figure(figsize=(10, 6))
       
       hierarchy.dendrogram(Z)
       
       plt.title('Dendrogram')
       
       plt.xlabel('Data points')
       
       plt.ylabel('Distance')
       
       plt.show()
       
   2. Additional Code Features:
      - code optimisation is the feature added to the codebase for suggesting improvements in the codebase
      - it can also be used for reasons similar to code refactoring (used for enhancing code by retaining most important aspects)
      - this is done using a similar process to adding code refactoring and generating tests for the codebase
      - system prompts and user prompts are given based on the task needed to be performed (in this case code optimization) and the call_openai_api function is called to perform the optimisation task
      - the call_openai_api function makes use of gpt turbo 3.5 model which makes use of tranformer mechanism to carry out this function, the output is stored in a file 
      - the function is routed at the backend and is also represented in the frontend
      - another way to add this functionality would be to mention it in the user prompt of the code refactoring function and increase the number of tokens of the response
      - the code is written in the file infinite_gpt.py:
     
      - 
         def ask_gpt_to_optimize_code(prompt_text, output_folder):
        
       system_prompt = """You are a skilled software engineer specializing in code optimization and performance improvements."""
   
       user_prompt = """Analyze the following code for performance bottlenecks and suggest optimizations to improve its efficiency. Provide only the optimized code and avoid any additional explanations or comments."""
       
       output_file = f'{output_folder}/optimized_code.txt'
   
       print(SHOULD_MOCK_AI_RESPONSE)
   
       if SHOULD_MOCK_AI_RESPONSE == 'True':
      
           print("Mocking AI response")
      
           mock_chunks_gpt(prompt_text, output_file)
      
       elif SHOULD_MOCK_AI_RESPONSE == 'False':
      
           print("Calling OpenAI API")
      
           response = call_openai_api(prompt_text, system_prompt, user_prompt)
      
           print(response)
      
           save_to_file(response, output_file)
      

      - the frontend representation is under the folder optimize_code
 4. Document customisation:
   - Iterative prompting is used for performing this task
   - the generate_text function under the call_openai_api_higher_tokens function is used for generating test using gpt_turbo_3.5 model based on the system prompt and user prompt
   - these prompts are given ib the code itself in the list 'messages'
   - a while loop is used to carry out iterative prompting until the user is satisfied with the results of the documentation
   - the feedback variable takes in the input of any customisation the user wants to add to the documentation and calls the generate test function to customise the documentation accordingly
   - the code is written in the infinite_gpt.py file:

   - 
   - #@backoff.on_exception(backoff.expo, openai.error.RateLimitError)
   - 
   def call_openai_api_higher_tokens(text, output_file):

       def generate_text(messages):
     
           response = openai.ChatCompletion.create(
     
               model="gpt-3.5-turbo-16k",
     
               messages=messages,
     
               max_tokens=2000,
     
               n=1,
     
               stop=None,
     
               temperature=0.5,
     
           )
     
           return response.choices[0].message['content']
     
   
       messages = [
     
           {"role": "system", "content": "You are a smart technical writer who understands code and can write documentation for it."},
     
           {"role": "user", "content": f"Give me a developers documentation of the following code. Give a brief intro, table of contents, function explanations, dependencies, API specs (if present), schema tables in markdown. Give in markdown format and try to strict to the headings\n\n: {text}."},
     
       ]
     
   
       while True:
     
           response = generate_text(messages)
     
           print(response)
     
   
           feedback = input("Is there any more customisation you would like to add? (yes/no): ")
     
           if feedback.lower() == "yes":
     
               break
     
           else:
     
               refinement = input("How can I customise your documentation? ")
     
               messages[1]["content"] += " " + refinement
     

       save_to_file(response, output_file)
 4. Investigating clustering mechanisms:
    - The DBSCAN clustering algorithm was used for clustering the embeddings
    - it is an algorithm that works best for high density variations in the data and also for large amount of data
    - DBSCAN does probablistic assignment of datapoints in clusters and is better at detecing outliers as compared to other algorithms
    - it also has the advantage of not needing to mention the number of clusters earlier, which was a hyperparameter in the agglomerative clustering mechanism
    - a function is written to perform the DBSCAN clustering which takes in two hyperparameters, namely epsilon and minimum number of samples
    - epsilon is the distance from the centre of the cluster which takes in other points to be made a part of the cluster, whereas the minimum sample represents the minimum number of points needed to make a cluster in the algorithm
    - the best parameters for this function by performing hyperparameter tuning and the metric for evalutation of the clusters is silhouette score
    - the best value search of epsilon is done in the range of 10 to 50 with the step size of 5, which is ideal for datasets in which the embeddings have not been normalized
    - the best value search of minimum number of samples is done in the range of 10 to 50 because of a larger dataset with varying densities
    - other methods which could be used in this case were KMeans clustering, but it wasn't used due to the following disadvantages:
    - sensitivity to centroid
    - specifying the number fo clusters
    - senstivity to outliers
    - assumes all clusters to be spherical
    - the code for this is under a file called convert_embeddings2.py:

     
     - from sklearn.cluster import DBSCAN
     - 
      from sklearn.metrics import silhouette_score

      import numpy as np
     
      
      #For hyperparameter tuning of eps and min_samples
     
      def tune_dbscan(X, eps_range, min_samples_range):
     
        best_score = -1
     
        best_params = []
     
        for eps in eps_range:
     
          for min_samples in min_samples_range:
     
            clustering = DBSCAN(eps=eps, min_samples=min_samples)
     
            labels = clustering.fit_predict(X)
     
            if len(set(labels)) > 1:  # Avoid single-cluster solutions
     
              score = silhouette_score(X, labels)
     
              if score > best_score:
     
                best_score = score
     
                best_params = [eps, min_samples]
     
        return best_params
     
          
      def clustering1(list1):
     
          # Convert list1 to a numpy array if it's not already
     
          X = np.array(list1)
     
          eps_range = np.arange(10, 50, 5)
     
          min_samples_range = range(10, 50)
     
          tune_dbscan(X, eps_range, min_samples_range)
     
          
          # DBSCAN parameters
     
          dbscan = DBSCAN(eps=best_params[0], min_samples=best_params[1], metric='cosine').fit(X)
     
          
          arr = dbscan.labels_
     
          unique_values = np.unique(arr)
     
          
          indices_list = []
     
          for val in unique_values:
     
              indices = np.where(arr == val)[0]
     
              indices_list.append(indices)
     
          
          return indices_list

      

4. Suggesting improvements in the acrhitecture of the process:
   1.
   - Instead of the BERTmodel for embeddings, a superior model called RoBERTa could have been used
   - It has the following advantages over the BERT model:
   - RoBERTa has a much larger training dataset than the BERT model which leads better learning of the model, thus improved accuracy
   - RoBERTa uses larger mini-batches and more training steps than BERT
   - One of the BERT model's training objectives was next sentence prediction, which means that it would predict if two sentences are consecutive in a given paragraph, which when removed gives better accuracy.This was done in the RoBERTa model
   2.  
   - 
