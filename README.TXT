NLP-KL PROJECT
RECOGNIZER OF MODEL FOR AI-GENERATED TEXTS


IN THE FOLDER YOU WILL FIND:
    1. Folder data, containing the raw data for train and test sets, in format .jsonl
    2. Folder preprocessed_and_tokenized_data, containing the processed and tokenized data for training
       and test sets, in format .parquet
    3. Python file preprocess.py, which was used to preprocess and tokenize the datasets. Also the functions
       in the file can be imported and used to preprocess and tokenize new raw data.
    4. The Jupyter notebook nlp_project.ipynb, containing the code for loading and vectorizing the data,
       and building the models. The code is divided in blocks.

-------------------------------------------------------------------------------------------------------------
1. DATA
-------------------------------------------------------------------------------------------------------------

The test set doesn't contains labels, so I divided the train set in two subsets for
training and testing the model during the preprocessing. The split was 75% train and
25% test, using random_state = 2024 for reproducibility of the results.

-------------------------------------------------------------------------------------------------------------
2. PREPROCESSED DATA
-------------------------------------------------------------------------------------------------------------

Subsets created during preprocessing for training and testing the model. in the processed
datasets we have 44062 labeled train examples and 14686 labeled test examples. 

-------------------------------------------------------------------------------------------------------------
3. preprocess.py
-------------------------------------------------------------------------------------------------------------

In this file, I define 4 functions to preprocess, tokenize and save the data. During the
preprocessing I transform all the characters to lowercases and remove extra whitespaces, 
numbers and punctuation. 

-------------------------------------------------------------------------------------------------------------
4. nlp_project.ipynb
-------------------------------------------------------------------------------------------------------------

Here we load the preprocessed data and then we build the models. We train 2 machine learning
models (SVC and Logistic Regression) and a deep learning model (MLPClassifier). All the 
models were created with the scikit-learn library. For the two machine learning models I
adjusted 36 pairs of hyperparameters (C and tol). In the neural network, I couldn't 
adjust any hyperparameter because of the computational cost. The metric used to evaluate the models
was the micro-averaged f1_score. Remember that f1 = 2*(recall*precission)/(recall + precission),
and its value belongs to the interval [0,1].

-------------------------------------------------------------------------------------------------------------
5. RESULTS 
-------------------------------------------------------------------------------------------------------------

After adjusting the hyperparameters, I obtained the next results:

For the SVC, the best combination found was C = 0.1 and tol = 0.1.
The mean of the f1_score provided by this combination was arround 0.7.

In the case of the Logistic Regression the best hyperparameters were C = 0.1 and tol = 0.175.
One more time, the mean of the f1_score obtained with this model is arround 0.7

In the case of the Multi Layer Perceptron (MLPClassifier), I couldn't adjust any
hyperparameter because the computational cost and time elapsed.

The f1_score obtained with the default parameters was also arround 0.7.

-------------------------------------------------------------------------------------------------------------
6. CONCLUSIONS
-------------------------------------------------------------------------------------------------------------

It's true that all the models had almost the same f1_score, but not all took the same time to train.
The slowest was the MLPClassifier, taking arround 2 hours for only training 1 model. Is normal this
model is not performing at its best because Multi Layer Perceptrons are not optimal for processing
sequences. Also the other models had their hyperparameters adjusted and this one no.

In the case of the Logistic Regression, adjusting the 36 pairs of hyperparameters took also arround 
2 hours, reaching some times the max_iterations = 1000. It took the same time training 36 Logistic Regression
models and training one only MLPClassifier model, obtaining similar f1_scores.

The most efficient model was the SVC. This model achieved also around 0.7 f1_score, but only took arround 35
minutes to train and test the 36 models to see which pair of hyperparameters was the optimal. Also the best
value for the hyperparameter C was the lowest considered, so maybe its optimal value is even lower and it
can reach a better accuracy being much faster than the other two. 

So, our final conclussion is SVC is the best of the three models considered for this specific problem.
Maybe a Recurrent Neural Network (RNN) could achieve a higher f1_score, but also will be for sure more
time-consuming than SVC. 

-------------------------------------------------------------------------------------------------------------
7. ESPECIFICATIONS
-------------------------------------------------------------------------------------------------------------

Processor	11th Gen Intel(R) Core(TM) i5-1155G7 @ 2.50GHz   2.50 GHz
RAM installed	16,0 GB (15,8 GB usable)
OS of 64 bits, x64 architecture