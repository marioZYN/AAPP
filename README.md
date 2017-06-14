This course project is based on the paper "Parallelizing Word2Vec in Shared and Distributed Memory"

Advanced Algorithm and Parallel Programming project, Politecnico di Milano

Author : ZHOU YINAN 
   	 yinan.zhou@mail.polimi.it

Academic Year: 2017

## Description

Word2Vec is used in NLP(Natural Language Processing) field which uses neural network to learn a numerical vector presentation for each word. Words having similar vector representation means having similar meanings. The original Word2Vec code is implemented by Google in C language and Pthread. You can find information here : https://code.google.com/archive/p/word2vec/

My implementation of Word2Vec is based on the original version and make some modifications. There are mainly two differences. First, instead of using stochastic gradient descent, I use mini-batch to train the neural network. Second, instead of using Pthread, I use OpenMP for parallelization. 
The file presentation.ppt provides the general idea of paper and some theory interpretation. 

This README file provides some descriptions and instructions on the project. Also some explanations on the code is provided for convenience. 

## Files

The zip file contains 6 documents:
- train.cpp
- demo.sh
- README.txt
- presentation.ppt
- license
- distance.c

## Running

By typing `./demo.sh`, it'll automatically compile the code and begin running. Note that it'll download the dataset text8 for training the neural netowrk. The training phase will last for some time. When training completes, an 'output' file will be genereated and the testing phase begins. In testing, you can input a word and get the most similar words as result. Typing "EXIT" to end testing. 

Besides the demo.sh, you can also compile the code yourself :
(distance.c is used to test the result)

g++ ./train.cpp -fopenmp [-parameters]
gcc ./distance.c -o distance -lm

some user defined parameters are listed here. If they are not specified, the default value will be used.

-**train\<file>** 

Use text data from <file> to train the neural network

-**size** 

Set the size of the word-vectors. Default value is 100

-**window**

Set max skip length between words. Default value is 5

-**sample**  

Sample parameter will be used for over/under sampling to keep the word frequence rank. Default is 1e-3, useful range is (0, 1e-5)

-**negative**  

Number of negative samples. Default value is 5

-**threads**  

Set the number of threads to run. Default value is 2

-**iter**  

Number of training iterations. Default value is 5

-**min-count**  

The word appears less than this threshold will be discarded

-**alpha**  

Set the starting learning rate. Default value is 0.025

-**save-vocab\<file>**  

The vocab file with words and counts will be stored

-**read-vocab\<file>**  

The vocabulary-count will be read from file

-**batch-size**  

The batch size used for mini-batch. Default value is 11

## Code explanation

The project implementation is based on the original C code by Google. A lot of codes are reused and some modifications are made. Nevertheless the C code is somehow difficult to read due to the lack of comments and somes tricks used to speedup the process. I add some comments in the code to help reading and some tricks are explained here for convenience. 

* sigma function approximation

In the training phase, sigma function is used to compute the predicted label. Since it needs to be computed in each iteration, a pre-computed table is used as a dictionary to speedup the process. 

```
expTable = (real *) malloc((EXP_TABLE_SIZE + 1) * sizeof(real));

for (int i = 0; i < EXP_TABLE_SIZE + 1; i++) {
     
     expTable[i] = exp((i / (real) EXP_TABLE_SIZE * 2 - 1) * MAX_EXP); // Precompute the exp() table
     expTable[i] = expTable[i] / (expTable[i] + 1);                    // Precompute f(x) = x / (x + 1)
}

```


* \</s>

In the word dictionary, </s> is used to indicate the end of one sentence. 

* dealing with high/low frequency words

Low frequency words (appear time less than threshold min_count) are discarded.
High frequency words are treated using sub-sampling. Large frequency words like "the, a , an..." appear quite a lot in the paragraph, but provide little information. When doing random sampling in the training, this factor is taken into account. High frequency words have a certain probability to be discarded when being sampled.

