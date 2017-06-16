//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.


#include <cstring>
#include <stdio.h>
#include <cmath>
#include <omp.h>
#include <algorithm>

using namespace std;

#define MAX_STRING 100
#define EXP_TABLE_SIZE 1000
#define MAX_EXP 6
#define MAX_SENTENCE_LENGTH 1000

typedef float real;
typedef unsigned long long ulonglong;

struct vocab_word {
    unsigned int  cn;
    char *word;
};

int negative = 5;
int min_count = 5; // threshold for discarding low frequency words
int num_threads = 2;
int min_reduce = 1; // used for reducing dictionary size
int iter = 3;
int window = 5;
int batch_size = 11;
int vocab_max_size = 1000;
int vocab_size = 0;
int hidden_size = 100;
int file_size = 0;
ulonglong train_words = 0;
real alpha = 0.025f; // learning rate
real sample = 1e-3f;
const real EXP_RESOLUTION = EXP_TABLE_SIZE / (MAX_EXP * 2.0f);
char train_file[MAX_STRING];
char output_file[MAX_STRING];
char save_vocab_file[MAX_STRING];
char read_vocab_file[MAX_STRING];
const int vocab_hash_size = 30000000;
const int table_size = 1e8;

struct vocab_word *vocab = NULL;
int *vocab_hash = NULL;
int *table = NULL;  // used for sampling
real *M_in = NULL;  // input matrix
real *M_out = NULL; // output matrix
real *expTable = NULL;  // used as a lookup table for sigmoid function 

// used for negative sampling
void InitUnigramTable() {

  table = (int *)malloc(table_size * sizeof(int));
  const real power = 0.75f;
  double train_words_pow = 0;
  for(int i = 0; i < vocab_size; i++){
    train_words_pow += pow(vocab[i].cn, power);
  }

  int i = 0;
  real d1 = pow(vocab[i].cn, power) / train_words_pow;
  for (int a = 0; a < table_size; a++) {
    table[a] = i;
    if (a / (real) table_size > d1){
      i++;
      d1 += pow(vocab[i].cn, power) / train_words_pow;
    }
    if (i >= vocab_size)
      i = vocab_size - 1;
  }



/*------------------------OpenMP Testing-----------------------------------------*/
  /*
  int n_proc = omp_get_num_procs();
  int n_thread = omp_get_num_threads();
  int max_thread = omp_get_thread_limit();
  int in_parallel = omp_in_parallel();
  int dynamic = omp_get_dynamic();
  int nested = omp_get_nested();

  #pragma omp parallel num_threads(4)
  {

    printf("I am thread %d\n",omp_get_thread_num());

    #pragma omp barrier
    printf("This is in barrier\n");

  #pragma omp master
    {
       n_proc = omp_get_num_procs();
       n_thread = omp_get_num_threads();
       max_thread = omp_get_thread_limit();
       in_parallel = omp_in_parallel();
       dynamic = omp_get_dynamic();
       nested = omp_get_nested();
    printf("number if processors = %d\n",n_proc);
    printf("current threads number = %d\n",n_thread);
    printf("max number of thread number = %d\n",max_thread);
    printf("in paralle ? %d\n",in_parallel);
    printf("dynamic able ? %d\n",dynamic);
    printf("nested able ? %d\n",nested);
  }
  }

  */
}

// Reads a single word from a file, assuming space + tab + EOL to be word boundaries
void ReadWord(char *word, FILE *fin) {
    int a = 0, ch;
    while (!feof(fin)) {
        ch = fgetc(fin);
        if (ch == 13)
            continue;
        if ((ch == ' ') || (ch == '\t') || (ch == '\n')) {
            if (a > 0) {
                if (ch == '\n')
                    ungetc(ch, fin);
                break;
            }
            if (ch == '\n') {
                strcpy(word, (char *) "</s>");
                return;
            } else
                continue;
        }
        word[a] = ch;
        a++;
        if (a >= MAX_STRING - 1)
            a--;   // Truncate too long words
    }
    word[a] = 0;
}

// Returns hash value of a word
int GetWordHash(char *word) {
    unsigned int hash = 0;
    for (int i = 0; i < strlen(word); i++)
        hash = hash * 257 + word[i];
    hash = hash % vocab_hash_size;
    return hash;
}

// Returns position of a word in the vocabulary; if the word is not found, returns -1
int SearchVocab(char *word) {
    int hash = GetWordHash(word);
    while (1) {
        if (vocab_hash[hash] == -1)
            return -1;
        if (!strcmp(word, vocab[vocab_hash[hash]].word))
            return vocab_hash[hash];
        hash = (hash + 1) % vocab_hash_size;
    }
    return -1;
}

// Reads a word and returns its index in the vocabulary
int ReadWordIndex(FILE *fin) {
    char word[MAX_STRING];
    ReadWord(word, fin);
    if (feof(fin))
        return -1;
    return SearchVocab(word);
}

// Adds a word to the vocabulary
int AddWordToVocab(char *word) {
    int hash, length = strlen(word) + 1;
    if (length > MAX_STRING)
        length = MAX_STRING;
    vocab[vocab_size].word = (char *) calloc(length, sizeof(char));
    strcpy(vocab[vocab_size].word, word);
    vocab[vocab_size].cn = 0;
    vocab_size++;
    // Reallocate memory if needed
    if (vocab_size + 2 >= vocab_max_size) {
        vocab_max_size += 1000;
        vocab = (struct vocab_word *) realloc(vocab, vocab_max_size * sizeof(struct vocab_word));
    }
    hash = GetWordHash(word);
    while (vocab_hash[hash] != -1)
        hash = (hash + 1) % vocab_hash_size;
    vocab_hash[hash] = vocab_size - 1;
    return vocab_size - 1;
}

// Used later for sorting by word counts
int VocabCompare(const void *a, const void *b) {
    return ((struct vocab_word *) b)->cn - ((struct vocab_word *) a)->cn;
}

// Sorts the vocabulary by frequency using word counts
void SortVocab() {
    // Sort the vocabulary and keep </s> at the first position
    qsort(&vocab[1], vocab_size - 1, sizeof(struct vocab_word), VocabCompare);
    memset(vocab_hash, -1, vocab_hash_size * sizeof(int));

    int size = vocab_size;
    train_words = 0;
    for (int i = 0; i < size; i++) {
        // Words occuring less than min_count times will be discarded from the vocab
        if ((vocab[i].cn < min_count) && (i != 0)) {
            vocab_size--;
            free(vocab[i].word);
        } else {
            // Hash will be re-computed, as after the sorting it is not actual
            int hash = GetWordHash(vocab[i].word);
            while (vocab_hash[hash] != -1)
                hash = (hash + 1) % vocab_hash_size;
            vocab_hash[hash] = i;
            train_words += vocab[i].cn;
        }
    }
    vocab = (struct vocab_word *) realloc(vocab, vocab_size * sizeof(struct vocab_word));
}

// Reduces the vocabulary by removing infrequent tokens
void ReduceVocab() {
    int count = 0;
    for (int i = 0; i < vocab_size; i++) {
        if (vocab[i].cn > min_reduce) {
            vocab[count].cn = vocab[i].cn;
            vocab[count].word = vocab[i].word;
            count++;
        } else {
            free(vocab[i].word);
        }
    }
    vocab_size = count;
    memset(vocab_hash, -1, vocab_hash_size * sizeof(int));

    for (int i = 0; i < vocab_size; i++) {
        // Hash will be re-computed, as it is not actual
        int hash = GetWordHash(vocab[i].word);
        while (vocab_hash[hash] != -1)
            hash = (hash + 1) % vocab_hash_size;
        vocab_hash[hash] = i;
    }
    min_reduce++;
}


// get word and counts from file
void LearnVocabFromTrainFile() {
    char word[MAX_STRING];

    memset(vocab_hash, -1, vocab_hash_size * sizeof(int));
    FILE *fin = fopen(train_file, "rb");
    if (fin == NULL) {
        printf("ERROR: training data file not found!\n");
        exit(1);
    }

    train_words = 0;
    vocab_size = 0;
    AddWordToVocab((char *) "</s>");
    while (1) {
        ReadWord(word, fin);
        if (feof(fin))
            break;
        train_words++;
        if ( train_words % 100000 == 0) {
            printf("%lldK%c", train_words / 1000, 13);
            fflush(stdout);
        }
        int i = SearchVocab(word);
        if (i == -1) {
            int a = AddWordToVocab(word);
            vocab[a].cn = 1;
        } else
            vocab[i].cn++;
        if (vocab_size > vocab_hash_size * 0.7)
            ReduceVocab();
    }
    SortVocab();
    printf("Vocab size: %d\n", vocab_size);
    printf("Words in train file: %lld\n", train_words);
    file_size = ftell(fin);
    printf("file size is %d\n", file_size);
    fclose(fin);
}

void SaveVocab() {
    FILE *fo = fopen(save_vocab_file, "wb");
    for (int i = 0; i < vocab_size; i++)
        fprintf(fo, "%s %d\n", vocab[i].word, vocab[i].cn);
    fclose(fo);
}

void ReadVocab() {
    char word[MAX_STRING];
    FILE *fin = fopen(read_vocab_file, "rb");
    if (fin == NULL) {
        printf("Vocabulary file not found\n");
        exit(1);
    }
    memset(vocab_hash, -1, vocab_hash_size * sizeof(int));

    char c;
    vocab_size = 0;
    while (1) {
        ReadWord(word, fin);
        if (feof(fin))
            break;
        int i = AddWordToVocab(word);
        fscanf(fin, "%d%c", &vocab[i].cn, &c);
    }
    SortVocab();
    printf("Vocab size: %d\n", vocab_size);
    printf("Words in train file: %lld\n", train_words);
    fclose(fin);

    // get file size
    FILE *fin2 = fopen(train_file, "rb");
    if (fin2 == NULL) {
        printf("ERROR: training data file not found!\n");
        exit(1);
    }
    fseek(fin2, 0, SEEK_END);
    file_size = ftell(fin2);
    fclose(fin2);
}

void InitNet() {
    M_in = (real *) malloc(vocab_size * hidden_size * sizeof(real));
    M_out = (real *) malloc(vocab_size * hidden_size * sizeof(real));
    if (!M_in || !M_out) {
        printf("Memory allocation failed\n");
        exit(1);
    }

    #pragma omp parallel for num_threads(num_threads) schedule(dynamic)
    for (int i = 0; i < vocab_size; i++) {
        memset(M_in + i * hidden_size, 0.f, hidden_size * sizeof(real));
        memset(M_out + i * hidden_size, 0.f, hidden_size * sizeof(real));
    }

    // initialization
    ulonglong next_random = 1;
    for (int i = 0; i < vocab_size * hidden_size; i++) {
        next_random = next_random * (ulonglong) 25214903917 + 11;
        M_in[i] = (((next_random & 0xFFFF) / 65536.f) - 0.5f) / hidden_size;
    }
}

ulonglong loadStream(FILE *fin, int *stream, const ulonglong total_words) {
    ulonglong word_count = 0;
    while (!feof(fin) && word_count < total_words) {
        int w = ReadWordIndex(fin);
        if (w == -1)
            continue;
        stream[word_count] = w;
        word_count++;
    }
    stream[word_count] = 0; // set the last word as "</s>"
    return word_count;
}


// key function used for traing neural network
void Train_SGNS() {

    if (read_vocab_file[0] != 0) {
        ReadVocab();
    }
    else {
        LearnVocabFromTrainFile();
    }
    if (save_vocab_file[0] != 0) SaveVocab();

    InitNet();
    InitUnigramTable();

    real starting_alpha = alpha;
    ulonglong word_count_actual = 0;
    double start = 0;

    // start parallezation
    #pragma omp parallel num_threads(num_threads)
    {
        int id = omp_get_thread_num();
        int local_iter = iter;
        ulonglong  next_random = id;
        ulonglong word_count = 0, last_word_count = 0;
        int sentence_length = 0, sentence_position = 0;
        int sen[MAX_SENTENCE_LENGTH];

        // load stream and assign different portion of file to each thread
        FILE *fin = fopen(train_file, "rb");
        fseek(fin, file_size * id / num_threads, SEEK_SET);

        // define the workload for each thread
        ulonglong local_train_words = train_words / num_threads + (train_words % num_threads > 0 ? 1 : 0);
        int *stream;
        int w;

        stream = (int *) malloc((local_train_words + 1) * sizeof(int));
        local_train_words = loadStream(fin, stream, local_train_words);
        fclose(fin);

        // temporary memory to store intermediate value in mini-batch
        real * inputM = (real *) malloc(batch_size * hidden_size * sizeof(real));
        real * outputM = (real *) malloc((1 + negative) * hidden_size * sizeof(real));
        real * temp = (real *) malloc((1 + negative) * batch_size * sizeof(real));

        // inputs store the input words
        int inputs[2 * window + 1];

        // outputs store the target word and negative samples
        int* outputs_id = (int *)malloc((1 + negative) * sizeof(int));
        int outputs_len = 0;

        #pragma omp barrier

        if (id == 0)
        {
            start = omp_get_wtime();
        }

        while (1) {

            // Every 10000 words update the learning progress
            if (word_count - last_word_count > 10000) {
                ulonglong diff = word_count - last_word_count;
                #pragma omp atomic
                word_count_actual += diff;

                last_word_count = word_count;
                double now = omp_get_wtime();
                printf("%cAlpha: %f  Progress: %.2f%%  Words/sec: %.2fk", 13, alpha,
                            word_count_actual / (real) (iter * train_words + 1) * 100,
                            word_count_actual / ((now - start) * 1000));
                fflush(stdout);     
                
                //update learning rate during training but avoid too small alpha
                alpha = starting_alpha * (1 - word_count_actual / (real) (iter * train_words + 1));
                if (alpha < starting_alpha * 0.0001f)
                    alpha = starting_alpha * 0.0001f;
            }

            // get the sentece from file
            if (sentence_length == 0) {
                while (1) {
                    
                    w = stream[word_count];
                    
                    word_count++;
                    if (w == 0) break;
                    // The subsampling randomly discards frequent words while keeping the ranking same
                    if (sample > 0) {
                        real ratio = (sample * train_words) / vocab[w].cn;
                        real ran = sqrtf(ratio) + ratio;
                        next_random = next_random * (ulonglong) 25214903917 + 11;
                        if (ran < (next_random & 0xFFFF) / 65536.f)
                            continue;
                    }
                    sen[sentence_length] = w;
                    sentence_length++;
                    if (sentence_length >= MAX_SENTENCE_LENGTH) break;
                }
                sentence_position = 0;
            }

            // handle end condtion
            if (word_count > local_train_words) {
                ulonglong diff = word_count - last_word_count;
                #pragma omp atomic
                word_count_actual += diff;

                local_iter--;
                if (local_iter == 0) break;
                word_count = 0;
                last_word_count = 0;
                sentence_length = 0;
                
                continue;
            }


            int target = sen[sentence_position];
            outputs_id[0] = target;

            // get all input contexts around the target word
            next_random = next_random * (ulonglong) 25214903917 + 11;
            int b = next_random % window;

            int num_inputs = 0;
            for (int i = b; i < 2 * window + 1 - b; i++) {
                if (i != window) {
                    int c = sentence_position - window + i;
                    if (c < 0)
                        continue;
                    if (c >= sentence_length)
                        break;
                    inputs[num_inputs] = sen[c];
                    num_inputs++;
                }
            }

            int num_batches = num_inputs / batch_size + ((num_inputs % batch_size > 0) ? 1 : 0);

            // start mini-batches
            for (int b = 0; b < num_batches; b++) {

                // generate negative samples for output layer
                int offset = 1;
                for (int k = 0; k < negative; k++) {
                    next_random = next_random * (ulonglong) 25214903917 + 11;
                    int sample = table[(next_random >> 16) % table_size];
                    if (!sample)
                        sample = next_random % (vocab_size - 1) + 1;
                    int* p = find(outputs_id, outputs_id + offset, sample);
                    outputs_id[offset] = sample;
                    offset++;
                }

                outputs_len = offset;


                // fetch input sub model
                int input_start = b * batch_size;
                int input_size  = min(batch_size, num_inputs - input_start);
                for (int i = 0; i < input_size; i++) {
                    memcpy(inputM + i * hidden_size, M_in + inputs[input_start + i] * hidden_size, hidden_size * sizeof(real));
                }
                // fetch output sub model
                int output_size = outputs_len;
                for (int i = 0; i < output_size; i++) {
                    memcpy(outputM + i * hidden_size, M_out + outputs_id[i] * hidden_size, hidden_size * sizeof(real));
                }

                //training using gradient descent. Store the gradient in temp

                for (int i = 0; i < output_size; i++) {
                    for (int j = 0; j < input_size; j++) {
                        real f = 0.f, g, e;
                        for (int k = 0; k < hidden_size; k++) {
                            f += outputM[i * hidden_size + k] * inputM[j * hidden_size + k];
                        }
                        int label = (i ? 0 : 1);
                        if (f > MAX_EXP)
                            g = (label - 1) * alpha;
                        else if (f < -MAX_EXP)
                            g = label * alpha;
                        else
                            g = (label - expTable[(int) ((f + MAX_EXP) * EXP_RESOLUTION)]) * alpha;
                        
                        temp[i * input_size + j] = g;
                    }
                }

                // update input matrix
                for (int i = 0; i < output_size; i++) {
                    int s = i * hidden_size;
                    int d = outputs_id[i] * hidden_size;

                    for (int j = 0; j < hidden_size; j++) {
                        real f = 0.f;
                        for (int k = 0; k < input_size; k++) {
                            f += temp[i * input_size + k] * inputM[k * hidden_size + j];
                        }

                        M_out[d + j] += f;

                    }
                }

                // update ooutput matrix
                for (int i = 0; i < input_size; i++) {
                    int s = i * hidden_size;
                    int d = inputs[input_start + i] * hidden_size;

                    for (int j = 0; j < hidden_size; j++) {
                        real f = 0.f;

                        for (int k = 0; k < output_size; k++) {
                            f += temp[k * input_size + i] * outputM[k * hidden_size + j];
                        }

                        M_in[d + j] += f;

                    }
                }

            }

            sentence_position++;
            if (sentence_position >= sentence_length) {
                sentence_length = 0;
            }
        }
        free(inputM);
        free(outputM);
        free(temp);
        
        free(stream);
        
    }
    
}

void saveModel() {
    // save the model
    FILE *fo = fopen(output_file, "wb");
    // Save the word vectors
    fprintf(fo, "%d %d\n", vocab_size, hidden_size);
    for (int a = 0; a < vocab_size; a++) {
        fprintf(fo, "%s ", vocab[a].word);
        for (int b = 0; b < hidden_size; b++)
            fwrite(&M_in[a * hidden_size + b], sizeof(real), 1, fo);
        fprintf(fo, "\n");
    }
    fclose(fo);
}

int ArgPos(char *str, int argc, char **argv) {
    for (int a = 1; a < argc; a++)
        if (!strcmp(str, argv[a])) {

            return a;
        }
    return -1;
}



int main(int argc, char** argv) {

   if (argc == 1) {
        printf("parallel word2vec (sgns) in shared memory system\n\n");
        printf("Options:\n");
        printf("Parameters for training:\n");
        printf("\t-train <file>\n");
        printf("\t\tUse text data from <file> to train the model\n");
        printf("\t-output <file>\n");
        printf("\t\tUse <file> to save the resulting word vectors\n");
        printf("\t-size <int>\n");
        printf("\t\tSet size of word vectors; default is 100\n");
        printf("\t-window <int>\n");
        printf("\t\tSet max skip length between words; default is 5\n");
        printf("\t-sample <float>\n");
        printf("\t\tSet threshold for occurrence of words. Those that appear with higher frequency in the training data\n");
        printf("\t\twill be randomly down-sampled; default is 1e-3, useful range is (0, 1e-5)\n");
        printf("\t-negative <int>\n");
        printf("\t\tNumber of negative examples; default is 5, common values are 3 - 10 (0 = not used)\n");
        printf("\t-threads <int>\n");
        printf("\t\tUse <int> threads (default 12)\n");
        printf("\t-iter <int>\n");
        printf("\t\tNumber of training iterations (default 5)\n");
        printf("\t-min-count <int>\n");
        printf("\t\tThis will discard words that appear less than <int> times; default is 5\n");
        printf("\t-alpha <float>\n");
        printf("\t\tSet the starting learning rate; default is 0.025\n");
        printf("\t-save-vocab <file>\n");
        printf("\t\tThe vocabulary will be saved to <file>\n");
        printf("\t-read-vocab <file>\n");
        printf("\t\tThe vocabulary will be read from <file>, not constructed from the training data\n");
        printf("\t-batch-size <int>\n");
        printf("\t\tThe batch size used for mini-batch training; default is 11 (i.e., 2 * window + 1)\n");
        printf("\nExamples:\n");
        printf("./word2vec -train data.txt -output vec.txt -size 200 -window 5 -sample 1e-4 -negative 5  -iter 3\n\n");
        return 0;
    }

    save_vocab_file[0] = 0;
    read_vocab_file[0] = 0;



    int i;
    if ((i = ArgPos((char *) "-size", argc, argv)) > 0)
        hidden_size = atoi(argv[i + 1]);
    if ((i = ArgPos((char *) "-train", argc, argv)) > 0)
        strcpy(train_file, argv[i + 1]);
    if ((i = ArgPos((char *) "-save-vocab", argc, argv)) > 0)
        strcpy(save_vocab_file, argv[i + 1]);
    if ((i = ArgPos((char *) "-read-vocab", argc, argv)) > 0)
        strcpy(read_vocab_file, argv[i + 1]);
    if ((i = ArgPos((char *) "-alpha", argc, argv)) > 0)
        alpha = atof(argv[i + 1]);
    if ((i = ArgPos((char *) "-output", argc, argv)) > 0)
        strcpy(output_file, argv[i + 1]);
    if ((i = ArgPos((char *) "-window", argc, argv)) > 0)
        window = atoi(argv[i + 1]);
    if ((i = ArgPos((char *) "-sample", argc, argv)) > 0)
        sample = atof(argv[i + 1]);
    if ((i = ArgPos((char *) "-negative", argc, argv)) > 0)
        negative = atoi(argv[i + 1]);
    if ((i = ArgPos((char *) "-threads", argc, argv)) > 0)
        num_threads = atoi(argv[i + 1]);
    if ((i = ArgPos((char *) "-iter", argc, argv)) > 0)
        iter = atoi(argv[i + 1]);
    if ((i = ArgPos((char *) "-min-count", argc, argv)) > 0)
        min_count = atoi(argv[i + 1]);
    if ((i = ArgPos((char *) "-batch-size", argc, argv)) > 0)
        batch_size = atoi(argv[i + 1]);



  vocab = (struct vocab_word *) calloc(vocab_max_size, sizeof(struct vocab_word));
  vocab_hash = (int *) malloc(vocab_hash_size * sizeof(int));
  expTable = (real *) malloc((EXP_TABLE_SIZE + 1) * sizeof(real));
  for (int i = 0; i < EXP_TABLE_SIZE + 1; i++) {
        expTable[i] = exp((i / (real) EXP_TABLE_SIZE * 2 - 1) * MAX_EXP); // Precompute the exp() table
        expTable[i] = expTable[i] / (expTable[i] + 1);                    // Precompute f(x) = x / (x + 1)
    }

  printf("number of threads: %d\n", num_threads);
  printf("number of iterations: %d\n", iter);
  printf("hidden size: %d\n", hidden_size);
  printf("number of negative samples: %d\n", negative); 
  printf("window size: %d\n", window);
  printf("batch size: %d\n", batch_size);
  printf("starting learning rate: %.5f\n", alpha);
  printf("starting training using file: %s\n\n", train_file);
  
  Train_SGNS();
  saveModel();

  return 0;
}
