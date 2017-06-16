 # @Author: mario
# @Date:   2017-06-02 08:34:05
# @Last Modified by:   mario
# @Last Modified time: 2017-06-13 10:27:56

gcc -o distance distance.c -lm

if [ ! -e text8 ]; then
  wget http://mattmahoney.net/dc/text8.zip -O text8.gz
  gzip -d text8.gz -f
fi

g++ -fopenmp train.cpp -o train

./train -train text8 -output output -threads 2 -iter 5  -batch-size 4 -size 50 -window 8 -negative 5


./distance output
