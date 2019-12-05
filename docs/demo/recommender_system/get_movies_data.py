import os
import sys
import datetime
import time
import getopt
from random import choice
import paddle
import paddle.fluid as fluid
import paddle.fluid.layers as layers
import paddle.fluid.nets as nets

FILE_IN = './'
FILE_OUT = './'

def main():
    try:
        opts, args = getopt.getopt(
            sys.argv[1:],
            "f:",
            ["file="],
        )
    except getopt.GetoptError:
        print("Usage: test.py -f filenames")
        sys.exit(2)

    for opt_name, opt_value in opts:
        if opt_name in ("-f", "--file"):
            name = opt_value

            data_in = FILE_IN + name
            data_out = FILE_OUT + 'movies_data.txt'
            with open(data_out, 'w+') as f:
                for line in open(data_in, 'r'):
                    line = line.strip('\n')
                    data = line.split('::')

                    m_title = data[1].split()
                    m_categories = data[2].split('|')
                    for i in range(len(m_title)-1):
                        title = str(m_title[i].lower())
                        m_title[i] = paddle.dataset.movielens.get_movie_title_dict()[title]

                    for i in range(len(m_categories)):
                        categories = str(m_categories[i])
                        m_categories[i] = paddle.dataset.movielens.movie_categories()[categories]
                    out = str(data[0]) + '::' + ','.join('%s' %id for id in m_title) + '::' + ','.join('%s' %id for id in m_categories)

                    # print(out)
                    f.write(out + '\n')


if __name__ == '__main__':
    print('Start.')
    main()
    print('Finsh.')
