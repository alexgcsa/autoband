#!/usr/bin/env python

import weka.core.jvm as jvm

jvm.start()

import sys
import cPickle as pickle
from pprint import pprint

from hyperband import Hyperband

import os

# loading data
from load_data import Load_Data


from reg_workflow import get_params, try_params

# importing metalearning
import reg_metalearning as ml

from weka.core.classes import Random

dir = '/home/silvia/hyperband-data/'


def main(directory):

    dir_list = os.listdir(directory)
    for e in dir_list:
        file_name = directory + os.path.basename(e)

        metalearning = ml.main(os.path.basename(e))

        load = Load_Data(file_name)

        train, valid, test = load.split_train_test_valid()

        for i in range(1):

            try:
                output_file1 = sys.argv[1]
                output_file2 = sys.argv[1]
                if not output_file1.endswith('.pkl'):
                    output_file1 += '.pkl'
                    output_file2 += '.pkl'
            except IndexError:
                output_file1 = 'results_ab_' + os.path.basename(e) + '_' + str(i) + '.pkl'
                output_file2 = 'results_ab_test_' + os.path.basename(e) + '_' + str(i) + '.pkl'

                print("Will save results to", output_file1, output_file2)

            # data = load(file_name)

            hb = Hyperband(get_params, try_params, train, valid, test, metalearning)
            results = hb.run(skip_last=1)
            # print(results)
            test_results = hb.tests(results)

            print("{} total, best in validation:\n".format(len(results)))

            for r in sorted(results, key=lambda x: x['loss']):
                print("loss: {:.2} | {} seconds | {:.1f} instances | run {} ".format(
                           r['loss'], r['seconds'], r['instances'], r['counter']))
                pprint(r['params'])
                print

            print("test results")
            for r in range(len(test_results)):
                t = test_results[r]
                print("loss: {:.2%} | auc: {:.2%} | {} seconds | {} run ".format(
                    t['loss'], t['auc'], t['seconds'], t['counter']))
                pprint(t['params'])
                print

            print("results: ", results)
            print("test results:    ", test_results)
            print("saving...")

            with open(output_file1, 'wb') as f:
                       pickle.dump(results, f)

            with open(output_file2, 'wb') as f:
                       pickle.dump(test_results, f)

    return 'finished'


main(dir)


jvm.stop()