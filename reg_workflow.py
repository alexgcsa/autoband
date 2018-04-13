from common_defs import *

# loading data
# from load_data import load


# importing methods
from weka.classifiers import Classifier
import defs.ada as ada
import defs.cae as cae
import defs.cfs as cfs
import defs.ht as ht
import defs.knn as knn
import defs.ig as ig
import defs.j48 as j48
import defs.jr as jr
import defs.log as log
import defs.mlp as mlp
import defs.nb as nb
import defs.relief as relief
import defs.rf as rf
import defs.rep as rep
import defs.sgd as sgd
import defs.svm as svm

import multiprocessing
import pynisher

from random import random


# space = {
#     'featureSelection': hp.choice('fs', ('cfs', 'relief', 'ig', 'cae', None)),
#     'learner': hp.choice('l', ('meta.AdaBoostM1', 'trees.HoeffdingTree', 'trees.J48', 'rules.JRip',
#                                'bayes.NaiveBayes', 'functions.Logistic', 'functions.MultilayerPerceptron',
#                                'functions.SMO', 'trees.RandomForest', 'trees.REPTree')),
# }


def get_params(metalearning):

    # print metalearning

    prob_relief = metalearning[0]
    prob_semfs = metalearning[1]
    prob_ig = metalearning[2]
    prob_cfs = metalearning[3]
    prob_cae = metalearning[4]

    # print prob_relief, prob_semfs, prob_ig, prob_cfs, prob_cae


    space = {
        'featureSelection': hp.pchoice('fs', [(prob_cfs, 'cfs'), (prob_relief, 'relief'), (prob_ig, 'ig'),
                                              (prob_cae, 'cae'), (prob_semfs, None)]),
        'learner': hp.choice('l', ('meta.AdaBoostM1', 'trees.HoeffdingTree', 'trees.J48',
                                   'bayes.NaiveBayes', 'functions.Logistic','functions.SGD', 'functions.MultilayerPerceptron',
                                   'functions.SMO', 'trees.RandomForest', 'trees.REPTree')),
    }

    params = sample(space)

    if params['learner'] == 'meta.AdaBoostM1':
        pr = ada.get_params()
    elif params['learner'] == 'trees.HoeffdingTree':
        pr = ht.get_params()
    elif params['learner'] == 'lazy.IBk':
        pr = knn.get_params()
    elif params['learner'] == 'trees.J48':
        pr = j48.get_params()
    elif params['learner'] == 'rules.JRip':
        pr = jr.get_params()
    elif params['learner'] == 'bayes.NaiveBayes':
        pr = nb.get_params()
    elif params['learner'] == 'functions.Logistic':
        pr = log.get_params()
    elif params['learner'] == 'functions.MultilayerPerceptron':
        pr = mlp.get_params()
    elif params['learner'] == 'functions.SGD':
        pr = sgd.get_params()
    elif params['learner'] == 'functions.SMO':
        pr = svm.get_params()
    elif params['learner'] == 'trees.RandomForest':
        pr = rf.get_params()
    elif params['learner'] == 'trees.REPTree':
        pr = rep.get_params()


    if params['featureSelection'] == 'cae':
        pr2 = cae.get_params()
    elif params['featureSelection'] == 'cfs':
        pr2 = cfs.get_params()
    elif params['featureSelection'] == 'relief':
        pr2 = relief.get_params()
    elif params['featureSelection'] == 'ig':
        pr2 = ig.get_params()

    if params['featureSelection'] == None:
        params = (params, pr)
    else:
        params = (params, pr, pr2)

    print(params)

    return params


def try_params(n_instances, params, train, valid, test, istest):

        pprint(params)

        wf = params[0]

        if wf['learner'] == 'meta.AdaBoostM1' and wf['featureSelection'] == None:
            wfr = ada.try_params(n_instances, params[1], train, valid, test, istest)
        elif wf['learner'] == 'trees.HoeffdingTree' and wf['featureSelection'] == None:
                wfr = ht.try_params(n_instances, params[1], train, valid, test, istest)
        elif wf['learner'] == 'lazy.IBk' and wf['featureSelection'] == None:
                wfr = knn.try_params(n_instances, params[1], train, valid, test, istest)
        elif wf['learner'] == 'trees.J48' and wf['featureSelection'] == None:
            wfr = j48.try_params(n_instances, params[1], train, valid, test, istest)
        elif wf['learner'] == 'rules.JRip' and wf['featureSelection'] == None:
            wfr = jr.try_params(n_instances, params[1], train, valid, test, istest)
        elif wf['learner'] == 'bayes.NaiveBayes' and wf['featureSelection'] == None:
            wfr = nb.try_params(n_instances, params[1], train, valid, test, istest)
        elif wf['learner'] == 'functions.Logistic' and wf['featureSelection'] == None:
            wfr = log.try_params(n_instances, params[1], train, valid, test, istest)
        elif wf['learner'] == 'functions.MultilayerPerceptron' and wf['featureSelection'] == None:
            wfr = mlp.try_params(n_instances, params[1], train, valid, test, istest)
        elif wf['learner'] == 'functions.SGD' and wf['featureSelection'] == None:
            wfr = sgd.try_params(n_instances, params[1], train, valid, test, istest)
        elif wf['learner'] == 'functions.SMO' and wf['featureSelection'] == None:
            wfr = svm.try_params(n_instances, params[1], train, valid, test, istest)
        elif wf['learner'] == 'trees.RandomForest' and wf['featureSelection'] == None:
            wfr = rf.try_params(n_instances, params[1], train, valid, test, istest)
        elif wf['learner'] == 'trees.REPTree' and wf['featureSelection'] == None:
            wfr = rep.try_params(n_instances, params[1], train, valid, test, istest)


        if wf['learner'] == 'meta.AdaBoostM1' and wf['featureSelection'] != None:
            clf = ada.get_class(params[1])
        elif wf['learner'] == 'trees.HoeffdingTree' and wf['featureSelection'] != None:
            clf = ht.get_class(params[1])
        elif wf['learner'] == 'lazy.IBk' and wf['featureSelection'] != None:
            clf = knn.get_class(params[1])
        elif wf['learner'] == 'trees.J48' and wf['featureSelection'] != None:
            clf = j48.get_class(params[1])
        elif wf['learner'] == 'rules.JRip' and wf['featureSelection'] != None:
            clf = jr.get_class(params[1])
        elif wf['learner'] == 'functions.Logistic' and wf['featureSelection'] != None:
            clf = log.get_class(params[1])
        elif wf['learner'] == 'functions.MultilayerPerceptron' and wf['featureSelection'] != None:
            clf = mlp.get_class(params[1])
        elif wf['learner'] == 'bayes.NaiveBayes' and wf['featureSelection'] != None:
            clf = nb.get_class(params[1])
        elif wf['learner'] == 'trees.RandomForest' and wf['featureSelection'] != None:
            clf = rf.get_class(params[1])
        elif wf['learner'] == 'trees.REPTree' and wf['featureSelection'] != None:
            clf = rep.get_class(params[1])
        elif wf['learner'] == 'functions.SGD' and wf['featureSelection'] != None:
            clf = sgd.get_class(params[1])
        elif wf['learner'] == 'functions.SMO' and wf['featureSelection'] != None:
            clf = svm.get_class(params[1])



        if wf['featureSelection'] == 'cae':
            wfr = cae.try_params(n_instances, params[2], clf, train, valid, test, istest)
        elif wf['featureSelection'] == 'cfs':
            wfr = cfs.try_params(n_instances, params[2], clf, train, valid, test, istest)
        elif wf['featureSelection'] == 'ig':
            wfr = ig.try_params(n_instances, params[2], clf, train, valid, test, istest)
        elif wf['featureSelection'] == 'relief':
            wfr = relief.try_params(n_instances, params[2], clf, train, valid, test, istest)

        return wfr


