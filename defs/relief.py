# meta classifier
from common_defs import *

# loading data
# from load_data import load


from weka.classifiers import Classifier
from weka.attribute_selection import ASSearch, ASEvaluation, AttributeSelection


space = {
    'weightByDistance': hp.choice('weightByDistance', (True, False)),
    'sampleSize': hp.quniform('sampleSize', -1, 10, 1),
    'numNeighbours': hp.quniform('numNeighbours', 1, 10, 1),
    'sigma': hp.quniform('sigma', 1, 10, 1),
}


def get_params():
    params = sample(space)
    return handle_integers(params)


def try_params(n_instances, params, base, train, valid, test, istest):

    n_instances = int(round(n_instances))
    # print "n_instances:", n_instances
    pprint(params)

    L = list([])

    if params['weightByDistance'] == True:
        L.append("-W")

    L.append("-M")
    L.append(str(params['sampleSize']))

    L.append("-K")
    L.append(str(params['numNeighbours']))

    L.append("-A")
    L.append(str(params['sigma']))

    # print L

    search = ASSearch(classname="weka.attributeSelection.Ranker")
    evaluator = ASEvaluation(classname="weka.attributeSelection.ReliefFAttributeEval", options=L)

    clf = Classifier(classname="weka.classifiers.meta.AttributeSelectedClassifier")

    clf.set_property("evaluator", evaluator.jobject)
    clf.set_property("search", search.jobject)
    clf.set_property("base", base.jobject)

    if istest:
        result = test_weka_classifier(clf, train, test)
    else:
        result = train_and_eval_weka_classifier(clf, train, valid, n_instances)

    return result