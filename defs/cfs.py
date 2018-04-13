# meta classifier
from common_defs import *

# loading data
# from load_data import load


from weka.classifiers import Classifier
from weka.attribute_selection import ASSearch, ASEvaluation, AttributeSelection


space = {
    'locallyPredictive': hp.choice('locallyPredictive', (True, False)),
    'missingSeparate': hp.choice('missingSeparate', (True, False)),
    'search': hp.choice('search', ('GreedyStepwise', 'BestFirst')),
}

def get_params():
    params = sample(space)
    return handle_integers(params)
#
def try_params(n_instances, params, base, train, valid, test, istest):
    n_instances = int(round(n_instances))
    pprint(params)

    L = list()

    if params['missingSeparate'] == True:
        L.append("-M")

    if params['locallyPredictive'] == False:
        L.append("-L")


    search = ASSearch(classname="weka.attributeSelection."+params['search'])
    evaluator = ASEvaluation(classname="weka.attributeSelection.CfsSubsetEval", options=L)

    clf = Classifier(classname="weka.classifiers.meta.AttributeSelectedClassifier")

    clf.set_property("evaluator", evaluator.jobject)
    clf.set_property("base", base.jobject)

    if istest:
        result = test_weka_classifier(clf, train, test)
    else:
        result = train_and_eval_weka_classifier(clf, train, valid, n_instances)

    return result
