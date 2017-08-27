import numpy as np
from sklearn.model_selection import KFold


class Stack(object):
    def __init__(self, train, label, test, layer_1, layer_2, nfold=3, seed=0):
        self.train = train      # train data, numpy array
        self.label = label      # train label, numpy array
        self.test = test        # test data, numpy array
        self.layer_1 = layer_1  # models in layer 1, a list of models
        self.layer_2 = layer_2  # model in layer 2, currently support single model only
        self.nfold = nfold
        self.seed = seed

        assert isinstance(train, np.ndarray) and \
               isinstance(label, np.ndarray) and \
               isinstance(test, np.ndarray), \
               'input must be numpy array'

        self.ntrain = train.shape[0]
        self.ntest = test.shape[0]

    def get_oof(self, clf):
        oof_train = np.zeros((self.ntrain, ))
        oof_test = np.zeros((self.ntest, ))
        oof_test_stack = np.empty((self.nfold, self.ntest))

        kf = KFold(n_splits=self.nfold, shuffle=True, random_state=self.seed)
        for i, (train_index, test_index) in enumerate(kf.split(self.train)):
            x_tr = self.train[train_index]
            y_tr = self.label[train_index]
            x_te = self.train[test_index]

            clf.fit(x_tr, y_tr)

            oof_train[test_index] = clf.predict(x_te)
            oof_test_stack[i, :] = clf.predict(self.test)

        oof_test[:] = oof_test_stack.mean(axis=0)
        oof_train = oof_train.reshape(-1, 1)
        oof_test = oof_test.reshape(-1, 1)
        assert oof_train.shape[0] == self.ntrain and \
               oof_test.shape[0] == self.ntest, \
               'train size mismatch for next layer'

        return oof_train, oof_test

    def predict(self):
        train_list = []
        test_list = []
        # generate data for layer 2
        for i, model in enumerate(self.layer_1):
            print("training model %d in layer 1..." % i)
            oof_train, oof_test = self.get_oof(model)
            train_list.append(oof_train)
            test_list.append(oof_test)
        train_layer_2 = np.concatenate(train_list, axis=1)
        test_layer_2 = np.concatenate(test_list, axis=1)
        # train layer 2
        print("training model in layer 2...")
        self.layer_2.fit(train_layer_2, self.label)
        return self.layer_2.predict(test_layer_2)


if __name__ == '__main__':
    # test case
    import xgboost as xgb
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn import datasets

    iris = datasets.load_iris()

    X = iris.data
    y = iris.target

    Xtr, Xv, ytr, yv = train_test_split(X, y, test_size=0.2, random_state=1992)

    class SklearnWrapper(object):
        def __init__(self, clf, seed=0, params=None):
            params['random_state'] = seed
            self.clf = clf(**params)

        def fit(self, x_train, y_train):
            self.clf.fit(x_train, y_train)

        def predict(self, x):
            return self.clf.predict(x)

    class XgbWrapper(object):
        def __init__(self, seed=0, params=None):
            self.param = params
            self.param['seed'] = seed
            self.nrounds = params.pop('nrounds', 250)

        def fit(self, x_train, y_train):
            dtrain = xgb.DMatrix(x_train, label=y_train)
            self.gbdt = xgb.train(self.param, dtrain, self.nrounds)

        def predict(self, x):
            return self.gbdt.predict(xgb.DMatrix(x))

    SEED = 1992
    et_params = {
        'n_jobs': -1,
        'n_estimators': 100,
        'max_features': 0.5,
        'max_depth': 12,
        'min_samples_leaf': 2,
    }

    rf_params = {
        'n_jobs': -1,
        'n_estimators': 100,
        'max_features': 0.2,
        'max_depth': 8,
        'min_samples_leaf': 2,
    }

    xgb_params = {
        'seed': 0,
        'colsample_bytree': 0.7,
        'silent': 1,
        'subsample': 0.7,
        'learning_rate': 0.075,
        'objective': 'multi:softmax',
        'num_class': 3,
        'max_depth': 7,
        'num_parallel_tree': 1,
        'min_child_weight': 1,
        'eval_metric': 'logloss',
        'nrounds': 350
    }

    xg = XgbWrapper(seed=SEED, params=xgb_params)
    et = SklearnWrapper(clf=ExtraTreesClassifier, seed=SEED, params=et_params)
    rf = SklearnWrapper(clf=RandomForestClassifier, seed=SEED, params=rf_params)

    stk = Stack(Xtr, ytr, Xv, [et, rf], xg)

    stk.predict()
