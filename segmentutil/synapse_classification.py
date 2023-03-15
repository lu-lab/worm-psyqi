import logging
import os
import time

import joblib
import numpy as np
from sklearn import ensemble
from sklearn import neural_network
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler


class SynapseClassifier(object):
    def __init__(self, name, logger=None):
        self.name = name
        if logger is None:
            self.logger = logging.getLogger()
        else:
            self.logger = logger
        self._scaler = StandardScaler()
        self._p_features = None
        self._n_features = None
        self._train_features = None
        self._train_features_nzd = None  # normalized features
        self._train_labels = None
        self._model = None

    def init_model(self, **kwargs):
        raise NotImplementedError()

    def fit(self, X, y, sample_weight=None):
        raise NotImplementedError()

    def add_trainig_data(self, positive_data: np.ndarray, negative_data: np.ndarray):
        if self._p_features is None:
            self._p_features = positive_data
        else:
            self._p_features = np.concatenate((self._p_features, positive_data), axis=0)
        if self._n_features is None:
            self._n_features = negative_data
        else:
            self._n_features = np.concatenate((self._n_features, negative_data), axis=0)

    def normalize(self):
        self._train_features = np.concatenate((self._p_features, self._n_features), axis=0)
        self._train_labels = np.concatenate((np.ones(shape=(self._p_features.shape[0],), dtype=float),
                                             -1 * np.ones(shape=(self._n_features.shape[0],), dtype=float)),
                                            axis=0)

        # normalize features
        self._train_features_nzd = self._scaler.fit_transform(self._train_features)

    def train(self, n_process=1):
        start_time = time.time()
        self.normalize()
        sample_weight = np.ones(len(self._train_labels))
        # ver. original
        sample_weight[:self._p_features.shape[0]] = min(2, self._n_features.shape[0] / self._p_features.shape[0])
        # ver. testing
        # sample_weight[:self._p_features.shape[0]] = self._n_features.shape[0] / self._p_features.shape[0]
        self.logger.debug("Start fitting model %s on training set:\n" % self.name)
        self.init_model(n_process=n_process)
        self.fit(self._train_features_nzd, self._train_labels, sample_weight=sample_weight)
        report = classification_report(self._train_labels, self._model.predict(self._train_features_nzd))
        self.logger.info("Trained model %s on training set:\n%s" % (self.name, report))

        elapsed_time = time.time() - start_time
        str_elapsed_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
        self.logger.info('Synapse classifier (%s) training finished. %s elapsed.' % (self.name, str_elapsed_time))

    def predict(self, test_features):
        """
        classification version
        :param test_features:
        :return:
        """
        test_features_nzd = self._scaler.transform(test_features)
        prediction = self._model.predict(test_features_nzd)
        result = prediction == 1.0
        return result

    def predict_proba(self, test_features):
        """
        classification version
        :param test_features:
        :return:
        """
        test_features_nzd = self._scaler.transform(test_features)
        proba = self._model.predict_proba(test_features_nzd)
        return proba[:, 1]


class SynapseClassifier_SVM(SynapseClassifier):
    def __init__(self, name, enable_prob, logger=None):
        super().__init__(name, logger)
        self._enable_prob = enable_prob

    def fit(self, X, y, sample_weight=None):
        """
        Fit data according to the given data
        Args:
            X:
            y:
            sample_weight:

        Returns:

        """
        self._model.fit(X, y, sample_weight=sample_weight)

    def init_model(self, **kwargs):
        self._model = svm.SVC(kernel='rbf')
        self._model.probability = self._enable_prob

    def save_model(self, save_dir: str):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        joblib.dump(self._model, os.path.join(save_dir, 'sk_svm_%s.dump' % self.name))
        joblib.dump(self._scaler, os.path.join(save_dir, 'sk_scaler_%s.dump' % self.name))

    def load_model(self, load_dir: str):
        try:
            self._scaler = joblib.load(os.path.join(load_dir, 'sk_scaler_%s.dump' % self.name))
            self._model = joblib.load(os.path.join(load_dir, 'sk_svm_%s.dump' % self.name))
            self._model.probability = self._enable_prob
            return True
        except Exception as ex:
            print(ex)
            return False


class SynapseClassifier_EnsembleSVC(SynapseClassifier):
    def __init__(self, name, enable_prob, logger=None):
        super().__init__(name, logger)
        self._enable_prob = enable_prob

    def fit(self, X, y, sample_weight=None):
        """
        Fit data according to the given data
        Args:
            X:
            y:
            sample_weight:

        Returns:

        """
        self._model.fit(X, y, sample_weight=sample_weight)

    def init_model(self, **kwargs):
        self._model = ensemble.BaggingClassifier(svm.SVC(kernel='rbf'), max_samples=0.05, n_estimators=20)
        self._model.probability = self._enable_prob

    def save_model(self, save_dir: str):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        joblib.dump(self._model, os.path.join(save_dir, 'sk_svm_%s.dump' % self.name))
        joblib.dump(self._scaler, os.path.join(save_dir, 'sk_scaler_%s.dump' % self.name))

    def load_model(self, load_dir: str):
        try:
            self._scaler = joblib.load(os.path.join(load_dir, 'sk_scaler_%s.dump' % self.name))
            self._model = joblib.load(os.path.join(load_dir, 'sk_svm_%s.dump' % self.name))
            self._model.probability = self._enable_prob
            return True
        except Exception as ex:
            print(ex)
            return False


class SynapseClassifier_RF(SynapseClassifier):
    """
    Random Forest classifier with tuning the hyper-parameters by cross-validation
    """

    def __init__(self, name, logger=None):
        super().__init__(name, logger)

    def fit(self, X, y, sample_weight=None):
        """
        Fit data according to the given data
        Args:
            X:
            y:
            sample_weight:

        Returns:

        """
        self._model.fit(X, y, sample_weight=sample_weight)
        self._best_estimator = self._model.best_estimator_
        self._best_estimator.fit(X, y, sample_weight=sample_weight)

    def predict(self, test_features):
        """
        classification version
        :param test_features:
        :return:
        """
        test_features_nzd = self._scaler.transform(test_features)
        prediction = self._best_estimator.predict(test_features_nzd)
        result = prediction == 1.0
        return result

    def predict_proba(self, test_features):
        """
        classification version
        :param test_features:
        :return:
        """
        test_features_nzd = self._scaler.transform(test_features)
        proba = self._best_estimator.predict_proba(test_features_nzd)
        return proba[:, 1]

    def init_model(self, **kwargs):
        param_grid = {
            'max_depth': range(3, 13, 3),
            'max_features': ['sqrt', 'log2'],
            # 'min_samples_leaf': [3, 4, 5],
            # 'min_samples_split': [8, 10, 12],
            'n_estimators': [32, 64, 128, ]
        }

        self._model = GridSearchCV(ensemble.RandomForestClassifier(), param_grid=param_grid, cv=5,
                                    n_jobs=kwargs['n_process'], scoring='f1')
        # self._model = ensemble.RandomForestClassifier(max_depth=12)

    def save_model(self, save_dir: str):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.logger.info("Save trained model (params = %s)\n" % self._best_estimator.get_params())
        joblib.dump(self._best_estimator, os.path.join(save_dir, 'sk_rf_%s.dump' % self.name))
        # self.logger.info("Save trained model (params = %s)\n" % self._model.get_params())
        # joblib.dump(self._model, os.path.join(save_dir, 'sk_rf_%s.dump' % self.name))
        joblib.dump(self._scaler, os.path.join(save_dir, 'sk_scaler_%s.dump' % self.name))

    def load_model(self, load_dir: str):
        try:
            self._scaler = joblib.load(os.path.join(load_dir, 'sk_scaler_%s.dump' % self.name))
            self._best_estimator = joblib.load(os.path.join(load_dir, 'sk_rf_%s.dump' % self.name))
            self.logger.info("Load trained model (params = %s)\n" % self._best_estimator.get_params())
            # self._model = joblib.load(os.path.join(load_dir, 'sk_rf_%s.dump' % self.name))
            # self.logger.info("Load trained model (params = %s)\n" % self._model.get_params())
            return True
        except Exception as ex:
            print(ex)
            return False


class SynapseClassifier_AdaBoost(SynapseClassifier):
    def __init__(self, name, logger=None):
        super().__init__(name, logger)

    def fit(self, X, y, sample_weight=None):
        """
        Fit data according to the given data
        Args:
            X:
            y:
            sample_weight:

        Returns:

        """
        self._model.fit(X, y, sample_weight=sample_weight)

    def init_model(self, **kwargs):
        self._model = ensemble.AdaBoostClassifier()

    def save_model(self, save_dir: str):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        joblib.dump(self._model, os.path.join(save_dir, 'sk_ada_%s.dump' % self.name))
        joblib.dump(self._scaler, os.path.join(save_dir, 'sk_scaler_%s.dump' % self.name))

    def load_model(self, load_dir: str):
        try:
            self._scaler = joblib.load(os.path.join(load_dir, 'sk_scaler_%s.dump' % self.name))
            self._model = joblib.load(os.path.join(load_dir, 'sk_ada_%s.dump' % self.name))
            return True
        except Exception as ex:
            print(ex)
            return False


class SynapseClassifier_MLP(SynapseClassifier):
    def __init__(self, name, logger=None):
        super().__init__(name, logger)

    def fit(self, X, y, sample_weight=None):
        """
        Fit data according to the given data
        Args:
            X:
            y:
            sample_weight:

        Returns:

        """
        self._model.fit(X, y)

    def init_model(self, **kwargs):
        self._model = neural_network.MLPClassifier()

    def save_model(self, save_dir: str):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        joblib.dump(self._model, os.path.join(save_dir, 'sk_mlp_%s.dump' % self.name))
        joblib.dump(self._scaler, os.path.join(save_dir, 'sk_scaler_%s.dump' % self.name))

    def load_model(self, load_dir: str):
        try:
            self._scaler = joblib.load(os.path.join(load_dir, 'sk_scaler_%s.dump' % self.name))
            self._model = joblib.load(os.path.join(load_dir, 'sk_mlp_%s.dump' % self.name))
            return True
        except Exception as ex:
            print(ex)
            return False
