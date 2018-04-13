from weka.core.converters import Loader

from weka.core.classes import Random


class Load_Data():
    def __init__(self, data_dir):
        self.directory = data_dir
        self.data = None
        self.train = None
        self.test = None
        self.valid = None

    def return_data(self):
        loader = Loader(classname="weka.core.converters.ArffLoader")

        self.data = loader.load_file(self.directory)

        self.data.class_is_last()

        # print self.data

        return self.data


    def split_train_test_valid(self):
        try:
            self.data = self.return_data()
            total_inst = self.data.num_instances
            train_, self.test = self.data.train_test_split(80.0, Random(1))
            self.train, self.valid = train_.train_test_split(75.0, Random(1))

            print('total_inst:  ', total_inst, '| train_inst: ', self.train.num_instances,
                  '| valid_inst: ', self.valid.num_instances, '| test_inst: ', self.test.num_instances)

        except Exception:
             pass

        return self.train, self.valid, self.test