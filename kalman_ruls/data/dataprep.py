import tensorflow as tf 
from sklearn.cluster import KMeans 
import pandas as pd 
import numpy as np 

class DataPrep():
    def __init__(self, main_PATH, dataset):
        """
        Inputs:
            dataset (str): can be either FD00x (x=1,2,3 or 4)
                to pick which CMAPSS dataset is loaded.
        """    
        # get data 
        train_path = main_PATH + "\\train_" + dataset + ".txt"
        test_path = main_PATH + "\\test_" + dataset + ".txt"
        testRUL_path = main_PATH + "\\RUL_" + dataset +  ".txt"

        train = pd.read_csv(train_path, parse_dates=False, delimiter=" ", decimal=".", header=None)
        test  = pd.read_csv(test_path, parse_dates=False, delimiter=" ", decimal=".", header=None)
        RUL = pd.read_csv(testRUL_path, parse_dates=False, decimal=".", header=None)

        tableNA = pd.concat([train.isnull().sum(), test.isnull().sum()], axis=1)
        tableNA.columns = ["train", "test"]

        # drop the columns that consist of missing values 
        train.drop(train.columns[[-1,-2]], axis=1, inplace=True)    
        test.drop(test.columns[[-1,-2]], axis=1, inplace=True)      

        cols = ['unit', 'cycles', 'op_setting1', 'op_setting2', 'op_setting3', 's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8'
                , 's9','s10', 's11', 's12', 's13', 's14', 's15', 's16', 's17', 's18', 's19', 's20', 's21']

        train.columns = cols 
        test.columns = cols  

        train = pd.merge(train, train.groupby('unit', as_index=False)['cycles'].max(), how='left', on='unit')
        train.rename(columns={"cycles_x": "cycles", "cycles_y": "maxcycles"}, inplace=True)
        train["TTF"] = train["maxcycles"] - train["cycles"]

        test = pd.merge(test, test.groupby('unit', as_index=False)['cycles'].max(), how='left', on='unit')
        test.rename(columns={"cycles_x": "cycles", "cycles_y": "maxcycles"}, inplace=True)

        # add to the class variables 
        self.train = train 
        self.test = test 
        self.RUL = RUL 

    def op_normalize(self, K=1): 
        """
        Normalizes the data based on operating mode clusters catogorized and found with 
        K-means clustering. 

        x_n = (x - op_mean) / op_std 

        This is done on the training data and the means and stdevs are saved and used to 
        normalize the testing data to keep consistant. 

        where: 
            x = sensor value 
            op_mean = mean of the operating mode cluster "x" is in 
            op_std = standard deviation of the operating mode cluster "x" is in 
            x_n = normalized sensor value 

        Inputs:
            K: Type = int 
                amount of clusters/operating modes for K-means clustering 
                *Note if K=1 then this is standard normalization 

        Outputs: 
            data_norm: Type, Pandas DataFrame 
                same as the input dataframe but the sensor values are 
                normalized based on the mean and standard deviation of the operating 
                class they are in 
        """
        data = self.train
        data_test = self.test

        # K-means clustering 
        op_set = [col for col in data.columns if col.startswith("op")]
        X = data[op_set].values
        self.kmeans = KMeans(n_clusters=K, n_init=10).fit(X)    # cluster using training data 

        # Append operating cluster id's to dataset 
        data_op = self.operating_modes(data)
        data_op_test = self.operating_modes(data_test)

        # operating modes needed to loop over 
        self.clusters = data_op["op_c"].max()

        # copy for later normalization
        data_n = data_op.copy()
        data_n_test = data_op_test.copy()

        # find the means and standard deviations of sensors in each opperating mode of training data 
        sensors = [col for col in data_n.columns if col.startswith("s")]
        self.means = []
        self.stds = [] 
        for c in range(0, self.clusters+1):
            sens = data_n[(data_n.op_c==c)][sensors]
            mean = sens.mean()
            std = sens.std()

            self.means.append(mean)
            self.stds.append(std)

        # use these means and stadard deviations to normalize the sensor values 
        drop_index = [[0]]
        for s in self.stds:
            if len(drop_index[0]) < len(s[s<1e-8]): # find the largest amount of sensors with a small stdev 
                drop_index = s[s<1e-8].index 

        if len(drop_index) == 1:
            if (drop_index) == [[0]]:
                None 
        else: 
            # drop sensor values with small standard deviation 
            for i in range(len(self.means)):
                self.means[i].drop(drop_index, inplace=True)
                self.stds[i].drop(drop_index, inplace=True)

            data_n = self.drop_sensors(data_n, drop_index)
            data_n_test = self.drop_sensors(data_n_test, drop_index)

            # drop sensors with 2 or less unique values 
            drop_index = self.drop_same(data_n)
            for i in range(len(self.means)):
                self.means[i].drop(drop_index, inplace=True)
                self.stds[i].drop(drop_index, inplace=True)

            data_n = self.drop_sensors(data_n, drop_index)
            data_n_test = self.drop_sensors(data_n_test, drop_index)

        # normalize the sensors for each unit in the dataset 
        self.ntrain = self.norm(data_n)
        self.ntest = self.norm(data_n_test)

    def operating_modes(self, data):
        """
        Use K-means to classify data into operating modes 
        """
        op_set = [col for col in data.columns if col.startswith("op")]
        X = data[op_set].values
        kmeans_pred = self.kmeans.predict(X)

        # append operating mode classifications to data 
        op_cluster = pd.DataFrame({"op_c": kmeans_pred})
        data_op = pd.concat([data, op_cluster], axis=1)

        return data_op

    def drop_sensors(self, data, drop_index):
        """
        Drops sensors based on index given (drop_index)
        """
        data.drop(drop_index, axis=1, inplace=True)

        return data

    def drop_same(self, data):
        """
        Returns the index to drop the sensors with only 2 or less unique values 
        in the series 
        """
        sensors = [col for col in data.columns if col.startswith("s")]
        drop_index = data[sensors].loc[:,data[sensors].nunique() < 3].columns

        return drop_index

    def norm(self, data):
        """
        normalize data based on the mean and standard deviation of the 
        operating condition it is apart of 
        """
        units = int(data["unit"].max())
        sensors = [col for col in data.columns if col.startswith("s")]
        for unit in range(1, units+1):
            for c in range(0, self.clusters+1):
                sens = data[(data.op_c==c) & (data.unit==unit)][sensors]
                sens = (sens - self.means[c]) / self.stds[c]
                data.loc[(data.op_c==c) & (data.unit==unit), sensors] = sens

        return data 

    def prep_data(self, df, T, max_rul=130):
        sensors = [col for col in df.columns if col.startswith("s")]

        xs = []
        ys = []
        ts = [] 
        for unit in range(1, max(df.unit)+1):
            rul = np.expand_dims(df[df.unit==unit].TTF.values, axis=-1)
            sen = df[df.unit==unit][sensors].values
            time = np.expand_dims(df[df.unit==unit].cycles.values, axis=-1)

            # split entire sequences into 'N' time series blocks of size T, (N, T, dim)
            sen =  np.transpose(np.lib.stride_tricks.sliding_window_view(sen, T, axis=0), axes=(0,2,1))     
            rul =  np.transpose(np.lib.stride_tricks.sliding_window_view(rul, T, axis=0), axes=(0,2,1))     
            time = np.transpose(np.lib.stride_tricks.sliding_window_view(time, T, axis=0), axes=(0,2,1))    

            # if rul > max_rul let rul = max_rul if it is below max_rul then rul = rul 
            rul = (rul > max_rul) * max_rul + (rul <= max_rul) * rul

            xs.append(sen)
            ys.append(rul)
            ts.append(time)

        x = np.concatenate(xs, axis=0)
        y = np.concatenate(ys, axis=0)
        t = np.concatenate(ts, axis=0)

        return x, y, t

    def valid_set(self, x_train, y_train, t_train, split=0.2):
        """
        Inputs: 
            x_train (tensor): training input tensor, size (*, seq, x_dim)
            y_train (tensor): training target tensor, size (*, seq, 1)
            t_train (tensor): corresponding training time points, size (*, seq, 1)
            split (float): a number between 0 and 1 to determine the % of data to be converted into a validation set 

        Outputs:
            x_train (tensor): new split training input tensor
            y_train (tensor): new split training targets 
            t_train (tensor): new split corresponding training time points 
            x_valid (tensor): validation inputs split from the original training dataset 
            y_valid (tensor): validation targets split from the original training dataset 
            t_valid (tensor): validation time points split from the original training dataset 
        """
        rng = np.random.default_rng()

        valid_size = int(split * x_train.shape[0])
        total_No_idxs = x_train.shape[0] - 1
        # get a list of random integers to serve as indicies to extract a validation dataset 
        valid_set = list(rng.choice(total_No_idxs, size=valid_size, replace=False))

        # get the remaining possible integers to serve as the indicies for the training dataset 
        train_set = list(np.linspace(0, x_train.shape[0]-1, x_train.shape[0]))
        train_set = [int(x) for x in train_set if x not in valid_set]

        # store new training and validation tensors/datasets
        x_valid = x_train[valid_set]
        y_valid = y_train[valid_set]
        t_valid = t_train[valid_set]
        x_train = x_train[train_set]
        y_train = y_train[train_set]
        t_train = t_train[train_set]

        return x_train, y_train, t_train, x_valid, y_valid, t_valid

    def prep_test(self, df, RUL, max_rul=130):
        """
        Prepares input test data used once the network is trained 

        Inputs:
            df (pandas DataFrame): the dataframe with the input sensor data for each unit 
            RUL (pandas DataFrame): the dataframe with the corresponding RUL for each unit 

        Outputs:
            x (list of tensors): the sensors in a tensor format, size (1, seq, inputs)
            y (list of tensors): the RUL values at each time point (seq)
            t (list of tensors): corresponding times (1, seq, 1)
        """
        sensors = [col for col in df.columns if col.startswith("s")]
        RUL = RUL.values

        x = [] 
        y = [] 
        t = [] 
        for unit in range(1, max(df.unit)+1):
            sen = df[df.unit==unit][sensors].values
            rul_T = RUL[unit-1]
            time = df[df.unit==unit].cycles.values

            seq = sen.shape[0]          # observed sequence length 
            total_len = rul_T + seq     # total length if run to end of life

            rul = np.linspace(total_len-1, rul_T, seq)
            rul = (rul > max_rul) * max_rul + (rul <= max_rul) * rul

            sen = tf.convert_to_tensor(sen)
            rul = tf.convert_to_tensor(rul)
            time = tf.convert_to_tensor(time)

            x.append(tf.expand_dims(sen, 0))
            y.append(tf.squeeze(rul, axis=-1))
            t.append(time[tf.newaxis,:,tf.newaxis])  

        return x, y, t      

    def get_dataloaders(self, bs, x_train, t_train, y_train, x_valid, t_valid, y_valid):
        """
        Inputs:
            bs (int): batch size of the outputs 
            x_train (tensor): training input tensor, size (*, seq, x_dim)
            y_train (tensor): training target tensor, size (*, seq, 1)
            x_valid (tensor): validation inputs split from the original training dataset, size (*, seq, x_dim)
            y_valid (tensor): validation targets split from the original training dataset, size (*, seq, 1)

        Outputs: 
            train_loader (dataloader): dataloader containing the training inputs and targets, size (bs, seq, x_dim) and (bs, seq, 1)
            valid_loader (dataloader): dataloader containing the validation inputs and targets, size (bs, seq, x_dim) and (bs, seq, 1)
        """
        train_loader = tf.data.Dataset.from_tensor_slices((np.float64(np.concatenate([x_train, t_train], axis=-1)), np.float64(y_train)))
        train_loader = train_loader.shuffle(x_train.shape[0], reshuffle_each_iteration=True).batch(bs)

        valid_loader = tf.data.Dataset.from_tensor_slices((np.float64(np.concatenate([x_valid, t_valid], axis=-1)), np.float64(y_valid)))
        valid_loader = valid_loader.batch(bs)

        return train_loader, valid_loader 
