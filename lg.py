import numpy as np

attrs = ['AMB_TEMP', 'CH4', 'CO', 'NMHC', 'NO', 'NO2',
         'NOx', 'O3', 'PM10', 'PM2.5', 'RAINFALL', 'RH',
         'SO2', 'THC', 'WD_HR', 'WIND_DIREC', 'WIND_SPEED', 'WS_HR']
DAYS = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])


def read_train_csv(fileName, N):

    data = np.genfromtxt(fileName, delimiter=',', skip_header=1, encoding='UTF-8')[
        :, 3:].astype(float)
    # 12 months, 20 days per month, 18 features per day, 24 hours per day
    data = data.reshape(12, -1, 18, 24)

    train_X, train_Y = get_N_hours_feat(data[0], N)

    for i in range(1, 12):
        X, Y = get_N_hours_feat(data[i], N)
        train_X = np.concatenate((train_X, X), axis=0)
        train_Y = np.concatenate((train_Y, Y), axis=0)
    return train_X, train_Y


def read_test_csv(fileName, N, isval):
    if not isval:
        test_days = DAYS-22
        cumul_days = [sum(test_days[:i]) for i in range(1, 13)]
    else:
        #test_days = 5
        cumul_days = [2*i for i in range(1, 13)]
    data = np.genfromtxt(fileName, delimiter=',', skip_header=1, encoding='UTF-8')[
        :, 3:].astype(float).reshape(-1, 18, 24)

    test_X, test_Y = get_N_hours_feat(data[:cumul_days[0]], N)

    for i in range(1, 12):
        X, Y = get_N_hours_feat(data[cumul_days[i-1]:cumul_days[i]], N)
        test_X = np.concatenate((test_X, X), axis=0)
        test_Y = np.concatenate((test_Y, Y), axis=0)

    return test_X, test_Y


def get_N_hours_feat(month_data, N):
    # month_data.shape = (num_of_date, 18, 24)
    
    # if N = 1
    data = month_data.transpose((0, 2, 1)).reshape(-1, 18)
    # data row:18 element of an hour per month,shape:(480,18)
    label = month_data.transpose((1, 0, 2)).reshape(18, -1)[9] ## feature[9]: pm2.5
    # label : one dimension , pm2.5 every hours per month,shape(480,)

    total_hours = len(label)  # 480

    feats = np.array([])
    for i in range(total_hours-N):  
        '''
        add w0, to discuss without w0, please comment this line!
        ex. np.append([1, 2, 3], [[4, 5, 6], [7, 8, 9]]) -> array([1, 2, 3, ..., 7, 8, 9])          
        '''
        cur_feat = np.append(data[i:i+N].flatten(), [1])
        feats = np.concatenate([feats, cur_feat], axis=0)
    
    # pass first N labels
    label = label[N:]
    
    # to discuss without w0, please change to N*18!
    feats = feats.reshape(-1, N*18+1)

    # feats.shape(479,19)
    # label.shape(479,)
    # each month has 479 hour, each hour has 19 features

    return feats, label


class Linear_Regression(object):

    def __init__(self):
        pass

    def train(self, X, Y):
        # TODO: the shape of W should be number of features
        W = []
        self.W = W

    def predict(self, X):
        # TODO
        pred_X = []
        return pred_X


def MSE(pred_Y, real_Y):
    # TODO: mean square error
    error = 9023093
    return error
