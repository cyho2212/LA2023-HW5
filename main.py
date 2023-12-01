import matplotlib.pyplot as plt
from lg import *


def plot_N_loss():

    all_train_losses = []
    all_val_losses = []
    N = 0 

    '''
    TODO:
        1. Plot  N vs MSEloss (train/validation) in one picture
        2. You need to use function run_train_test(N, f1, f2)
        3. Return N value that yields the minimum loss on the validation set 
    '''

    ### no need to modify code below
    ### all_train_losses and all_val_losses have length 30
            
    x = [i for i in range(1, 31)]
    plt.plot(x, all_train_losses, 'b', label='train')
    plt.plot(x, all_val_losses, 'r', label='val')
    plt.legend()
    plt.xlabel('N')
    plt.ylabel('MSE loss')
    plt.savefig('result.png')
    return N


def run_train_test(N, f1, f2):

    '''
    Args:
        N: last N hour
        f1: training set file (ex. ./data/train.csv)
        f2: validation set or testing set (ex. ./data/validation.csv)
    '''
    train_X, train_Y = read_train_csv(f1, N)
    model = Linear_Regression()
    model.train(train_X, train_Y)
    pred_train_Y = model.predict(train_X)
    train_loss = MSE(pred_train_Y, train_Y)

    test_X, test_Y = read_test_csv(f2, N, "val" in f2)
    pred_test_Y = model.predict(test_X)
    test_loss = MSE(pred_test_Y, test_Y)
    print(train_loss, test_loss)
    return train_loss, test_loss


if __name__ == '__main__':

    ## N yields the minimum loss on the validation set
    N = plot_N_loss()

    ## predict test set, then output ans.txt
    assert N != 0
    train_loss, test_loss = run_train_test(
        N, "data/train.csv", "data/test.csv")
    ans = open('ans.txt', mode='w')
    line = str(N) + '\n' + str(test_loss)
    ans.write(line)
    ans.close()
