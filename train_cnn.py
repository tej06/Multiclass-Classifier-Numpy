from CNN.network import *
from CNN.utils import *

from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
import pickle

parser = argparse.ArgumentParser(description='Parameters for CNN')

parser.add_argument('-lr', '--learning_rate', default=0.01, help='path of folder containing data')
parser.add_argument('-b', '--batch_size', default=32, help='path of folder containing data')
parser.add_argument('-e', '--epochs', default=10, help='path of folder containing data')
parser.add_argument('-dp', '--data_path', default='./data', help='path of folder containing data')
parser.add_argument('-sf', '--save_file', default='./model_params.pkl', help='path of file to save weights')

if __name__ == '__main__':

    args = parser.parse_args()
    lr = flaot(args.learning_rate)
    batch_size = int(args.batch_size)
    epochs = int(args.epochs)
    data_path = args.data_path
    save_path = args.save_file

    classes = ["broadleaf", "grass", "soil", "soybean"]
    cost = train(classes, lr=lr, batch_size=batch_size, num_epochs=epochs, data_path=data_path, save_path=save_file)

    params, cost = pickle.load(open(save_file, 'rb'))
    [f1, f2, w3, w4, b1, b2, b3, b4] = params

    # Get test data
    data_type = 'test'
    X, y = extract_data(data_path, classes)
    m = X.shape[0]
    X = np.asarray(X, np.float32)
    print(X.shape, y.shape)
    # Normalize the data
    X /= 255.0

    corr = 0
    digit_count = [0 for i in range(10)]
    digit_correct = [0 for i in range(10)]

    print()
    print("Computing accuracy over test set:")

    t = tqdm(range(len(X)), leave=True)

    for i in t:
        x = X[i]
        pred, prob = predict(x, f1, f2, w3, w4, b1, b2, b3, b4)
        digit_count[int(y[i])]+=1
        if pred==y[i]:
            corr+=1
            digit_correct[pred]+=1

        t.set_description("Acc:%0.2f%%" % (float(corr/(i+1))*100))

    print("Overall Accuracy: %.2f" % (float(corr/len(test_data)*100)))
