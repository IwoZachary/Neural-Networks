import mnist

from mlp import  MLPv2
from multiprocessing import Pool

def test(num_inputs, hidden_layers,learnig_rate, epoches, weight_loc, weight_scale, bias_loc, bias_scale, train_x, train_y, batch_num, test_x, test_y):
    results = []
    mlp = MLPv2(num_inputs, hidden_layers, learnig_rate, epoches, weight_loc, weight_scale, bias_loc, bias_scale)
    results.append(mlp.train(train_x, train_y, batch_num))
    print(mlp.validate(test_x, test_y))

if __name__ == "__main__":


    a = mnist.train_images()[:500]
    b = mnist.test_images()[:500]
    train_x = []
    test_x = []
    for el1, el2 in zip(a, b):
        train_x.append(el1.flatten())
        test_x.append(el2.flatten())

    train_y = mnist.train_labels()[:100]
    test_y = mnist.test_labels()[:100]
    with Pool(processes=1) as pool:
        multiple_results = [pool.apply_async(test(784, [100, 100], 0.01, 4, 0, 0.5, 0, 0.5, train_x, train_y, 1, test_x, test_y)) for i in range(1)]

