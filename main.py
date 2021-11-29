import mnist

from MLP import  MLPv2
from multiprocessing import Pool

def test(num_inputs, hidden_layers,learnig_rate, epoches, weight_loc, weight_scale, train_x, train_y, batch_num, test_x, test_y):
    results = []
    mlp = MLPv2(num_inputs, hidden_layers, learnig_rate, epoches, weight_loc, weight_scale)
    results.append(mlp.train(train_x, train_y, batch_num))
    print(mlp.validate(test_x, test_y))

if __name__ == "__main__":


    a = mnist.train_images()
    b = mnist.test_images()[:100]
    train_x = []
    test_x = []
    for el1, el2 in zip(a, b):
        train_x.append(el1.flatten())
        test_x.append(el2.flatten())

    train_y = mnist.train_labels()
    test_y = mnist.test_labels()[:100]
    with Pool(processes=10) as pool:
        multiple_results = [pool.apply_async(test(784, [100, 50], 0.01, 1000, 0, 0.5, train_x, train_y, 1, test_x, test_y)) for i in range(1)]
