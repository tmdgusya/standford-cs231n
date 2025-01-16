from xmlrpc.client import MAXINT

import numpy as np
from Dataloader import load

"""
We'll use CIFAR-10 dataset to train and test our nearest neighbors.

10 classes
50,000 training images
10,000 testing images
"""

class NearestNeighbor:

    def __init__(self):
        self.X = None
        self.Y = None
        self.label_names = {
            0: 'airplane',
            1: 'automobile',
            2: 'bird',
            3: 'cat',
            4: 'deer',
            5: 'dog',
            6: 'frog',
            7: 'horse',
            8: 'ship',
            9: 'truck',
        }

    def train(self, images, labels):
        """
        Memorize all data and labels

        X will be 10000 * 3072
        Y will be 10000

        Time complexity: O(1)
        :param images:
        :param labels:
        :return:
        """
        self.X = images
        self.Y = labels
        return self

    def getL1distance(self, image):
        return np.sum(np.abs(self.X - image), axis=(1, 2, 3))

    def predict_(self, test_image):
        label_index = np.argmin(self.getL1distance(test_image))
        return self.Y[label_index]

    def predict(self, test_images):
        """
        Predict the label of the most similar training image

        Time-complexity: O(n)
        :param test_images:
        :return:
        """
        return [self.predict_(image) for image in test_images]

    def check_accuracy(self, valid_images, valid_labels):
        predictions = self.predict(valid_images)
        return np.mean(np.equal(predictions, valid_labels))



if __name__ == "__main__":
    test_data = load('cifar-10-batches-py/test_batch')
    valid_data = load('cifar-10-batches-py/data_batch_1')

    train_Y = np.array(test_data[b'labels'])
    train_X = np.array(test_data[b'data']).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)

    valid_Y = np.array(valid_data[b'labels'])
    valid_X = np.array(valid_data[b'data']).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)

    model = NearestNeighbor()
    model.train(train_X, train_Y)
    print(f'train accuracy: {model.check_accuracy(valid_X, valid_Y)}')
