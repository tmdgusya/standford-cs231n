from collections import Counter

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
        image = image[np.newaxis, :, :, :]
        return np.sum(np.abs(self.X - image), axis=(1, 2, 3))

    def k_nearest_neighbors(self, image, K):
        distances = self.getL1distance(image)
        """
        arr = np.array([40, 10, 20, 30])
        sorted_indices = np.argsort(arr)
        print(sorted_indices)  # 출력: [1, 2, 3, 0]
        """
        nearest_indices = np.argsort(distances)[:K]  # Get indices of K smallest distances
        nearest_labels = [self.Y[i] for i in nearest_indices]  # Get the labels of the nearest neighbors
        most_common = Counter(nearest_labels).most_common(1)[0][0]  # Find the most common label
        return most_common

    def predict_(self, test_image):
        label_index = np.argmin(self.getL1distance(test_image))
        return self.Y[label_index]

    def predict_2(self, test_image):
        label_index = self.k_nearest_neighbors(test_image, 7)
        return label_index

    def predict(self, test_images):
        """
        Predict the label of the most similar training image

        Time-complexity: O(n)
        :param test_images:
        :return:
        """
        return [self.predict_2(image) for image in test_images]

    def check_accuracy(self, valid_images, valid_labels):
        predictions = self.predict(valid_images)
        return np.mean(np.equal(predictions, valid_labels))



if __name__ == "__main__":
    test_data = load('cifar-10-batches-py/test_batch')
    valid_data = load('cifar-10-batches-py/data_batch_1')

    train_Y = np.array(test_data[b'labels'])
    train_X = np.array(test_data[b'data']).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)

    valid_Y = np.array(valid_data[b'labels'])[0:1000]
    valid_X = np.array(valid_data[b'data'])[0:1000].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)

    model = NearestNeighbor()
    model.train(train_X, train_Y)
    print(f'train accuracy: {model.check_accuracy(valid_X, valid_Y)}')
