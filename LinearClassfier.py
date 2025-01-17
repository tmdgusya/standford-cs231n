import numpy as np

from Dataloader import load


class LinearClassifier:

    def __init__(self, input_dim, output_dim, learning_rate=0.001):
        self.W = np.random.randn(output_dim, input_dim) * np.sqrt(2.0 / input_dim)
        self.b = np.zeros((output_dim, 1))  # 바이어스 초기화
        self.learning_rate = learning_rate

    def loss(self, images, labels):
        scores = np.dot(self.W, images) + self.b
        scores = np.clip(scores, -1e10, 1e10)  # Prevent overflow
        loss = np.sum((labels - scores) ** 2 / images.shape[1])

        dOut = -2 * (labels - scores) / images.shape[1]
        dW = np.dot(dOut, images.T)
        db = np.sum(dOut, axis=1, keepdims=True)
        return loss, dW, db

    def update_parameters(self, dW, dB):
        self.W -= self.learning_rate * dW
        self.b -= self.learning_rate * dB

    def train(self, images, labels, epochs=1000):

        for epoch in range(epochs):
            loss, dW, db = self.loss(images, labels)
            self.update_parameters(dW, db)

            if epoch % 100 == 0:
                print(f"Epoch: {epoch}, Loss: {loss}")

    def predict(self, images):
        labels = np.dot(self.W, images) + self.b
        return np.argmax(labels, axis=0)

    def evaluate(self, images, labels):
        predictions = self.predict(images)
        return np.mean(predictions == labels)

if __name__ == '__main__':
    test_data = load('cifar-10-batches-py/test_batch')
    valid_data = load('cifar-10-batches-py/data_batch_1')

    train_Y = np.array(test_data[b'labels'])
    train_X = np.array(test_data[b'data']).reshape(-1, 32 * 32 * 3).T / 255.0
    train_Y_one_hot = np.zeros((10, train_Y.size))
    train_Y_one_hot[train_Y, np.arange(train_Y.size)] = 1

    valid_X = np.array(valid_data[b'data'])[0:1000].reshape(-1, 32 * 32 * 3).T / 255.0
    valid_Y = np.array(valid_data[b'labels'])[0:1000]
    valid_Y_one_hot = np.zeros((10, valid_Y.size))
    valid_Y_one_hot[valid_Y, np.arange(valid_Y.size)] = 1

    model = LinearClassifier(input_dim=32 * 32 * 3, output_dim=10, learning_rate=0.001)
    model.train(train_X, train_Y_one_hot, epochs=100000)

    print(model.evaluate(valid_X, valid_Y_one_hot))