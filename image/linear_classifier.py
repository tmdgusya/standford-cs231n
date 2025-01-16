
class LinearClassifier():
    """
    f(x, W) = Wx + b

    X size will be 3072*1
    W size will be 10*3072
    b size will be 10

    f(x, W) size will be 10*1

    """

    def __init__(self):
        self.W = None
        self.b = None