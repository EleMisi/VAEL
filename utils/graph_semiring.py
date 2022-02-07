import torch
from problog.evaluator import Semiring


class GraphSemiring(Semiring):

    def __init__(self, batch_size=32, device=torch.device('cpu')):
        Semiring.__init__(self)
        self.eps = 1e-12
        self.batch_size = batch_size
        self.device = device

    def negate(self, a):
        """Returns the negation of the probability a: 1-a."""
        return self.one() - a

    def one(self):
        """Returns the identity element of the multiplication."""
        one = torch.ones(self.batch_size)
        return one.to(self.device)

    def zero(self):
        """Returns the identity element of the addition."""
        zero = torch.zeros(self.batch_size)
        return zero.to(self.device)

    def is_zero(self, a):
        """Tests whether the given value is the identity element of the addition up to a small constant"""
        return ((a >= -self.eps) & (a <= self.eps)).any()

    def is_one(self, a):
        """Tests whether the given value is the identity element of the multiplication up to a small constant"""
        return ((a >= 1.0 - self.eps) & (a <= 1.0 + self.eps)).any()

    def plus(self, a, b):
        """Computes the addition of the given values."""
        if self.is_zero(b):
            return a
        if self.is_zero(a):
            return b
        return a + b

    def times(self, a, b):
        """Computes the multiplication of the given values."""
        if self.is_one(b):
            return a
        if self.is_one(a):
            return b
        return a * b

    def set_weights(self, weights):
        self.weights = weights

    def normalize(self, a, z):
        return a / z

    def value(self, a):
        """Transform the given external value into an internal value."""
        v = self.weights.get(a, a)
        return v
