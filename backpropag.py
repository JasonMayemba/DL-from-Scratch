from graphviz import Digraph
import os

class Backpropag:
    def __init__(self, _children=(), _op='', data, label=''):
        self.data = data
        self._prev = set(_children)
        self._op = _op
        self.label = label
        self.grad = 0.0  # Initialize gradient to zero
        self_backward = lambda: None  # Placeholder for backward function

    def __repr__(self):
        return f"Backpropag(data={self.data}, label={self.label})"

    def __add__(self, other):
        other = other if isinstance(other, Backpropag) else Backpropag(data=other)
        out = Backpropag(_children=(self, other), _op='+', data=self.data + other.data)

        def backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward

        return out 
