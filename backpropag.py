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