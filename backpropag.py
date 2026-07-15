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
        out._backward = backward

        return out

    def backward(self):

        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        self.grad = 1.0 # Initialize the gradient of the output node to 1.0 (always)
        for node in reversed(topo):
            node._backward()  # Call the backward function for each node

    def trace(root):
        nodes, edges = set(), set()
        def build(v):
            if v not in nodes:
                nodes.add(v)
                for child in v._prev:
                    edges.add((child, v))
                    build(child)
        build(root)
        return nodes, edges

    def draw_dot(root, filename='backpropagation_graph'):
        dot = Digraph(format='png', graph_attr={'rankdir': 'LR'})
        nodes, edges = Backpropag.trace(root)

        for n in nodes:
            uid = str(id(n))
            dot.node(name=uid, label=f"{n.label}\ndata={n.data:.4f}\ngrad={n.grad:.4f}", shape='record')

            if n._op:
                dot.node(name=uid + n._op, label=n._op)
                dot.edge(uid + n._op, uid)

        for n1, n2 in edges:
            dot.edge(str(id(n1)), str(id(n2)))

        dot.render(filename, cleanup=True)

if __name__ == "__main__":
    # 1. Définir des variables d'entrée (poids w, entrée x, biais b)
    x = Backpropag(2.0, label='x')
    w = Backpropag(-3.0, label='w')
    b = Backpropag(6.7, label='b')

    # 2. Effectuer les calculs (Passe Avant)
    wx = w * x; wx.label = 'w*x'
    y = wx + b; y.label = 'y (sortie)'
    
    # 3. Lancer la rétropropagation (Passe Arrière)
    y.backward()
    
    # 4. Générer le graphe visuel
    graphe = Backpropag.draw_dot(y)
    
    # Sauvegarde le graphe dans un fichier "graphe_retroprop.svg" et l'ouvre
    
    print("Le graphe de calcul a été généré avec succès !")
