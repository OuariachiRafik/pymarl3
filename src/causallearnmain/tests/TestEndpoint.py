from causallearnmain.causallearn.graph.Edge import Edge
from causallearnmain.causallearn.graph.Endpoint import Endpoint
from causallearnmain.causallearn.graph.GraphNode import GraphNode

# n1 = GraphNode('x1')
# n2 = GraphNode('x2')
#
# e = Edge(n1, n2, Endpoint.TAIL, Endpoint.ARROW)
ep1 = Endpoint(1)
ep2 = Endpoint(0)
ep3 = Endpoint(-1)
print(ep1)
