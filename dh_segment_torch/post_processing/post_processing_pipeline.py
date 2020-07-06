from typing import List, Any, Dict, Union

import networkx as nx
import numpy as np

from dh_segment_torch.config.registrable import Registrable
from dh_segment_torch.post_processing.operation import Operation


class PostProcessingPipeline(Registrable):
    default_implementation = "default"

    # TODO intermediary result store to be used later in the pipeline ?

    def __init__(self, operations: List[Operation]):
        self.operations = operations

    def apply(self, probabilities: np.array) -> Any:
        result = probabilities
        for operation in self.operations:
            result = operation(result)
        return result


PostProcessingPipeline.register("default")(PostProcessingPipeline)

operation_type = Union[Operation, List[Union[Operation, str]], str]


class OperationsInputs(Registrable):
    default_implementation = "default"

    def __init__(
        self, inputs: Union[str, List[str]], ops: Union[Operation, List[Operation]]
    ):
        if isinstance(inputs, str):
            inputs = [inputs]
        self.inputs = inputs

        if isinstance(ops, Operation):
            ops = [ops]
        self.ops = ops


OperationsInputs.register("default")(OperationsInputs)


@PostProcessingPipeline.register("dag")
class DagPipeline(PostProcessingPipeline):
    def __init__(
        self, operations: Dict[str, OperationsInputs],
    ):
        self.operations = operations

        self.dag = build_dag_from_operations(self.operations)
        self.inputs_names = set(
            [name for name, in_degree in list(self.dag.in_degree()) if in_degree == 0]
        )

    def apply(self, *args, **kwargs):
        inputs = self.parse_and_validate_input(*args, **kwargs)

        results = {}
        results.update(inputs)
        for input_name in nx.topological_sort(self.dag):
            if input_name in self.inputs_names:
                continue
            inputs = self.operations[input_name].inputs
            inputs = [results[input_] for input_ in inputs]
            input_ = inputs[0]
            args = inputs[1:]
            operations = self.operations[input_name].ops
            result = input_
            for operation in operations:
                result = operation(result, *args)
            results[input_name] = result
        return results

    def parse_and_validate_input(self, *args, **kwargs):
        inputs = {}
        for name, arg in zip(self.inputs_names, args):
            inputs[name] = arg
        for name, arg in kwargs.items():
            inputs[name] = arg
        input_received = set(inputs.keys())
        input_expected = self.inputs_names
        union = input_received.union(input_expected)
        intersection = input_received.intersection(input_expected)
        not_matching = union.difference(intersection)
        if len(not_matching) > 0:
            raise ValueError(f"Got unexpected or missing arguments: {not_matching}")
        return inputs


def build_dag_from_operations(operations: Dict[str, OperationsInputs],) -> nx.DiGraph:

    nodes = set()
    edges = []
    for key, operations_inputs in operations.items():
        nodes.add(key)
        for input_ in operations_inputs.inputs:
            nodes.add(input_)
            edges.append((input_, key))

    graph = nx.DiGraph()

    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)

    if not nx.is_directed_acyclic_graph(graph):
        raise ValueError(f"Graph from operations {operations} was not a DAG.")

    return graph
