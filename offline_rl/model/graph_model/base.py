import torch as th
import torch.nn as nn
from offline_rl.model.graph_model.dimention import Dimention, unique_dimention_nb


class Node(nn.Module):
    def __init__(self):
        super().__init__()
        self.shape = None
        self.output = None

    @property
    def numeric_shape(self):
        return self.calc_numeric_shape(self.shape)

    def calc_numeric_shape(self, shape):
        return [dimention.value for dimention in shape]

    @property
    def named_shape(self):
        return self.calc_named_shape(self.shape)

    def calc_named_shape(self, shape):
        return [dimention.name for dimention in shape]

    def reset(self):
        raise NotImplementedError()

    def node_forward(self, **inputs):
        if self.output is None:
            self.output = self._node_forward(**inputs)
        return self.output

    def _node_forward(self):
        raise NotImplementedError()

    def forward(self, **inputs):
        self.reset()
        return self.node_forward(**inputs)


class Input(Node):
    def __init__(self, name, shape):
        super().__init__()
        self.name = name
        self.shape = shape

    def reset(self):
        pass

    def _node_forward(self, **inputs):
        if self.name not in inputs:
            raise NameError("Input {} not found in the input dict".format(self.name))
        # TODO : test if the shape is right
        return inputs[self.name]


class BlowUp(Node):
    def __init__(self, parent: Node, shape):
        super().__init__()
        self.parent = parent
        self.shape = shape

        # test if it is possible to blow up the parent shape to the output_shape
        shape_id = 0
        for dimention in self.parent.shape:
            while dimention != self.shape[shape_id]:
                shape_id += 1
                if shape_id > len(self.shape):
                    raise NameError(
                        "the shapes do not match. Parent shape {}, target shape {}".format(
                            self.parent.shape, self.shape
                        )
                    )
            shape_id += 1

    def reset(self):
        self.parent.reset()
        self.output = None

    def _node_forward(self, **inputs):
        input = self.parent(**inputs)
        return input + th.zeros(self.numeric_shape)


class Concatenate(Node):
    def __init__(self, parents: list[Node], axis):
        super().__init__()
        self.parents = parents
        self.axis = axis

        # test if it is possible to concatenate all parents
        for i, dimention in enumerate(self.parents[0].shape):
            if (
                not all([parent.shape[i] == dimention for parent in self.parents])
                and i != axis
            ):
                raise NameError(
                    "The shapes of the parents do not match : "
                    + "\n".join([str(parent.shape) for parent in self.parents])
                )

        self.shape = list(self.parents[0].shape)
        for parent in self.parents[1:]:
            self.shape[axis] = self.shape[axis] + parent.shape[axis]

    def reset(self):
        for parent in self.parents:
            parent.reset()
        self.output = None

    def _node_forward(self, **inputs):
        children_tensors = [parent(**inputs) for parent in self.parents]
        return th.concat(children_tensors, dim=self.axis)


class SelfAttention(Node):
    def __init__(self, parent: Node, batch_axes, feature_axis):
        super().__init__()
        self.parent = parent

        self.batch_axes = batch_axes
        self.feature_axes = [feature_axis]
        self.sequence_axes = list(
            set(range(len(parent.shape))) - set(batch_axes) - set(self.feature_axes)
        )
        self.all_axes = [self.batch_axes, self.sequence_axes, self.feature_axes]
        # self.axis = axis
        # self.shape = list(self.parent.shape)
        # self.shape[axis] = embed_dim
        self.first_perm = self.batch_axes + self.sequence_axes + self.feature_axes
        self.second_perm = [-1] * len(self.first_perm)
        for i, j in enumerate(self.first_perm):
            self.second_perm[j] = i

        self.attn_shape = [parent.shape[axes[0]] for axes in self.all_axes]
        for i, axes in enumerate(self.all_axes):
            for ax in axes[1:]:
                self.attn_shape[i] = self.attn_shape[i] * parent.shape[ax]

        self.to_out_shape = [self.parent.shape[i] for i in self.first_perm]
        self.shape = self.parent.shape

        self.attention = nn.MultiheadAttention(
            self.attn_shape[2].value, 4, batch_first=True
        )

    def reset(self):
        self.parent.reset()
        self.output = None

    def _node_forward(self, **inputs):
        input = self.parent(**inputs)
        input = th.permute(input, self.first_perm)
        input = input.view(self.calc_numeric_shape(self.attn_shape))

        result, _ = self.attention(input, input, input)
        result = result.view(self.calc_numeric_shape(self.to_out_shape))
        return th.permute(result, self.second_perm)


class Linear(Node):
    def __init__(self, parent: Node, axis, out_dimention):
        super().__init__()
        self.parent = parent
        self.axis = axis
        self.shape = list(self.parent.shape)
        self.shape[axis] = out_dimention

        need_reshape = axis != (len(self.shape) - 1)
        self.first_perm = list(set(range(len(self.shape))) - set([axis])) + [axis]
        self.second_perm = [-1] * len(self.first_perm)
        for i, j in enumerate(self.first_perm):
            self.second_perm[j] = i

        self.layer = nn.Linear(parent.shape[axis].value, self.shape[axis].value)

    def reset(self):
        self.parent.reset()
        self.output = None

    def _node_forward(self, **inputs):
        input = self.parent(**inputs)
        input_ = th.permute(input, self.first_perm)
        result = self.layer(input_)
        result = th.permute(result, self.second_perm)
        return result + input


class ReLU(Node):
    def __init__(self, parent: Node):
        super().__init__()
        self.parent = parent
        self.shape = self.parent.shape
        self.activation = nn.ReLU()

    def reset(self):
        self.parent.reset()
        self.output = None

    def _node_forward(self, **inputs):
        input = self.parent(**inputs)
        return self.activation(input)
