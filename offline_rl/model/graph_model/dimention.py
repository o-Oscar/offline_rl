# from offline_rl.model.graph_model.base import

unique_dimention_nb = 0


class Dimention:
    def __init__(self, name, value):
        self.name = name
        self.value = value

    def __eq__(self, other):
        return self.name == other.name and self.value == other.value

    def __add__(self, other):
        return AddDimention(self, other)

    def __mul__(self, other):
        return MulDimention(self, other)


class AddDimention(Dimention):
    def __init__(self, first: Dimention, second: Dimention):
        self.first = first
        self.second = second

    @property
    def name(self):
        return self.first.name + "+" + self.second.name

    @property
    def value(self):
        return self.first.value + self.second.value


class MulDimention(Dimention):
    def __init__(self, first: Dimention, second: Dimention):
        self.first = first
        self.second = second

    @property
    def name(self):
        return self.first.name + "*" + self.second.name

    @property
    def value(self):
        return self.first.value * self.second.value
