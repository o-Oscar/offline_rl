import json
import time
from cProfile import label
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


class Logger:
    def __init__(
        self, path: Path, name: str = None, description: str = None, save: bool = True
    ):
        self.read_only = path.exists()

        self.name = name
        self.description = description

        path.mkdir(parents=True, exist_ok=True)
        self.path = path

        self.start_time = time.time()

        self.data = {}
        self.total_data = 0

        if self.read_only:
            self.load()

        self.do_save = save

    def log(self, name, value):
        cur_time = time.time()
        if not name in self.data:
            self.new_name(name)

        self.data[name]["values"].append(value)
        self.data[name]["times"].append(cur_time)

        self.check_save()
        self.total_data += 1

    def new_name(self, name):
        self.data[name] = {"values": [], "times": []}

    def check_save(self):
        if not self.do_save:
            return
        if self.read_only:
            raise NameError("Not supposed to write on an laready saved log")
        if (self.total_data + 1) % 100 == 0:
            self.save()

    def save(self):
        with open(self.path / "data.json", "w") as f:
            to_save = {
                "name": self.name,
                "description": self.description,
                "data": self.data,
            }
            json.dump(to_save, f)

    def load(self):
        json_path = self.path / "data.json"
        if json_path.exists():
            with open(self.path / "data.json", "r") as f:
                all_data = json.load(f)
                self.name = all_data["name"]
                self.description = all_data["description"]
                self.data = all_data["data"]

    def plot_wall(self, name):
        xs = np.array(self.data[name]["times"])
        xs = xs - xs[0]
        ys = np.array(self.data[name]["values"])
        print(self.name, self.description, sep=" : ")
        plt.plot(xs, ys, label=self.name)
        plt.xlabel("wall clock time (s)")
        plt.ylabel(name)

    def plot(self, name):
        ys = np.array(self.data[name]["values"])
        print(self.name, self.description, sep=" : ")
        plt.plot(ys, label=self.name)
        plt.xlabel("iterations")
        plt.ylabel(name)
