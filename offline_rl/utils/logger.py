import json
import time
from cProfile import label
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


class Logger:
    def __init__(
        self,
        path: Path,
        name: str = None,
        description: str = None,
        save: bool = True,
        save_interval: float = 20,
    ):
        self.read_only = path.exists()

        self.name = name
        self.description = description

        path.mkdir(parents=True, exist_ok=True)
        self.path = path

        self.start_time = time.time()

        self.data = {}
        self.total_data = 0
        self.data_file_nb = 0

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
        data_file_name = "data{}.json".format(self.data_file_nb)
        with open(self.path / data_file_name, "w") as f:
            json.dump(self.data, f)
        for key in self.data.keys():
            self.new_name(key)

        self.data_file_nb += 1
        with open(self.path / "metadata.json", "w") as f:
            to_save = {
                "name": self.name,
                "description": self.description,
                "data_file_nb": self.data_file_nb,
                "data_keys": list(self.data.keys()),
            }
            json.dump(to_save, f)

    def load(self):
        json_path = self.path / "metadata.json"
        if json_path.exists():
            with open(json_path, "r") as f:
                all_data = json.load(f)
                self.name = all_data["name"]
                self.description = all_data["description"]
                self.data_file_nb = all_data["data_file_nb"]
                data_keys = all_data["data_keys"]

            self.data = {}
            for key in data_keys:
                self.new_name(key)

            for data_file_id in range(self.data_file_nb):
                data_file_name = "data{}.json".format(data_file_id)
                with open(self.path / data_file_name, "r") as f:
                    cur_data = json.load(f)
                for key, value in cur_data.items():
                    self.data[key]["values"].extend(value["values"])
                    self.data[key]["times"].extend(value["times"])

    def subsample_log(self, inp: np.ndarray, n_target: int):
        return inp[:: len(inp) // n_target]

    def convolve(self, inp: np.ndarray, gaussian):
        filtered = np.convolve(inp, gaussian)
        n = len(filtered) - len(inp)
        return filtered[int(np.floor(n / 2)) : int(-np.ceil(n / 2))]

    def filter_log(self, inp: np.ndarray, sigma):
        xs = np.linspace(-3, 3, 2 * 3 * sigma)
        gaussian = np.exp(-np.square(xs) / 2)
        gaussian = gaussian / np.sum(gaussian)
        esp = self.convolve(inp, gaussian)
        esp_2 = self.convolve(np.square(inp), gaussian)

        return esp, np.sqrt(esp_2 - np.square(esp))

    def plot_wall(self, name, do_subsample=True, filter_sigma=0):
        xs = np.array(self.data[name]["times"])
        xs = xs - xs[0]
        ys = np.array(self.data[name]["values"])
        self.plot_data(xs, ys, do_subsample, filter_sigma)
        plt.xlabel("wall clock time (s)")
        plt.ylabel(name)

    def plot(self, name, do_subsample=True, filter_sigma=0):
        ys = np.array(self.data[name]["values"])
        xs = np.arange(len(ys))
        self.plot_data(xs, ys, do_subsample, filter_sigma)
        plt.xlabel("iterations")
        plt.ylabel(name)

    def plot_data(self, xs, ys, do_subsample, filter_sigma):
        if filter_sigma > 0:
            ys, std = self.filter_log(ys, filter_sigma)
        if do_subsample:
            n_target = min(len(xs), 1000)
            xs = self.subsample_log(xs, n_target)
            ys = self.subsample_log(ys, n_target)
            if filter_sigma > 0:
                std = self.subsample_log(std, n_target)

        plt.plot(xs, ys, label=self.name)
        if filter_sigma > 0:
            plt.fill_between(xs, ys - std, ys + std, alpha=0.3)

    def summerize(self):
        print(self.name, self.description, sep=" : ")
