# -*- coding: utf-8 -*-

import os


class Task(object):
    """Base class object for tasks."""
    def __init__(self):
        self.checkpoint_dir = None

    def run(self):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError

    def evaluate(self):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError

    def save_checkpoint(self):
        raise NotImplementedError

    def load_model_from_checkpoint(self):
        raise NotImplementedError

    def load_history_from_checkpoint(self):
        raise NotImplementedError

    @property
    def best_ckpt(self):
        return os.path.join(self.checkpoint_dir, "best_model.pt")

    @property
    def last_ckpt(self):
        return os.path.join(self.checkpoint_dir, "last_model.pt")
