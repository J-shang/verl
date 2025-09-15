# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Callable

import torch
import triton


class CUDATIMING_LOGGER(Callable):
    def __init__(self):
        self._timers: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]] = {}

    def start(self, tag: str):
        if tag not in self._timers:
            self._timers[tag] = []
        self._timers[tag].append([torch.cuda.Event(enable_timing=True)])
        self._timers[tag][-1][0].record()

    def end(self, tag: str):
        self._timers[tag][-1].append(torch.cuda.Event(enable_timing=True))
        self._timers[tag][-1][1].record()

    def clear(self):
        self._timers = {}

    def summary(self, num_iters: int):
        result = {
            tag: sum([start.elapsed_time(end) for start, end in timer_list[-num_iters:]]) / num_iters
            for tag, timer_list in self._timers.items()
        }
        self.clear()
        return result

    def __call__(self, tag):
        if tag not in self._timers or len(self._timers[tag][-1]) == 2:
            self.start(tag)
        else:
            self.end(tag)


TIMING_LOGGER = CUDATIMING_LOGGER()


def is_hip():
    return triton.runtime.driver.active.get_current_target().backend == "hip"
