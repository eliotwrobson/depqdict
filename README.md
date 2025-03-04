# DepqDict

[![Project Status: Active – The project has reached a stable, usable state and is being actively developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)


DepqDict is an efficient double-ended priority queue implemented in Python. It allows efficient retrieval and extraction of the minimum and maximum priority elements, as well as updating priorities of arbitrary elements. Implemented using a
[min-max heap](https://en.wikipedia.org/wiki/Min-max_heap).

## Features

- Supports both min and max priority queue operations.
- Efficient push and pop operations for both min and max elements.
- Allows updating priorities of arbitrary keys.
- Implements the `MutableMapping` interface, behaving like a dictionary.

## Installation

You can install DepqDict from PyPI using pip:

```sh
pip install depqdict
```

## Usage

```python
from depqdict import DepqDict

pq = DepqDict()
pq['task1'] = 5
pq['task2'] = 2
pq['task3'] = 8

print(pq.min_item())  # ('task2', 2)
print(pq.max_item())  # ('task3', 8)

pq['task1'] = 1  # Update priority
print(pq.min_item())  # ('task1', 1)

pq.pop_min_item()  # Removes ('task1', 1)
```

## API with Runtimes

### Retrieve Min/Max Items

```python
pq.min_item()  # O(1)
pq.max_item()  # O(1)
```
Returns the key-value pair with the lowest or highest priority.

### Remove Min/Max Items

```python
pq.pop_min_item()  # O(log n)
pq.pop_max_item()  # O(log n)
```
Removes and returns the key-value pair with the lowest or highest priority.

### Push-Pop Operations

```python
pq.push_pop_min_item('task4', 3)  # O(log n)
pq.push_pop_max_item('task5', 9)  # O(log n)
```
Pushes a new key-priority pair into the queue and returns the smallest or largest item. Faster than using two separate operations.

### Insert or Update a Key

```python
pq['task6'] = 4  # O(log n)
```
Inserts or updates the priority of a key.

### Delete a Key

```python
del pq['task3']  # O(log n)
```
Removes a key from the queue.

## License

This project is licensed under the GPL-3.0 License, as it is based on heapdict by Evgeniy Selezniov, which is also licensed under GPL-3.0. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

This package is based on [heapdict](https://github.com/nanouasyn/heapdict), originally developed by Evgeniy Selezniov.
