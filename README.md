# Pacman-Agent
## Intro. to Artificial Intelligence Spring 2015

A Pacman project based on UC Berkeley [Intro to AI][1]

It contains many kinds of agent that using AI technique.

## Environment Requirement

- Python 2.7
- python-tk

## Usage

Following will reveal the usage on UNIX-like platform.

If you want to run on Windows, [WinPython][2] is recommended.

First, change working directory to *Pacman-Agent*.

Second, type such as:

    python ./pacman.py -l P1-3 -g StraightRandomGhost -p FroggerAgent

to enjoy it!!

*argument:*
 -p: agent type
 -l: layout
 -g: ghost type

To get more information about the argument, refer to *spec/*

## Some Techniques

- Rule-based
- DFS
- BFS
- A* search
- Alpha-Beta pruning
- minimax search
- Evaluation function

[1]: http://ai.berkeley.edu/project_overview.html
[2]: https://winpython.github.io
