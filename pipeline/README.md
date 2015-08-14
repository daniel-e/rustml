# Machine Learning Pipeline

When doing machine learning it is quite common that you split a big task into a set of small subtasks which are executed sequentially or in parallel. Each subtask reads data from a file, a database, from stdin or some other source, performs a transformation of the data and produces some output that again is stored in a file, a database or that is simply written to stdout. The output is usually consumed by another subtask until the final result has been created.

This is a typical example of a pipeline:
```
*****************    +-------------+    *********    +-------+    **********    +----------+
* list of files *--->| parse_files |--->* words *--->| TFIDF |--->* scores *--->| keywords |
*****************    +-------------+    *********    +-------+    **********    +----------+
```
The goal of this simplified machine learning task is to create a list of keywords for a list of files based on the contents of the files. The task is divided into some subtasks. The first task is `parse_files`. This task takes the list of files as input and creates for each file in that list a list of words. The output of the task is a list of list of words. The output can be written into a file or a database or it can be written to stdout. The second task `TFIDF` takes this list of list of words and computes for each word a score (e.g. the TFIDF) which reflects the importance of the word for the appropriate file. This task outputs the words with their score for each file. Finally, the task `keywords` selects the words with the highest score as the keywords of a file.

Each subtask could be implemented in separate binary or script and the binaries or scripts could be linked together via standard Unix pipes, e.g.
```
./parse_files.py < list_of_files | ./TFIDF.py | ./keywords.py > result
```

The advantage of this approach is that a subtask that implements algorithm A can be simply replaced by a subtask that implements algorithm B. In the example above `TFIDF.py` (which computes the term frequency inverse document frequency) could be replaced by the script `TF.py`, which simply uses the term frequencies to compute the score of each word.

## The pipeline in rustml

Rustml offers a simple but very powerful way to build a pipeline that can be easily configured via one configuration file using JSON. The typical workflow is as follows:

* call `pipe_init` to create a new pipeline
* edit the configuration file
* call `pipe_config` to create a Makefile that contains all dependencies between the subtasks
* call `make` to build the subtasks (if not up-to-date) and execute the pipeline

The Rustml pipeline uses GNU Make to model the dependencies between the subtasks so that only those parts of the pipeline needs to be rebuild and executed that have changed. 
