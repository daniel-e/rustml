# Machine Learning Pipeline

When doing machine learning it is quite common that you split a big task into a set of small subtasks which are executed sequentially or in parallel. Each subtask reads data from a file, a database, from stdin or some other source, performs a transformation of the data and produces some output that again is stored in a file, a database or that is simply written to stdout. The output is usually consumed by another subtask until the final result has been created.

This is a typical example of a pipeline:
```
*****************    +-------------+    *****************    +-------+    ********************    +----------+
* list of files *--->| parse_files |--->* list of words *--->| TFIDF |--->* words with score *--->| keywords |
*****************    +-------------+    *****************    +-------+    ********************    +----------+
```
The goal of this simplified machine learning task is to create a list of keywords for a list of files. The task is divided into some subtasks. The first task is `parse_files`. This task takes the list of files as input and creates for each file in that list a list of words. The output of the task is a list of list of words. The output can be written into a file or a database or it can be written to stdout. The second task `TFIDF` takes this list of list of words and computes for each word a score (e.g. the TFIDF) which reflects the importance of the word for the appropriate file. This task outputs the words with their score for each file. Finally, the task `keywords` selects the words with the highest score as the keywords of a file.

Each subtask could be implemented in separate binary or script and the binaries or scripts could be linked together via standard Unix pipes, e.g.
```
./parse_files.py < list_of_files | ./TFIDF.py | ./keywords.py > result
```

