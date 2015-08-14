# Machine Learning Pipeline

When doing machine learning it is quite common that you split a big task
into a set of small subtasks which are executed sequentially or in
parallel. Each subtask reads data from a file, a
database, from stdin or some other source, performs a transformation of the
data and produces some output that again is stored in a file, a
database or that is simply written to stdout. The output is usually consumed by
another subtask until the final result has been created.

This is an example of a typical pipeline:


