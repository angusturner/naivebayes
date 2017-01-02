# naivebayes

Implementation of a naive bayes classifier in rust.
Achieves ~91% accuracy on the wisconsin breast cancer data set:
http://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29

## Project Structure

Naive bayes is implemented entirely in `main.rs`, with `io.rs` and `util.rs`
providing a handful of helper functions for inputting data and matrix
manipulation.

## Progress
* [X] Import data from CSV
* [X] Basic algorithm for discrete input features
* [ ] Implement Gaussian PDF, to allow for continuous inputs
* [ ] Refactor, to allow arbitrary number of classes for each feature and a mix of continuous/discrete features
