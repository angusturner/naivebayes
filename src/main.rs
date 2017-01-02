extern crate rulinalg as rl;
extern crate rand;
extern crate naivebayes;

use naivebayes::util::*;
use naivebayes::io::read_csv;
use rl::matrix::{Axes, Matrix, MatrixSlice, BaseMatrix, BaseMatrixMut};

#[allow(non_snake_case)]
fn main() {
    // read in the data set, omitting entries with missing attributes
    let raw: Matrix<f64> = read_csv("breast-cancer-wisconsin.csv");

    // remove the first column, pertaining to patient IDs
    let (_, data) = raw.split_at(1, Axes::Col);

    // separate the final column, containing class labels, from the feature attributes
    let (X, y_raw) = data.split_at(9, Axes::Col);

    // map the class labels {2, 4} to {0, 1}
    // 0 = benign, 1 = malignant
    let y = y_raw.into_matrix().apply(&map_class_labels);

    // split the data into training and test sets at 0.67 ratio
    let m = data.rows();
    let split_index = (m as f64 * 0.5).round() as usize;
    let (X_train, X_test) = X.split_at(split_index, Axes::Row);
    let (y_train, y_test) = y.split_at(split_index, Axes::Row);

    // print out the various matrix dimensions
    println!("X_train [{}, {}]", X_train.rows(), X_train.cols());
    println!("X_test [{}, {}]", X_test.rows(), X_test.cols());
    println!("y_train [{}, {}]", y_train.rows(), y_train.cols());
    println!("y_test [{}, {}]", y_test.rows(), y_test.cols());

    // define some useful constants
    // TODO: determine these from the input data
    let J = 10; // number of discrete values that each attribute x_i can take on
    let K = 2; // number of discrete values for the data labels y_i
    let n = X_train.cols(); // number of features
    let m = X_train.rows(); // number of training examples
    let l = 1f64; // constant for laplace smoothing of the max-likelihood estimates

    // get the frequency of each class in the training set
    let freq: Vec<f64> = class_freq(&y_train, K);

    // get the prior probabilities by dividing through by number of training examples
    let priors: Vec<f64> = class_priors(&freq, l, m);

    // sort the training data into groups according to their class label
    let clusters = group_by_class(&X_train, &y_train, K);

    // get the likelihoods P (X | Y)
    let likelihoods: Vec<Matrix<f64>> = likelihoods(&clusters, &freq, J, l);

    // sum the evidence of each test example falling into each class K
    // NOTE: this is P(Y|X) * P(X). We do not need to divide through by
    // P(X) in order to make prediction.
    let evidence: Matrix<f64> = evidence(&X_test, &likelihoods, &priors);

    // calculated predicted classes by taking the column index of the max in each row
    let prediction: Vec<usize> = row_max(&evidence);

    // calculate test-set accuracy
    let y_test_vec: Vec<f64> = y_test.iter().map(|x| *x).collect::<Vec<f64>>();
    let test_acc = accuracy(&prediction, &y_test_vec);

    //
    println!("Test Accuracy {}", test_acc);
}

// For each value of the K output classes, construct a matrix that contains the 'likelihood' of
// of each of the n input features taking on each of its J possible values. Return in a vector.
fn likelihoods(clusters: &Vec<Matrix<f64>>, freq: &Vec<f64>, J: usize, l: f64) -> Vec<Matrix<f64>> {
    let n = clusters[0].cols();
    let K = freq.len();
    let mut likelihoods: Vec<Matrix<f64>> = vec![Matrix::zeros(J, n); K];

    // iterate over each cluster
    for k in 0..clusters.len() {
        // compute # occurences of each value for each feature
        for i in 0..clusters[k].rows() {
            // collect row x_i into a vector
            let x_i = clusters[k].row(i).iter().collect::<Vec<&f64>>();
            for j in 0..x_i.len() {
                let c = *x_i[j] as usize - 1;
                likelihoods[k][[c, j]] += 1.0;
            }
        }

        // divide through by total occurences of class k, plus the smoothing parameter
        likelihoods[k] = (&likelihoods[k] + l) / (freq[k] + (l * J as f64));
    }

    likelihoods
}

/// Get the priors probabilities for each of the output classes.
/// prior(y_k) = ( #D{Y = y_k} + l ) / ( |D| + l*K ), where
/// y_k = the class (i.e malignant / not malignant)
/// D = the set of all labels (Y_train)
/// #D{x} = number of elements in set D fulfilling predicate x
/// K = number of output classes
/// l = smoothing constant (value of 1 for "Laplace Smoothing")
fn class_priors(freq: &Vec<f64>, l: f64, m: usize) -> Vec<f64> {
    let K = freq.len();
    freq.iter().map(|x| (x + l) / (m as f64 + l * K as f64)).collect::<Vec<f64>>()
}

/// Calculate the frequency of each label class in the training set
/// y_train - the training labels
/// K - the number of labels
fn class_freq(y_train: &MatrixSlice<f64>, K: usize) -> Vec<f64> {
    // calculate the frequency of each class occurence
    let mut freq: Vec<f64> = vec![0f64; K];
    y_train.row_iter()
        .fold(freq, |mut acc, y_i| {
            acc[y_i[0] as usize] += 1.0;
            acc
        })
}

// Group the training examples in X_train by their label, returning a vector of matrices where
// the i-th entry contains the training examples with label `i`
fn group_by_class(X_train: &MatrixSlice<f64>,
                  y_train: &MatrixSlice<f64>,
                  K: usize)
                  -> Vec<Matrix<f64>> {
    let n: usize = X_train.cols();
    let mut clusters: Vec<Matrix<f64>> = vec![Matrix::zeros(0, n); K];
    X_train.row_iter()
        .zip(y_train.row_iter())
        .fold(clusters, |mut acc, (x_i, y_i)| {
            acc[y_i[0] as usize] = acc[y_i[0] as usize].vcat(&x_i);
            acc
        })
}

/// Determine the evidence for each test example taking on each of the K possible classes.
/// P(Y = y_k) * P(X_i | Y = y_k)
fn evidence(X_test: &MatrixSlice<f64>,
            likelihoods: &Vec<Matrix<f64>>,
            priors: &Vec<f64>)
            -> Matrix<f64> {
    let K: usize = priors.len();
    let mut evidence: Matrix<f64> = Matrix::zeros(X_test.rows(), K);
    for k in 0..K {
        for i in 0..X_test.rows() {
            // calculate the probability of class k for this training example
            let x_i = X_test.row(i).iter().collect::<Vec<&f64>>();
            for j in 0..x_i.len() {
                let c = *x_i[j] as usize - 1;
                evidence[[i, k]] += likelihoods[k][[c, j]];
            }

            // multiply by the class prior P(Y = y_k)
            evidence[[i, k]] = &evidence[[i, k]] * priors[k];
        }
    }
    evidence
}

/// compute the accuracy by mapping the labels and corresponding predictions
/// to an iter of booleans, and then computing the fraction of `true` entries
fn accuracy(pred: &Vec<usize>, y: &Vec<f64>) -> f64 {
    let (correct, total) = pred.iter()
        .zip(y)
        .map(|(a, b)| (*a as f64) == *b)
        .fold((0f64, 0f64), |(correct, total), val| {
            match val {
                true => (correct + 1.0, total + 1.0),
                false => (correct, total + 1.0),
            }
        });
    correct / total
}

// map the class labels {2, 4} -> {0, 1}
fn map_class_labels(y_i: f64) -> f64 {
    match y_i {
        2.0 => 0.0,
        4.0 => 1.0,
        _ => panic!("Invalid class label"),
    }
}
