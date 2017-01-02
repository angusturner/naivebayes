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
    let split_index = (m as f64 * 0.67).round() as usize;
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
    let l = 1f64; // constant for laplace smoothing of the max-likelihood estimates

    // get the frequency of each class in the training set
    let mut freq: Vec<f64> = vec![0f64; K];
    freq = y_train.row_iter()
    .fold(freq, |mut acc, y_i| {
        acc[y_i[0] as usize] += 1.0;
        acc
    });

    // get the class priors from formula
    // prior(y_k) = ( #D{Y = y_k} + l ) / ( |D| + l*K ), where
    // y_k = class we interested in
    // D = the set of all labels (Y_train)
    // #D{x} = number of elements in set D fulfilling predicate x
    // K = number of output classes (malignant / not maligant)
    // l = smoothing constant (value of 1 for "Laplace Smoothing")
    let m = y_train.rows();
    let priors = freq.iter().map(|x| (x + 1.0) / (m as f64 + l * K as f64)).collect::<Vec<f64>>();

    // print out the class priors
    println!("{:?}", priors);

    // sort the training data into groups according to their class label
    let mut clusters: Vec<Matrix<f64>> = vec![Matrix::zeros(0, n); K];
    clusters = X_train.row_iter().zip(y_train.row_iter())
    .fold(clusters, |mut acc, (x_i, y_i)| {
        acc[y_i[0] as usize] = acc[y_i[0] as usize].vcat(&x_i);
        acc
    });

    // for every feature-class combination, compute the # occurences of each of the J discrete
    // values.
    let mut params: Vec<Matrix<f64>> = vec![Matrix::zeros(J, n); K];

    // iterate over each cluster
    for k in 0..clusters.len() {
        // compute frequency of each value occurence for each feature
        for i in 0..clusters[k].rows() {
            // collect row x_i into a vector
            let x_i = clusters[k].row(i).iter().collect::<Vec<&f64>>();
            for j in 0..x_i.len() {
                let c = *x_i[j] as usize - 1;
                params[k][[c, j]] += 1.0;
            }
        }

        // divide through by total occurences of class k, plus the smoothing parameter
        params[k] = (&params[k] + l) / (freq[k] + (l * J as f64));
    }

    // make predictions for the test set
    let mut probs: Matrix<f64> = Matrix::zeros(X_test.rows(), K);
    for k in 0..K {
        for i in 0..X_test.rows() {
            // calculate the probability of class k for this training example
            let x_i = X_test.row(i).iter().collect::<Vec<&f64>>();
            for j in 0..n {
                let c = *x_i[j] as usize - 1;
                probs[[i, k]] += params[k][[c, j]];
            }

            // multiply by the class prior
            probs[[i, k]] = &probs[[i, k]] * priors[k];
        }
    }

    // convert the probability matrix into a vector containing the index of the maximum in every row
    let pred: Vec<usize> = row_max(&probs);
    
    let y_test_vec: Vec<f64> = y_test.iter().map(|x| *x).collect::<Vec<f64>>();
    let test_acc = accuracy(&pred, &y_test_vec);

    println!("{:?}", pred);
    println!("Test Accuracy {}", test_acc);
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
        _ => panic!("Invalid class label")
    }
}
