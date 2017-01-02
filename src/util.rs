/// this module contains helper functions for matrix manipulation
extern crate rulinalg as rl;

use std::cmp;
use self::rl::matrix::{Matrix, BaseMatrix};

/// reduce a matrix into a vector containing the index of the maximum in every row
pub fn row_max(mat: &Matrix<f64>) -> Vec<usize> {
    let (m, n) = dims(&mat);

    // transpose, then convert to vector (column-major order)
    let (_i, _v, res) = mat.iter()
        // how to abuse reduce patterns
        .fold((0usize, 0f64, vec![0usize; m]), |acc, &val| {
            let (mut i, mut v, mut vec) = acc;
            if &i % &n == 0 {
                v = 0f64;
            }
            if val > v {
                v = val;
                let ind = &i/&n;
                vec[ind] = (&i % &n) as usize;
            }
            i += 1;
            (i, v, vec)
        });
    res
}

/// get matrix dimensions
pub fn dims(mat: &Matrix<f64>) -> (usize, usize) {
    (mat.rows(), mat.cols())
}
