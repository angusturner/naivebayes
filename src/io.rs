/// this module will contain functions for importing and exporting data
extern crate csv;
extern crate rustc_serialize;
extern crate rulinalg as rl;

use self::rl::matrix::Matrix;

/// read CSV into a matrix
pub fn read_csv(file_path: &str) -> Matrix<f64> {
    // initialise some stuff
    let mut rdr = csv::Reader::from_file(file_path).unwrap().has_headers(false);
    let mut out: Vec<f64> = vec![]; // output vector
    let mut m: usize = 0; // number of rows
    let mut n: usize = 0; // number of cols
    let mut skipped = 0u32;

    // iterate through records
    for record in rdr.decode() {
        // decode row into a vector
        let mut row: Vec<f64> = match record {
            Ok(res) => res,
            Err(err) => vec![],
        };

        if row.len() > 0 {
            // get number of columns
            if n == 0 {
                n = row.len();
            }

            // append row to output vector
            out.append(&mut row);

            // increment number of rows
            m += 1usize;
        } else {
            skipped += 1;
        }
    }

    // Print number of skipped rows
    println!("Omitted {} rows due to parse errors", skipped);

    // reshape data into matrix
    Matrix::new(m, n, out)
}
