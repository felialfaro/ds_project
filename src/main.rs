use std::error::Error;
use csv::ReaderBuilder;
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct Record{
	product: String,
	sku: String,
	cost: f64, 
	lead_time: u32,
	status: String, 
}

fn main() -> Result<(), Box<dyn Error>> {
    // Specify the dataset file path
    let file_path = "Data Science Project Dataset.csv";

    let mut rdr = ReaderBuilder::new()
        .has_headers(true) 
        .from_path(file_path)?;

    // Read and print each record
    println!("Dataset Records:");
    for result in rdr.records() {
        let record = result?;
        println!("{:?}", record);
    }

    Ok(())
}

