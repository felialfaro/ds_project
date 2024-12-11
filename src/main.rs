use std::error::Error;
use csv::ReaderBuilder;
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct Record{
    #[serde(rename = "Product type")]
    product: String,
    #[serde(rename = "SKU")]
    sku: String,
    #[serde(rename = "Price")]
    cost: f64,
    #[serde(rename = "Lead time")]
    lead_time: u32,
    #[serde(rename = "Availability")]
    status: String,
}

fn calculate_average_cost(records: &[Record]) -> f64 {
	let total_cost: f64 = records.iter().map(|r| r.cost).sum();
	total_cost / records.len() as f64
}

fn main() -> Result<(), Box<dyn Error>> {
    // Specify the dataset file path
    let file_path = "Data Science Project Dataset.csv";

    let mut rdr = ReaderBuilder::new()
        .has_headers(true) 
        .from_path(file_path)?;

    let mut records: Vec<Record> = Vec::new();
    for result in rdr.deserialize() {
	let record: Record = result?;
	records.push(record);
    }   

    println!("Dataset Records:");
    for result in rdr.records() {
        let record = result?;
        println!("{:?}", record);
    }
    let avg_cost = calculate_average_cost(&records);
    println!("Average Cost: {:.2}", avg_cost);

    Ok(())
}

