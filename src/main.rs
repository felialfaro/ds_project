use std::error::Error;
use std::fs::File;
use serde::Deserialize;
use csv::{ReaderBuilder, Writer};
use linfa::prelude::*;
use linfa_clustering::KMeans;
use linfa_linear::LinearRegression;
use plotters::prelude::*;
use std::collections::HashMap;

#[derive(Debug, Deserialize)]
struct Record {
    #[serde(rename = "Product type")]
    product: String,
    #[serde(rename = "SKU")]
    sku: String,
    #[serde(rename = "Costs")]
    cost: f64,
    #[serde(rename = "Manufacturing lead time")]
    lead_time: u32,
    #[serde(rename = "Availability")]
    status: String,
}

fn calculate_correlation(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len() as f64;
    let mean_x = x.iter().sum::<f64>() / n;
    let mean_y = y.iter().sum::<f64>() / n;
    let numerator: f64 = x.iter().zip(y).map(|(&xi, &yi)| (xi - mean_x) * (yi - mean_y)).sum();
    let denominator = (x.iter().map(|&xi| (xi - mean_x).powi(2)).sum::<f64>()
        * y.iter().map(|&yi| (yi - mean_y).powi(2)).sum::<f64>()).sqrt();
    numerator / denominator
}

fn perform_correlation_analysis(records: &[Record]) {
    let costs: Vec<f64> = records.iter().map(|r| r.cost).collect();
    let lead_times: Vec<f64> = records.iter().map(|r| r.lead_time as f64).collect();

    let correlation = calculate_correlation(&costs, &lead_times);
    println!("Correlation between Cost and Lead Time: {:.2}", correlation);
}

fn feature_engineering(records: &[Record]) -> Vec<HashMap<String, f64>> {
    records
        .iter()
        .map(|r| {
            let mut features = HashMap::new();
            features.insert("cost".to_string(), r.cost);
            features.insert("lead_time".to_string(), r.lead_time as f64);
            features.insert("cost_per_lead_time".to_string(), r.cost / r.lead_time as f64);
            features
        })
        .collect()
}

fn detect_outliers(records: &[Record]) {
    let costs: Vec<f64> = records.iter().map(|r| r.cost).collect();
    let mean = costs.iter().sum::<f64>() / costs.len() as f64;
    let std_dev = (costs.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / costs.len() as f64).sqrt();

    for (i, cost) in costs.iter().enumerate() {
        let z_score = (cost - mean) / std_dev;
        if z_score.abs() > 3.0 {
            println!("Record {} is an outlier with Z-score: {:.2}", i, z_score);
        }
    }
}

fn visualize_data(records: &[Record]) -> Result<(), Box<dyn Error>> {
    let root = BitMapBackend::new("scatter_plot.png", (640, 480)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption("Cost vs Lead Time", ("sans-serif", 20).into_font())
        .margin(10)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(
            0f64..records.iter().map(|r| r.cost).fold(0. / 0., f64::max), // Max cost
            0f64..records.iter().map(|r| r.lead_time as f64).fold(0. / 0., f64::max), // Max lead time
        )?;

    chart.configure_mesh().draw()?;

    let data: Vec<(f64, f64)> = records
        .iter()
        .map(|r| (r.cost, r.lead_time as f64))
        .collect();

    chart.draw_series(data.iter().map(|&(x, y)| {
        Circle::new((x, y), 5, ShapeStyle {
            color: BLUE.to_rgba(),
            filled: true,
            stroke_width: 1,
        })
    }))?;

    Ok(())
}

fn calculate_average_cost(records: &[Record]) -> f64 {
    let total_cost: f64 = records.iter().map(|r| r.cost).sum();
    total_cost / records.len() as f64
}

fn perform_clustering(records: &[Record], num_clusters: usize) -> Result<(), Box<dyn Error>> {
    // Prepare the data: Extract `cost` and `lead_time` into a 2D array
    let data: ndarray::Array2<f64> = ndarray::Array2::from_shape_vec(
        (records.len(), 2),
        records
            .iter()
            .flat_map(|r| vec![r.cost, r.lead_time as f64])
            .collect(),
    )?;

    // Wrap the data in a DatasetBase (features only, no targets)
    let dataset = DatasetBase::from(data);

    // Perform K-Means clustering
    let kmeans = KMeans::params(num_clusters).fit(&dataset)?;

    println!("Cluster assignments:");
    for (i, cluster) in kmeans.predict(dataset.records()).iter().enumerate() {
        println!("Record {}: Cluster {}", i, cluster);
    }

    // Write clustering results to a CSV file
    let mut wtr = Writer::from_writer(File::create("clustering_results.csv")?);
    wtr.write_record(&["Product", "SKU", "Cost", "Lead Time", "Status", "Cluster"])?;

    for (record, cluster) in records.iter().zip(kmeans.predict(dataset.records())) {
        wtr.write_record(&[
            &record.product,
            &record.sku,
            &record.cost.to_string(),
            &record.lead_time.to_string(),
            &record.status,
            &cluster.to_string(),
        ])?;
    }

    wtr.flush()?;
    println!("Clustering results saved to clustering_results.csv");

    Ok(())
}

fn perform_regression(records: &[Record]) -> Result<(), Box<dyn Error>> {
    // Prepare the dataset
    let features: ndarray::Array2<f64> = ndarray::Array2::from_shape_vec(
        (records.len(), 1),
        records.iter().map(|r| r.cost).collect(),
    )?;
    let targets: ndarray::Array1<f64> = ndarray::Array1::from(
        records.iter().map(|r| r.lead_time as f64).collect::<Vec<f64>>(),
    );

    let dataset = Dataset::new(features, targets);

    // Train the linear regression model
    let model = LinearRegression::default().fit(&dataset)?;

    println!("Regression Model Coefficients: {:?}", model.params());
    println!("Regression Model Intercept: {:?}", model.intercept());

    // Predict lead times for the dataset
    let predictions = model.predict(dataset.records());

    println!("Predicted Lead Times:");
    for (i, pred) in predictions.iter().enumerate() {
        println!("Record {}: Predicted Lead Time = {:.2}", i, pred);
    }

    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let file_path = "Data Science Project Dataset.csv";

    let mut rdr = ReaderBuilder::new()
        .has_headers(true)
        .from_path(file_path)?;

    let mut records: Vec<Record> = Vec::new();
    for result in rdr.deserialize() {
        let record: Record = result?;
        records.push(record);
    }

    println!("Parsed Records:");
    for record in &records {
        println!("{:?}", record);
    }
    // Perform correlation analysis
    perform_correlation_analysis(&records);

    // Feature engineering
    let engineered_features = feature_engineering(&records);
    println!("Engineering Features: {:?}", engineered_features);

    //Outlier detection
    detect_outliers(&records);

    //Visualization
    visualize_data(&records)?;

    // Calculate and print average cost
    let avg_cost = calculate_average_cost(&records);
    println!("Average Cost: {:.2}", avg_cost);

    // Perform clustering
    perform_clustering(&records, 3)?;

    // Perform regression
    perform_regression(&records)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calculate_average_cost() {
        let records = vec![
            Record {
                product: "Product A".to_string(),
                sku: "SKU001".to_string(),
                cost: 100.0,
                lead_time: 10,
                status: "Available".to_string(),
            },
            Record {
                product: "Product B".to_string(),
                sku: "SKU002".to_string(),
                cost: 200.0,
                lead_time: 15,
                status: "Unavailable".to_string(),
            },
        ];

        let avg_cost = calculate_average_cost(&records);
        assert_eq!(avg_cost, 150.0); // (100 + 200) / 2
    }

    #[test]
    fn test_perform_clustering() {
        let records = vec![
            Record {
                product: "Product A".to_string(),
                sku: "SKU001".to_string(),
                cost: 100.0,
                lead_time: 10,
                status: "Available".to_string(),
            },
            Record {
                product: "Product B".to_string(),
                sku: "SKU002".to_string(),
                cost: 200.0,
                lead_time: 20,
                status: "Unavailable".to_string(),
            },
        ];

        let result = perform_clustering(&records, 2);
        assert!(result.is_ok()); // Check that clustering runs successfully
    }

    #[test]
    fn test_perform_regression() {
        let records = vec![
            Record {
                product: "Product A".to_string(),
                sku: "SKU001".to_string(),
                cost: 100.0,
                lead_time: 10,
                status: "Available".to_string(),
            },
            Record {
                product: "Product B".to_string(),
                sku: "SKU002".to_string(),
                cost: 200.0,
                lead_time: 20,
                status: "Unavailable".to_string(),
            },
        ];

        let result = perform_regression(&records);
        assert!(result.is_ok()); // Check that regression runs successfully
    }
}
