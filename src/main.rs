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
    #[serde(rename = "Price")]
    price: f64,
    #[serde(rename = "Number of products sold")]
    quantity_sold: u32,
    #[serde(rename = "Costs")]
    cost: f64,
    #[serde(rename = "Manufacturing lead time")]
    lead_time: u32,
    #[serde(rename = "Shipping times")]
    shipping_time: u32,
    #[serde(rename = "Shipping costs")]
    shipping_cost: f64,
    #[serde(rename = "Location")]
    location: String,
    #[serde(rename = "Customer demographics")]
    demographic: String,
    #[serde(rename = "Availability")]
    status: String,
}

    fn calculate_revenue_analysis(records: &[Record]) -> HashMap<String, f64> {
    let mut product_revenues: HashMap<String, f64> = HashMap::new();
    for record in records {
        let revenue = record.price * record.quantity_sold as f64;
        *product_revenues.entry(record.product.clone()).or_insert(0.0) += revenue;
    }

    let mut sorted_products: Vec<_> = product_revenues.iter().collect();
    sorted_products.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());

    println!("Most Profitable Products:");
    for (product, revenue) in sorted_products.iter().take(5) {
        println!("{}: ${:.2}", product, revenue);
    }

    product_revenues
}

fn analyze_supply_chain_efficiency(records: &[Record]) {
    let shipping_times: Vec<f64> = records.iter().map(|r| r.shipping_time as f64).collect();
    let shipping_costs: Vec<f64> = records.iter().map(|r| r.shipping_cost).collect();

    let correlation = calculate_correlation(&shipping_times, &shipping_costs);
    println!("Correlation between Shipping Time and Shipping Cost: {:.2}", correlation);

    let bottlenecks: Vec<_> = records
        .iter()
        .filter(|r| r.lead_time > 30 && r.status == "Available")
        .collect();

    println!("Supply Chain Bottlenecks:");
    for record in bottlenecks {
        println!("Product: {}, Lead Time: {} days", record.product, record.lead_time);
    }
}

fn identify_optimization_opportunities(records: &[Record]) {
    let mut carrier_costs: HashMap<String, f64> = HashMap::new();
    for record in records {
        *carrier_costs.entry(record.location.clone()).or_insert(0.0) += record.shipping_cost;
    }

    let mut sorted_carriers: Vec<_> = carrier_costs.iter().collect();
    sorted_carriers.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());

    println!("High-Cost Shipping Locations:");
    for (carrier, cost) in sorted_carriers.iter().take(5) {
        println!("{}: ${:.2}", carrier, cost);
    }
}

fn perform_predictive_modeling(records: &[Record]) -> Result<(), Box<dyn Error>> {
    // Predict Revenue based on Price and Quantity Sold
    let features: ndarray::Array2<f64> = ndarray::Array2::from_shape_vec(
        (records.len(), 2),
        records
            .iter()
            .flat_map(|r| vec![r.price, r.quantity_sold as f64])
            .collect(),
    )?;
    let targets: ndarray::Array1<f64> = ndarray::Array1::from(
        records
            .iter()
            .map(|r| r.price * r.quantity_sold as f64)
            .collect::<Vec<f64>>(),
    );

    let dataset = Dataset::new(features, targets);

    let model = LinearRegression::default().fit(&dataset)?;

    println!("Revenue Prediction Model Coefficients: {:?}", model.params());
    println!("Revenue Prediction Model Intercept: {:.2}", model.intercept());

    Ok(())
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

fn detect_outliers(records: &[Record]) -> Result<(), Box<dyn Error>> {
    let costs: Vec<f64> = records.iter().map(|r| r.cost).collect();
    let mean = costs.iter().sum::<f64>() / costs.len() as f64;
    let std_dev = (costs.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / costs.len() as f64).sqrt();

    let mut wtr = Writer::from_writer(File::create("outliers.csv")?);
    wtr.write_record(&["Record ID", "Cost", "Z-Score"])?;

    for (i, cost) in costs.iter().enumerate() {
        let z_score = (cost - mean) / std_dev;
        if z_score.abs() > 3.0 {
            wtr.write_record(&[i.to_string(), cost.to_string(), z_score.to_string()])?;
        }
    }
    wtr.flush()?;
    println!("Outliers detected and saved to outliers.csv");

    Ok(())
}
fn visualize_data_with_trend(records: &[Record]) -> Result<(), Box<dyn Error>> {
    let root = BitMapBackend::new("scatter_plot_with_trend.png", (640, 480)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption("Cost vs Lead Time with Trend Line", ("sans-serif", 20).into_font())
        .margin(10)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(
            0f64..records.iter().map(|r| r.cost).fold(0. / 0., f64::max),
            0f64..records.iter().map(|r| r.lead_time as f64).fold(0. / 0., f64::max),
        )?;

    chart.configure_mesh().draw()?;

    let data: Vec<(f64, f64)> = records
        .iter()
        .map(|r| (r.cost, r.lead_time as f64))
        .collect();

    // Scatter plot points
    chart.draw_series(data.iter().map(|&(x, y)| {
        Circle::new((x, y), 5, ShapeStyle {
            color: BLUE.to_rgba(),
            filled: true,
            stroke_width: 1,
        })
    }))?;

    // Add trend line
    let x_vals: Vec<f64> = data.iter().map(|(x, _)| *x).collect();
    let y_vals: Vec<f64> = data.iter().map(|(_, y)| *y).collect();
    let slope = calculate_correlation(&x_vals, &y_vals)
        * (y_vals.iter().sum::<f64>() / x_vals.iter().sum::<f64>());
    let intercept = y_vals.iter().sum::<f64>() / y_vals.len() as f64;

    chart.draw_series(LineSeries::new(
        data.iter()
            .map(|&(x, _)| (x, slope * x + intercept)),
        &RED,
    ))?;

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

    // Wrap the data in a DatasetBase
    let dataset = DatasetBase::from(data);

    // Perform K-Means clustering
    let kmeans = KMeans::params(num_clusters).fit(&dataset)?;

    //println!("Cluster assignments:");
    //for (i, cluster) in kmeans.predict(dataset.records()).iter().enumerate() {
    //    println!("Record {}: Cluster {}", i, cluster);
    //}

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

    // Save predictions to CSV
    let mut wtr = Writer::from_writer(File::create("regression_predictions.csv")?);
    wtr.write_record(&["Record ID", "Predicted Lead Time"])?;

    for (i, pred) in predictions.iter().enumerate() {
        wtr.write_record(&[i.to_string(), pred.to_string()])?;
    }
    wtr.flush()?;
    println!("Predicted lead times saved to regression_predictions.csv");

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

    //println!("Parsed Records:");
    //for record in &records {
    //    println!("{:?}", record);
    //}

    // Perform Revenue Analysis
    calculate_revenue_analysis(&records);

    // Analyze supply chain efficiency
    analyze_supply_chain_efficiency(&records);

    //Identify optimization opportunities
    identify_optimization_opportunities(&records);

    //Perform predictive modeling
    perform_predictive_modeling(&records);

    // Perform correlation analysis
    perform_correlation_analysis(&records);

    // Feature engineering
    let engineered_features = feature_engineering(&records);

    //Save engineered features to CSV
    let mut wtr = Writer::from_writer(File::create("engineered_features.csv")?);
    wtr.write_record(&["Cost", "Lead Time", "Cost per Lead Time"])?;

    for feature in &engineered_features {
	wtr.write_record(&[
	    feature["cost"].to_string(),
            feature["lead_time"].to_string(),
            feature["cost_per_lead_time"].to_string(),
        ])?;
    }
    wtr.flush()?;
    println!("Engineered features saved to engineered_features.csv");

    //Outlier detection
    detect_outliers(&records);

    //Visualization
    visualize_data_with_trend(&records)?;

    // Calculate and print average cost
    let avg_cost = calculate_average_cost(&records);
    println!("Average Cost per product: {:.2}", avg_cost);

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
fn test_calculate_revenue_analysis() {
    let records = vec![
        Record {
            product: "Product A".to_string(),
            sku: "SKU001".to_string(),
            price: 10.0,
            quantity_sold: 5,
            cost: 0.0,
            lead_time: 0,
            shipping_time: 0,
            shipping_cost: 0.0,
            location: "".to_string(),
            demographic: "".to_string(),
            status: "".to_string(),
        },
        Record {
            product: "Product B".to_string(),
            sku: "SKU002".to_string(),
            price: 20.0,
            quantity_sold: 2,
            cost: 0.0,
            lead_time: 0,
            shipping_time: 0,
            shipping_cost: 0.0,
            location: "".to_string(),
            demographic: "".to_string(),
            status: "".to_string(),
        },
    ];

    let product_revenues = calculate_revenue_analysis(&records);

    assert_eq!(product_revenues.get("Product A"), Some(&50.0));
    assert_eq!(product_revenues.get("Product B"), Some(&40.0));
}

    #[test]
    fn test_calculate_correlation() {
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![1.0, 2.0, 3.0];
        let correlation = calculate_correlation(&x, &y);
    assert!((correlation - 1.0).abs() < 1e-6); // Perfect positive correlation
}

    #[test]
    fn test_detect_outliers() {
    let records = vec![
        Record {
        product: "Product A".to_string(),
        sku: "SKU001".to_string(),
        price: 10.0,
        quantity_sold: 5,
        cost: 100.0,
        lead_time: 10,
        shipping_time: 5,
        shipping_cost: 20.0,
        location: "New York".to_string(),
        demographic: "Adults".to_string(),
        status: "Available".to_string(),
    },
        Record {
	    product: "Product B".to_string(),
            sku: "SKU002".to_string(),
            price: 20.0,
            quantity_sold: 2,
            cost: 200.0,
            lead_time: 15,
            shipping_time: 7,
            shipping_cost: 30.0,
            location: "Los Angeles".to_string(),
            demographic: "Teens".to_string(),
            status: "Unavailable".to_string(),
        },
    ];

    let result = detect_outliers(&records);
    assert!(result.is_ok()); // Check outliers.csv file
}

    #[test]
    fn test_feature_engineering() {
    let records = vec![
        Record {
        product: "Product A".to_string(),
        sku: "SKU001".to_string(),
        price: 10.0,
        quantity_sold: 5,
        cost: 100.0,
        lead_time: 10,
        shipping_time: 5,
        shipping_cost: 20.0,
        location: "New York".to_string(),
        demographic: "Adults".to_string(),
        status: "Available".to_string(),
        },
    ];

    let features = feature_engineering(&records);
    assert_eq!(features[0]["cost_per_lead_time"], 10.0);
}


    #[test]
    fn test_calculate_average_cost() {
        let records = vec![
            Record {
        product: "Product A".to_string(),
        sku: "SKU001".to_string(),
        price: 10.0,
        quantity_sold: 5,
        cost: 100.0,
        lead_time: 10,
        shipping_time: 5,
        shipping_cost: 20.0,
        location: "New York".to_string(),
        demographic: "Adults".to_string(),
        status: "Available".to_string(),
            },
            Record {
        product: "Product B".to_string(),
        sku: "SKU002".to_string(),
        price: 20.0,
        quantity_sold: 2,
        cost: 200.0,
        lead_time: 15,
        shipping_time: 7,
        shipping_cost: 30.0,
        location: "Los Angeles".to_string(),
        demographic: "Teens".to_string(),
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
        price: 10.0,
        quantity_sold: 5,
        cost: 100.0,
        lead_time: 10,
        shipping_time: 5,
        shipping_cost: 20.0,
        location: "New York".to_string(),
        demographic: "Adults".to_string(),
        status: "Available".to_string(),
            },
            Record {
        product: "Product B".to_string(),
        sku: "SKU002".to_string(),
        price: 20.0,
        quantity_sold: 2,
        cost: 200.0,
        lead_time: 15,
        shipping_time: 7,
        shipping_cost: 30.0,
        location: "Los Angeles".to_string(),
        demographic: "Teens".to_string(),
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
        price: 10.0,
        quantity_sold: 5,
        cost: 100.0,
        lead_time: 10,
        shipping_time: 5,
        shipping_cost: 20.0,
        location: "New York".to_string(),
        demographic: "Adults".to_string(),
        status: "Available".to_string(),
            },
            Record {
        product: "Product B".to_string(),
        sku: "SKU002".to_string(),
        price: 20.0,
        quantity_sold: 2,
        cost: 200.0,
        lead_time: 15,
        shipping_time: 7,
        shipping_cost: 30.0,
        location: "Los Angeles".to_string(),
        demographic: "Teens".to_string(),
        status: "Unavailable".to_string(),
            },
        ];

        let result = perform_regression(&records);
        assert!(result.is_ok()); // Check that regression runs successfully
    }
}
