from google.cloud import bigquery

PROJECT_ID = "gen-lang-client-0915147620"
DATASET_ID = "m5_raw"
TABLE_ID = "sales_train_validation"

FILE_PATH = "data/raw/sales_train_validation.csv"

client = bigquery.Client(project=PROJECT_ID)

table_ref = f"{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}"

job_config = bigquery.LoadJobConfig(
    source_format=bigquery.SourceFormat.CSV,
    skip_leading_rows=1,
    autodetect=True,
    write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
)

print("Starting upload...")
print(f"File: {FILE_PATH}")
print(f"Destination: {table_ref}")

with open(FILE_PATH, "rb") as source_file:
    job = client.load_table_from_file(
        source_file,
        table_ref,
        job_config=job_config,
    )

print("Upload started. Waiting for BigQuery...")

job.result()

table = client.get_table(table_ref)

print("Upload completed successfully!")
print(f"Rows: {table.num_rows}")
print(f"Columns: {len(table.schema)}")
print(f"Table: {table.full_table_id}")