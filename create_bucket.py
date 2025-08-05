#%%
import os
import pandas as pd
import glob
import logging
import time
from typing import List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(threadName)s] - %(message)s',
    handlers=[
        logging.FileHandler('influxdb_import.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

#%%
# Configuration
INFLUXDB_URL = os.environ["INFLUXDB_URL"]
INFLUXDB_TOKEN = os.environ["INFLUXDB_TOKEN"]
INFLUXDB_ORG = os.environ["INFLUXDB_ORG"]
BUCKET_NAME = "crypto"
PARENT_DIR = "/Users/orentapiero/DATA/binance_klines/"

# Performance settings
BATCH_SIZE = 5000  # Process records in batches
MAX_WORKERS = 6    # Reduced from 12 to avoid overwhelming InfluxDB
MAX_RETRIES = 3
RETRY_DELAY = 1    # seconds

# Thread-safe counters
stats_lock = Lock()
stats = {
    'files_processed': 0,
    'files_failed': 0,
    'total_records': 0,
    'start_time': time.time()
}

def update_stats(files_processed=0, files_failed=0, total_records=0):
    """Thread-safe stats update."""
    with stats_lock:
        stats['files_processed'] += files_processed
        stats['files_failed'] += files_failed
        stats['total_records'] += total_records

def print_final_stats():
    """Print final import statistics."""
    elapsed = time.time() - stats['start_time']
    logger.info(f"""
=== IMPORT COMPLETED ===
Files processed: {stats['files_processed']}
Files failed: {stats['files_failed']}
Total records: {stats['total_records']:,}
Total time: {elapsed:.2f} seconds
Records/second: {stats['total_records']/elapsed:.2f}
=========================""")

#%%
# Setup bucket
def setup_bucket():
    """Create bucket if it doesn't exist."""
    try:
        with InfluxDBClient(url=INFLUXDB_URL, token=INFLUXDB_TOKEN, org=INFLUXDB_ORG) as client:
            buckets_api = client.buckets_api()
            buckets = buckets_api.find_buckets().buckets
            if not any(b.name == BUCKET_NAME for b in buckets):
                buckets_api.create_bucket(bucket_name=BUCKET_NAME, org=INFLUXDB_ORG)
                logger.info(f"Bucket '{BUCKET_NAME}' created.")
            else:
                logger.info(f"Bucket '{BUCKET_NAME}' already exists.")
    except Exception as e:
        logger.error(f"Error setting up bucket: {e}")
        raise

setup_bucket()

#%%
def read_and_validate_csv(csv_path: str) -> Optional[Tuple[pd.DataFrame, str]]:
    """Read and validate CSV file."""
    expected_columns = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
    ]
    
    ticker = os.path.basename(os.path.dirname(csv_path))
    
    try:
        # First, peek at the file to detect headers
        with open(csv_path, 'r') as f:
            first_line = f.readline().strip()
        
        # Check if first line contains headers
        skip_header = False
        if first_line:
            first_value = first_line.split(',')[0].strip()
            if not first_value.isdigit():
                skip_header = True
        
        # Read CSV
        df = pd.read_csv(
            csv_path, 
            header=None if not skip_header else 0,
            names=expected_columns if not skip_header else None,
            skiprows=1 if skip_header and pd.read_csv(csv_path, nrows=1).columns[0] != 'open_time' else 0
        )
        
        # If we detected headers but they're not the expected ones, re-read without headers
        if skip_header and list(df.columns) != expected_columns:
            df = pd.read_csv(csv_path, header=None, names=expected_columns)
        
        # Validate and clean data
        if 'ignore' in df.columns:
            df = df.drop(columns=['ignore'])
        
        # Convert numeric columns
        numeric_columns = [
            "open", "high", "low", "close", "volume", "quote_asset_volume",
            "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "number_of_trades"
        ]
        
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Handle timestamps
        df['open_time'] = pd.to_numeric(df['open_time'], errors='coerce')
        df['close_time'] = pd.to_numeric(df['close_time'], errors='coerce')
        
        # Drop invalid rows
        initial_rows = len(df)
        df = df.dropna()
        
        if len(df) == 0:
            logger.warning(f"No valid data in {csv_path}")
            return None
        
        if len(df) < initial_rows:
            logger.info(f"Dropped {initial_rows - len(df)} invalid rows from {csv_path}")
        
        # Convert timestamps to datetime
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
        df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
        
        return df, ticker
        
    except Exception as e:
        logger.error(f"Error reading {csv_path}: {e}")
        return None

def create_points_batch(df_batch: pd.DataFrame, ticker: str) -> List[Point]:
    """Create batch of InfluxDB points."""
    points = []
    
    for _, row in df_batch.iterrows():
        try:
            point = Point("klines").tag("symbol", ticker)
            
            # Add numeric fields
            numeric_fields = [
                "open", "high", "low", "close", "volume", "quote_asset_volume",
                "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume"
            ]
            
            for field in numeric_fields:
                if field in row and pd.notna(row[field]):
                    point = point.field(field, float(row[field]))
            
            # Add integer field
            if 'number_of_trades' in row and pd.notna(row['number_of_trades']):
                point = point.field("number_of_trades", int(row['number_of_trades']))
            
            # Set timestamp
            point = point.time(row['open_time'], WritePrecision.MS)
            points.append(point)
            
        except Exception as e:
            logger.warning(f"Error creating point for row: {e}")
            continue
    
    return points

def write_batch_with_retry(write_api, points: List[Point]) -> bool:
    """Write batch with retry logic."""
    for attempt in range(MAX_RETRIES):
        try:
            write_api.write(bucket=BUCKET_NAME, org=INFLUXDB_ORG, record=points)
            return True
        except Exception as e:
            logger.warning(f"Write attempt {attempt + 1} failed: {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))  # Exponential backoff
            else:
                logger.error(f"Failed to write batch after {MAX_RETRIES} attempts")
                return False
    return False

def import_csv_to_influxdb(csv_path: str) -> dict:
    """Import single CSV file to InfluxDB."""
    result = {
        'file': csv_path,
        'success': False,
        'records_imported': 0,
        'error': None
    }
    
    try:
        # Read and validate CSV
        csv_data = read_and_validate_csv(csv_path)
        if csv_data is None:
            result['error'] = "Failed to read or validate CSV"
            return result
        
        df, ticker = csv_data
        total_rows = len(df)
        
        # Process in batches
        with InfluxDBClient(url=INFLUXDB_URL, token=INFLUXDB_TOKEN, org=INFLUXDB_ORG) as client:
            write_api = client.write_api(write_options=SYNCHRONOUS)
            
            records_imported = 0
            for i in range(0, total_rows, BATCH_SIZE):
                batch_df = df.iloc[i:i + BATCH_SIZE]
                points = create_points_batch(batch_df, ticker)
                
                if points and write_batch_with_retry(write_api, points):
                    records_imported += len(points)
                else:
                    logger.error(f"Failed to write batch {i//BATCH_SIZE + 1} for {csv_path}")
        
        result['success'] = True
        result['records_imported'] = records_imported
        logger.info(f"Successfully imported {records_imported}/{total_rows} records from {csv_path}")
        
    except Exception as e:
        result['error'] = str(e)
        logger.error(f"Error importing {csv_path}: {e}")
    
    return result

#%%
def main():
    """Main execution function."""
    try:
        # Find all CSV files
        csv_files = glob.glob(os.path.join(PARENT_DIR, "**", "*.csv"), recursive=True)
        
        if not csv_files:
            logger.error(f"No CSV files found in {PARENT_DIR}")
            return
        
        logger.info(f"Found {len(csv_files)} CSV files to process")
        
        # Process files with thread pool
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # Submit all tasks
            future_to_file = {
                executor.submit(import_csv_to_influxdb, csv_path): csv_path 
                for csv_path in csv_files
            }
            
            # Process completed tasks with progress bar
            with tqdm(total=len(csv_files), desc="Processing files") as pbar:
                for future in as_completed(future_to_file):
                    csv_path = future_to_file[future]
                    try:
                        result = future.result()
                        
                        if result['success']:
                            update_stats(files_processed=1, total_records=result['records_imported'])
                            pbar.set_postfix({
                                'processed': stats['files_processed'],
                                'failed': stats['files_failed'],
                                'records': f"{stats['total_records']:,}"
                            })
                        else:
                            update_stats(files_failed=1)
                            logger.error(f"Failed to process {csv_path}: {result.get('error', 'Unknown error')}")
                        
                    except Exception as e:
                        update_stats(files_failed=1)
                        logger.error(f"Exception processing {csv_path}: {e}")
                    
                    pbar.update(1)
        
        print_final_stats()
        
    except KeyboardInterrupt:
        logger.info("Import interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")

if __name__ == "__main__":
    main()

#%%