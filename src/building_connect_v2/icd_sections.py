import logging
import os
import time
import glob
import shutil
import cProfile
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window


logging.basicConfig(filename='program_log_icd.txt', level=logging.INFO)

def log_time_and_step(step_name):
    logging.info(f"{step_name}: {time.ctime()}")

def load_data(spark):
    log_time_and_step("Loading Data")
    csv_file_path = "data/OASIS_v2.csv"
    df = spark.read.csv(csv_file_path, header=True, inferSchema=True)#.sample(0.1,seed=42)
    return df

def prepare_data(df):
    log_time_and_step("Preparing Data")
    # List of ICD sections
    icd_sections = [
        'A00-B99', 'C00-D49', 'D50-D89', 'E00-E89', 'F01-F99', 'G00-G99', 
        'H00-H59', 'H60-H95', 'I00-I99', 'J00-J99', 'K00-K95', 'L00-L99', 
        'M00-M99', 'N00-N99', 'O00-O9A', 'P00-P96', 'Q00-Q99', 'R00-R99', 
        'S00-T88', 'U00-U85', 'V00-Y99', 'Z00-Z99'
    ]


    output_dir = "data/icd_sections"
    os.makedirs(output_dir, exist_ok=True)

    for section in icd_sections:
        # Filter patients with at least one diagnosis in this section
        df_section = df.filter(F.col(section) > 0)
        
        # Count patients
        count = df_section.count()
        print(f"{section}: {count} patients")
        
        if count == 0:
            print(f"No patients in section {section}, skipping...")
            continue
        
        # Save each section as a single CSV
        temp_path = os.path.join(output_dir, f"{section}_temp")
        final_csv = os.path.join(output_dir, f"OASIS_{section}.csv")
        
        df_section.coalesce(1).write.mode("overwrite").option("header", "true").csv(temp_path)
        
        # Move the part file to the final CSV
        part_file = glob.glob(os.path.join(temp_path, "part-*.csv"))[0]
        shutil.move(part_file, final_csv)
        shutil.rmtree(temp_path)
        
        log_time_and_step(f"CSV for {section} saved at {final_csv}")
    return df



def main():
    start_time = time.time()
    log_time_and_step("Start Program")
    spark = SparkSession.builder \
        .appName("Split") \
        .config("spark.driver.memory", "8g") \
        .config("spark.executor.memory", "8g") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    spark.conf.set("spark.sql.shuffle.partitions", "400")
    spark.conf.set("spark.sql.debug.maxToStringFields", 1000)

    output_dir = "data"
    os.makedirs(output_dir, exist_ok=True)

    df = load_data(spark)
    prepare_data(df)



    spark.stop()  # Stop the Spark session when done
    log_time_and_step("End Program")
    logging.info(f"Total Time: {time.time() - start_time:.2f} seconds")

    # Profile the code

if __name__ == "__main__":
    cProfile.run('main()')
