from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.ml.functions import vector_to_array
import os
import pandas as pd

def create_spark_session():
    spark = SparkSession.builder \
        .appName("CreditCardFraudDetection") \
        .config("spark.driver.memory", "2g") \
        .config("spark.sql.shuffle.partitions", "10") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    return spark

def load_data(spark, path):
    print("Loading data...")
    df = spark.read.csv(path, header=True, inferSchema=True)
    print(f"Total records: {df.count()}")
    print(f"Fraud cases: {df.filter(col('Class') == 1).count()}")
    print(f"Normal cases: {df.filter(col('Class') == 0).count()}")
    return df

def preprocess_data(df):
    print("Preprocessing data...")
    feature_cols = [c for c in df.columns if c != 'Class']

    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features_raw")
    df = assembler.transform(df)

    scaler = StandardScaler(inputCol="features_raw", outputCol="features_scaled",
                            withStd=True, withMean=True)
    scaler_model = scaler.fit(df)
    df = scaler_model.transform(df)

    # Convert vector back to individual columns
    df = df.withColumn("features_array", vector_to_array("features_scaled"))
    for i, c in enumerate(feature_cols):
        df = df.withColumn(c + "_scaled", col("features_array")[i])

    scaled_cols = [c + "_scaled" for c in feature_cols]
    df = df.select(scaled_cols + [col("Class").alias("label")])

    print("Preprocessing complete!")
    return df

def save_data(df, output_path):
    print("Saving processed data...")
    os.makedirs(output_path, exist_ok=True)

    # Convert to pandas and save — bypasses winutils/Hadoop native issue on Windows
    print("Converting to pandas...")
    pandas_df = df.toPandas()

    train_df = pandas_df.sample(frac=0.8, random_state=42)
    test_df = pandas_df.drop(train_df.index)

    train_df.to_csv(output_path + "train.csv", index=False)
    test_df.to_csv(output_path + "test.csv", index=False)

    print(f"Train size: {len(train_df)}")
    print(f"Test size: {len(test_df)}")
    print(f"✅ Saved to {output_path}")

if __name__ == "__main__":
    spark = create_spark_session()
    df = load_data(spark, "data/creditcard.csv")
    df = preprocess_data(df)
    save_data(df, "data/processed/")
    spark.stop()
    print("✅ Preprocessing complete!")