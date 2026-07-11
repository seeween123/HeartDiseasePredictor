import great_expectations as ge
from typing import Tuple, List


def validate_heart_disease_data(df) -> Tuple[bool, List[str]]:
    """
    Comprehensive data validation for Heart Disease dataset using Great Expectations.
    
    This function implements critical data quality checks that must pass before model training.
    It validates data integrity, business logic constraints, and statistical properties
    that the ML model expects.
    
    """
    print("🔍 Starting data validation with Great Expectations...")
    
    # Convert pandas DataFrame to Great Expectations Dataset
    ge_df = ge.dataset.PandasDataset(df)
    
    # === SCHEMA VALIDATION - ESSENTIAL COLUMNS ===
    print("   📋 Validating schema and required columns...")
    
    required_columns = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "ca",
    "thal",
    "target"
    ]

    for col in required_columns:
        ge_df.expect_column_to_exist(col)
        ge_df.expect_column_values_to_not_be_null(col)

    # === BUSINESS LOGIC VALIDATION ===
    print("   💼 Validating business logic constraints...")
    
    # === NUMERIC RANGE VALIDATION ===
    print("   📊 Validating numeric ranges and business constraints...")

    ge_df.expect_column_values_to_be_in_set("target",[0, 1])

    ge_df.expect_column_values_to_be_between("sex", min_value=0, max_value=1)
    
    ge_df.expect_column_values_to_be_between("fbs", min_value=0, max_value=1)

    ge_df.expect_column_values_to_be_between("ca", min_value=0, max_value=3)

    ge_df.expect_column_values_to_be_between("cp", min_value=1, max_value=4)

    ge_df.expect_column_values_to_be_between("restecg", min_value=0, max_value=2)

    ge_df.expect_column_values_to_be_between("age", min_value=0, max_value=120)

    ge_df.expect_column_values_to_be_between("thalach", min_value=10, max_value=250)

    ge_df.expect_column_values_to_be_in_set("thal", [3, 6, 7])

    ge_df.expect_column_values_to_be_between("exang", min_value=0, max_value=1)

    ge_df.expect_column_values_to_be_between("slope", min_value=1, max_value=3)

    ge_df.expect_column_values_to_be_between(
        "trestbps",
        min_value=50,
        max_value=300,
    )

    ge_df.expect_column_values_to_be_between(
        "chol",
        min_value=50,
        max_value=700,
    )

    ge_df.expect_column_values_to_be_between(
        "oldpeak",
        min_value=0,
        max_value=10,
)
    # === DATA CONSISTENCY CHECKS ===
    #print("   🔗 Validating data consistency...")
    
    
    # === RUN VALIDATION SUITE ===
    print("   ⚙️  Running complete validation suite...")
    results = ge_df.validate()
    
    # === PROCESS RESULTS ===
    # Extract failed expectations for detailed error reporting
    failed_expectations = []
    for r in results["results"]:
        if not r["success"]:
            expectation_type = r["expectation_config"]["expectation_type"]
            failed_expectations.append(expectation_type)
    
    # Print validation summary
    total_checks = len(results["results"])
    passed_checks = sum(1 for r in results["results"] if r["success"])
    failed_checks = total_checks - passed_checks
    
    if results["success"]:
        print(f"✅ Data validation PASSED: {passed_checks}/{total_checks} checks successful")
    else:
        print(f"❌ Data validation FAILED: {failed_checks}/{total_checks} checks failed")
        print(f"   Failed expectations: {failed_expectations}")
    
    return results["success"], failed_expectations
