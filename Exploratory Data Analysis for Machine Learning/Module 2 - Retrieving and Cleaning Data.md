# Module 2 — Retrieving and Cleaning Data

> Exploratory Data Analysis for Machine Learning — IBM / Coursera

## Retrieving and Cleaning Data

### Reading CSV files

Comma-separated (CSV) files consist of rows of data, separated by commas. In pandas, CSV files can typically be read using just a few lines of code:

```python
import pandas as pd
filepath = 'data/iris_data.csv'

data = pd.read_csv(filepath)
print(data.head())
```

**Useful arguments:**

```python
# Different delimiters — tab-separated file (.tsv):
data = pd.read_csv(filepath, sep='\t')

# Different delimiters — space-separated file:
data = pd.read_csv(filepath, delim_whitespace=True)

# Don't use first row for column names:
data = pd.read_csv(filepath, header=None)

# Specify column names:
data = pd.read_csv(filepath, names=['name1', 'name2'])
```

### JSON files

A standard way to store data across platforms. JSON files are very similar in structure to Python dictionaries.

Reading JSON files into Python:

```python
import pandas as pd
data = pd.read_json(filepath)

# Write dataframe to JSON
data.to_json('outputfile.json')
```

### SQL Databases

Represent a set of relational databases with fixed schemas. There are many types of SQL databases, which function similarly. Examples:

- Microsoft SQL Server
- Postgres
- AWS Redshift
- Oracle DB
- Db2 Family

Reading SQL data:

```python
# SQL Data Imports
import sqlite3 as sq3
import pandas as pd

# Initialize path to SQLite database
path = 'data/classic_rock.db'

# Create connection to SQL database
con = sq3.Connection(path)

# Write query
query = '''SELECT * FROM rock_songs;'''

# Execute query
data = pd.read_sql(query, con)
```

### NoSQL Databases

Not relational, and vary more in structure. Depending on the application, they may perform more quickly or reduce technical overhead. Most NoSQL databases store data in JSON format. Examples:

- **Document databases** — MongoDB, CouchDB
- **Key-value stores** — Riak, Voldemort, Redis
- **Graph databases** — Neo4j, HyperGraph
- **Wide-column stores** — Cassandra, HBase

Reading NoSQL data:

```python
# NoSQL Data Imports
from pymongo import MongoClient

# Create a Mongo connection
con = MongoClient()

# Choose database (con.list_database_names()
# will display available databases)
# db = con.database_name

# Create a cursor object using a query
cursor = db.collection_name.find(query)

# Expand cursor and construct DataFrame
df = pd.DataFrame(list(cursor))
```

### APIs and Cloud Data Access

```python
# UCI Cars data set — url location
data_url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data'

# Read data into pandas
df = pd.read_csv(data_url, header=None)
```

## Data Cleaning

### Learning Goals

- Why data cleaning is important for Machine Learning
- Issues that arise with messy data
- How to identify duplicate or unnecessary data
- Policies for dealing with outliers

### Why is Data Cleaning so important?

Decisions and analytics are increasingly driven by data and models. Key aspects of the Machine Learning workflow depend on cleaned data:

- **Observations** — an instance of the data (usually a point or row in a dataset)
- **Labels** — output variable(s) being predicted
- **Algorithms** — computer programs that estimate models based on available data
- **Features** — information we have for each observation (variables)
- **Model** — hypothesized relationship between observations and data

### The main data problems

- Lack of data
- Too much data
- Bad data

**How can data be messy?**

- Duplicate or unnecessary data
- Inconsistent text and typos
- Missing data
- Outliers
- Data source issues:
  - Multiple systems
  - Different database types
  - On premises, in cloud, and more

**Duplicate or unnecessary data:** Pay attention to duplicate values and research why there are multiple values. It's a good idea to look at the features you're bringing in and filter the data as necessary (be careful not to filter too much if you may use those features later).

### Working with Missing Values

- **Remove the data** — remove the row(s) or columns entirely
- **Impute the data** — replace with substituted values (fill in the missing data with the most common value, the average, etc.)
- **Mask the data** — create a category for missing values

### Outliers

An outlier is an observation in the data that is distant from most other observations.

- Typically, these observations are aberrations and do not accurately represent the phenomenon we are trying to explain through the models.
- If we do not identify and deal with outliers, they can have a significant impact on the model.
- It is important to remember that some outliers are informative and provide insights into the data.

**How to find them?**

- **Plots** — Histogram, Density plot, Box plot
- **Statistics** — Interquartile Range, Standard Deviation
- **Residuals** — Standardized, Deleted, Studentized

**Plots:**

```python
# Histogram
sns.distplot(data, bins=20)

# Boxplot
sns.boxplot(data)
```

**Statistics:**

```python
import numpy as np

# Calculate the interquartile range
q25, q50, q75 = np.percentile(data, [25, 50, 75])
iqr = q75 - q25

# Calculate the min/max limits to be considered an outlier
min_limit = q25 - 1.5 * iqr
max_limit = q75 + 1.5 * iqr

print(min_limit, q25, q50, q75, max_limit)

[x for x in data['Unemployment'] if x > max_limit]
```

**Residuals:** Differences between actual and predicted values of the outcome variable; they represent model failure. Approaches to calculating residuals:

- **Standardized** — residual divided by standard error
- **Deleted** — residual from fitting the model on all data excluding the current observation
- **Studentized** — deleted residuals divided by residual standard error (based on all data, or all data excluding the current observation)

**Policies for outliers:**

- Remove them
- Assign the mean or median value
- Transform the variable
- Predict what the value should be:
  - Using "similar" observations to predict likely values
  - Using regression

---

## Summary / Review

### Retrieving Data

You can retrieve data from multiple sources: SQL databases, NoSQL databases, APIs, and cloud data sources. The two most common formats for delimited flat files are comma-separated (CSV) and tab-separated (TSV). It is also possible to use special characters as separators. SQL represents a set of relational databases with fixed schemas.

### Reading in Database Files

The steps to read in a database file using the `sqlite` library are:

1. Create a path variable that references the path to your database.
2. Create a connection variable that references the connection to your database.
3. Create a query variable that contains the SQL query that reads in the data table.
4. Create an observations variable to assign the `read_sql` function from pandas.
5. Create a tables variable to read in the data from the table `sqlite_master`.

JSON files are a standard way to store data across platforms; their structure is similar to Python dictionaries. NoSQL databases are not relational and vary more in structure — most store data in JSON format.

### Data Cleaning

Data cleaning is important because messy data will lead to unreliable outcomes. Common issues that make data messy: duplicate or unnecessary data, inconsistent data and typos, missing data, outliers, and data source issues.

Common policies to deal with missing data: remove a row with missing columns, impute the missing data, and mask the data by creating a category for missing values.

Common methods to find outliers: through plots, statistics, or residuals. Common policies to deal with outliers: remove them, impute them, use a variable transformation, or use a model that is resistant to outliers.
