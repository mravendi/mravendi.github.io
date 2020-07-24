# SQL Cheatsheet: Postgresql 


## SELECT ...
- selecting a column from a table 
```sql
SELECT col_nm FROM table_name LIMIT 10;
```

- selecting all columns from a table
```sql
SELECT * FROM table_name LIMIT 10;
```

- selecting a column with condition
```sql
SELECT col_nm FROM table_name 
WHERE cnd_exp
LIMIT 10
```

- truncate or round double numbers

```sql
SELECT TRUNC(CAST(col_nm AS numeric), 2) 
FROM table_name
LIMIT 10

SELECT ROUND(col_name::numeric,2)    
FROM table_name
LIMIT 10
```


## Timestamps and Datetime
- get the number of days between two datetime columns
```sql
SELECT date_col1, date_col2,
       date_col1::date - date_col2::date AS days_diff
       FROM table_name
```


## window functions

- to get the sum of an expression over the past 5 days:
```sql
SUM(expr)
       OVER(
              PARTITION BY col1
              ORDER BY extract(epoch from datetime)
              RANGE BETWEEN 3600*24*5 PRECEDING AND CURRENT ROW
       ) AS sum_last_5
```


- to get the sum of an expression over the next 5 days:
```sql
SUM(expr)
       OVER(
              PARTITION BY col1
              ORDER BY extract(epoch from datetime)
              RANGE BETWEEN CURRENT ROW AND FOLLOWING 3600*24*5
       ) AS sum_next_5
```


- to get the sum of an expression over a window in the future, define two window functions from current row and then use the subtract to get the difference.



## CREATE

```sql
DROP TABLE IF EXISTS table_name;
CREATE TABLE table_name AS
       -- define the table columns here
       SELECT cols 
       FROM source_table
       WHERE condtitions
       DISTRIBUTED RANDOMLY;
```

## CASE ... WHEN ...

```sql
CASE 
       WHEN cond1 THEN val1 
       WHEN cond2 THEN val2 
       ...
       ELSE valn 
       END
```







