# SQL Cheatsheet: Postgresql 

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


- get the number of days between two datetime columns

```sql
SELECT date_col1, date_col2,
       date_col1::date - date_col2::date AS days_diff
       FROM table_name
```

- using a window function

For instance, to get the sum of an expression over the past 5 days:

```sql
SUM(expr)
       OVER(
              PARTITION BY col1
              ORDER BY extract(epoch from datetime)
              RANGE BETWEEN 3600*24*5 PRECEDING AND CURRENT ROW
       ) AS num_last_30
```


