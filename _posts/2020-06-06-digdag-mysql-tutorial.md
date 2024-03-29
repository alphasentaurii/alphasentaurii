---
layout: post
title:  "Digdag MySQL Tutorial"
date:   2020-06-06 06:06:06 -1800
categories: programming
tags: mysql tutorial
author: Ru Keïn
---

# Digdag MySQL Tutorial
In this project, we'll create a digdag workflow that executes an embulk script for ingesting csv files to a MySQL database. We'll then write SQL queries to prepare and analyze the data. 

## About Embulk and Digdag

Embulk and Digdag are open source libraries for data ingestion and data pipeline orchestration, respectively. These libraries were invented at Treasure Data and are foundational to the Treasure Data product.

The `Digdag` project demonstrates how to use `SQL queries` with `digdag` and `embulk` open source libraries for ingesting and analyzing data. We'll load a MySQL database from CSV files and perform data analysis using automated workflows with digdag via SQL queries.

![GitHub repo size](https://img.shields.io/github/repo-size/alphasentaurii/digdag-mysql)
![GitHub license](https://img.shields.io/github/license/alphasentaurii/digdag-mysql?color=black)

## Prerequisites

Before you begin, ensure you have met the following requirements:

* You have access to `sudo` privileges
* You have a `<Windows/Linux/Mac>` machine.


## Directory structure
.
├── README.md
└── embulk_to_mysql
      └── embulk_to_mysql.dig
      └── tasks
            └── seed_customers.yml
            └── seed_pageviews.yml
            └── config_customers.yml
            └── config_pageviews.yml
      └── queries
            └── create_customers.sql
            └── update_customers.sql
            └── create_pageviews.sql
            └── count_pageviews.sql
            └── top_3_users.sql
      └── data
            └── customers
                  └── customers_1.csv
                  └── customers_2.csv
            └── pageviews
                  └── pageviews_1.csv
                  └── pageviews_2.csv

### Installing Java 8

*Note: these are the steps for installing Java 8 from Oracle on an AWS remote server running Debian 10. If you're using a different environment you will need to adjust accordingly.*

Check which version of Java you're running. If you get an runtime error saying Java is not installed (when you go to run digdag or embulk) follow the steps below.

```bash
$ java -version
```

1. Download the tar file from [Oracle](https://www.oracle.com/java/technologies/javase/javase9-archive-downloads.html): jdk-9.0.4_linux-x64_bin.tar.gz
2. Secure copy (`scp`) the tar file to the remote server
3. Unzip tar file into your JVM directory (you may need to create first)
4. Install Java
5. Set Java directory
6. Check version

```bash
$ sudo -s # root privileges
$ mkdir /usr/lib/jvm # create java folder
$ tar zxvf jre-8u261-linux-x64.tar.gz -C /usr/lib/jvm # unzip tar file into jvm
# install java
$ update-alternatives --install "/usr/bin/java" "java" "/usr/lib/jvm/jre1.8.0_261/bin/java" 1
# set java directory
$ update-alternatives --set java /usr/lib/jvm/jre1.8.0_261/bin/java

$ java -version

java version "1.8.0_261"
Java(TM) SE Runtime Environment (build 1.8.0_261-b12)
Java HotSpot(TM) 64-Bit Server VM (build 25.261-b12, mixed mode)
```

### Install `digdag`

```bash
$ sudo -s # use root privileges 
$ curl -o ~/bin/digdag --create-dirs -L "https://dl.digdag.io/digdag-latest"
$ chmod +x ~/bin/digdag
$ echo 'export PATH="$HOME/bin:$PATH"' >> ~/.bashrc

# Check to make sure `digdag` is installed correctly:
$ digdag --help
```

### Create digdag project

```bash
$ cd ~/
$ mkdir digdag-mysql
$ cd digdag-mysql
$ digdag init embulk_to_mysql
$ cd digdag-mysql/embulk_to_mysql
```

### Install Embulk

```bash
curl --create-dirs -o ~/.embulk/bin/embulk -L "https://dl.embulk.org/embulk-latest.jar"
chmod +x ~/.embulk/bin/embulk
echo 'export PATH="$HOME/.embulk/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

## Install Plugin(s)

```bash
$ embulk gem install embulk-output-mysql
```

### Install MariaDB/MySQL

```bash
$ sudo apt install mariadb-server -y
$ sudo apt install mysql-secure-installation
Enter current password for root (enter for none): [enter]
Set root password? [Y/n] n
 ... skipping.
Remove anonymous users? [Y/n] y
Disallow root login remotely? [Y/n] y
Remove test database and access to it? [Y/n] y
Reload privilege tables now? [Y/n] y
```

#### Create Database

```bash
$ sudo mariadb

Welcome to the MariaDB monitor.  Commands end with ; or \g.
Your MariaDB connection id is 53
Server version: 10.3.22-MariaDB-0+deb10u1 Debian 10

Copyright (c) 2000, 2018, Oracle, MariaDB Corporation Ab and others.

Type 'help;' or '\h' for help. Type '\c' to clear the current input statement.

MariaDB> CREATE DATABASE treasure_data DEFAULT CHARACTER SET utf8 COLLATE utf8_unicode_ci;
Query OK, 1 row affected (0.003 sec)

MariaDB> GRANT ALL ON treasure_data.* TO 'digdag'@'localhost' IDENTIFIED BY 'digdag' WITH GRANT OPTION;
Query OK, 0 rows affected (0.000 sec)

MariaDB> FLUSH PRIVILEGES;
Query OK, 0 rows affected (0.000 sec)

MariaDB> exit
```

#### Test non-root user login

```bash
$ mariadb -u digdag -p
Enter password: 
Welcome to the MariaDB monitor.  Commands end with ; or \g.
Your MariaDB connection id is 54
Server version: 10.3.22-MariaDB-0+deb10u1 Debian 10

Copyright (c) 2000, 2018, Oracle, MariaDB Corporation Ab and others.

Type 'help;' or '\h' for help. Type '\c' to clear the current input statement.

MariaDB [(none)]> SHOW DATABASES;
+---------------------+
| Database            |
+---------------------+
| information_schema  |
| treasure_data       |
+---------------------+
2 rows in set (0.000 sec)

MariaDB [(none)]> exit
```

# Data Ingestion

## Create EMBULK Scripts

*Requirements*

- Files that have a prefix of “customers” should ingest to a table called “customers_tmp”
- Files that have a prefix of “pageviews” should ingest to a table called “pageviews_tmp”
- Ensure that all records from all files are ingested to the appropriate tables. 
- Any timestamps should be ingested to the database as `string/varchar`

### Customers Embulk Script

```bash
$ mkdir tasks
$ nano tasks/seed_customers.yml
```

```bash
# seed_customers.yml
in:
  type: file
  path_prefix: ./data/customers/
  parser:
    charset: UTF-8
    newline: CRLF
    type: csv
    delimiter: ','
    quote:: '"'
    escape: null
    null_string: 'NULL'
    skip_header_lines: 1
    columns:
    - {name: user_id, type: string}
    - {name: first_name, type: string}
    - {name: last_name, type: string}
    - {name: job_title, type: string}
out:
  type: mysql
  host: localhost
  user: digdag
  password: digdag
  database: treasure_data
  table: customers_tmp
  mode: insert
```

### Pageviews Embulk Script

```bash
$ nano tasks/seed_pageviews.yml
```

```bash
# seed_pageviews.yml
in:
  type: file
  path_prefix: ./data/pageviews/
  parser:
    charset: UTF-8
    newline: CRLF
    type: csv
    delimiter: ','
    quote:: '"'
    escape: null
    null_string: 'NULL'
    skip_header_lines: 1
    columns:
    - {name: user_id, type: string}
    - {name: url, type: string}
    - {name: user_agent, type: string}
    - {name: timestamp, type: string, format: varchar}
out:
  type: mysql
  host: localhost
  user: digdag
  password: digdag
  database: treasure_data
  table: pageviews_tmp
  mode: insert
```

---

## SQL Queries

### Customers Table

Creates a new table called `customers` that:
- Includes all columns from customers_tmp
- Parses the “user_agent” column to add a new column called ‘operating_system’ that contains one of the following values ("Windows", "Macintosh", "Linux", or "Other"). 

```bash
$ mkdir queries
$ nano queries/create_customers.sql
```

`create_customers.sql`

```sql
--create_customers.sql--
CREATE TABLE customers
WITH t AS (SELECT user_id, MAX(timestamp) as time 
FROM pageviews_tmp 
GROUP BY user_id)
, s AS (SELECT p.user_id, p.user_agent, p.timestamp 
  FROM pageviews_tmp p 
  JOIN t ON p.user_id = t.user_id 
  AND p.timestamp = t.time) 
SELECT c.user_id, c.first_name, c.last_name, c.job_title, s.user_agent AS operating_system 
FROM customers_tmp c 
JOIN s ON c.user_id = s.user_id
```

`update_customers.sql`

```bash
$ nano queries/update_customers.sql
```

```sql
--update_customers.sql
UPDATE customers 
SET operating_system = 'Macintosh' 
WHERE operating_system LIKE '%Mac%';

UPDATE customers 
SET operating_system = 'Linux' 
WHERE operating_system LIKE '%X11%';

UPDATE customers 
SET operating_system = 'Windows' 
WHERE operating_system LIKE '%Windows%';

UPDATE customers 
SET operating_system = 'Other'
WHERE operating_system NOT REGEXP 'Macintosh|Linux|Windows';
```

### Pageviews Table

Creates a new table called `pageviews` that:
- Includes all columns from pageviews_tmp
- Excludes all records where job_title contains “Sales”

```bash
$ nano queries/create_pageviews.sql
```

`create_pageviews.sql`
```sql
--create_pageviews.sql
CREATE TABLE pageviews 
SELECT * FROM pageviews_tmp
WHERE user_id IN (
    SELECT user_id
		FROM customers_tmp
		WHERE job_title NOT LIKE '%Sales%');
```

### Count Pageviews

Returns the total number of pageviews from users who are browsing with a Windows operating system or have “Engineer” in their job title.

```bash
$ nano queries/count_pageviews.sql
```

`count_pageviews.sql`

```sql
--count_pageviews.sql--
SELECT COUNT(url) AS total_views 
FROM pageviews 
WHERE user_id 
IN (
  SELECT user_id 
  FROM customers 
  WHERE operating_system = 'Windows' 
  OR job_title LIKE '%Engineer%'
  )
```

Returns:

```bash
+-------------+
| total_views |
+-------------+
|        456  |
+-------------+
1 row in set (0.009 sec)
```

### Top 3 Users and Last Page Viewed

Returns top 3 user_id’s (ranked by total pageviews) who have viewed a web page with a “.gov” domain extension and the url of last page they viewed.

```bash
$ nano queries/top_3_users.sql
```

`top_3_users.sql`

```sql
--top_3_users.sql
WITH p2 AS (
	SELECT user_id, max(timestamp) last_timestamp 
	FROM pageviews 
	WHERE user_id 
    IN (
        SELECT user_id 
        FROM pageviews 
        WHERE url LIKE '%.gov%'
        ) 
	GROUP BY user_id 
	ORDER BY COUNT(url) DESC 
    LIMIT 3)
SELECT user_id, url last_page_viewed 
FROM pageviews 
WHERE user_id 
IN (
    SELECT user_id 
	FROM p2 
	WHERE timestamp=last_timestamp
    )
ORDER BY timestamp DESC;
```

Returns:

```bash
+--------------------------------------+--------------------------------------------+
| user_id                              | last_page_viewed                           |
+--------------------------------------+--------------------------------------------+
| 5d9b8515-823e-49b8-ad44-5c91ef23462f | https://microsoft.com/morbi/porttitor.aspx |
| 6cf36c9e-1fa7-491d-a6e1-9c785d68a3d0 | http://nps.gov/quis/odio/consequat.json    |
| 752119fa-50dc-4011-8f13-23aa8d78eb18 | http://goo.ne.jp/nunc.html                 |
+--------------------------------------+--------------------------------------------+
3 rows in set (0.011 sec)
```

## Write a digdag workflow

```bash
$ nano embulk_to_mysql.dig
```

```bash
# embulk_to_mysql.dig
timezone: UTC

_export:
  workflow_name: "embulk_to_mysql"
  start_msg:     "digdag ${workflow_name} start"
  end_msg:       "digdag ${workflow_name} finish"
  error_msg:     "digdag ${workflow_name} error"
  host: localhost
  port: 3306
  user: digdag
  password: digdag
  database: td_coding_challenge
  strict_transaction: false
  q1: queries/page_count.txt
  q2: queries/top_users.txt

+start:
  echo>: ${start_msg}

# Data Preparation

+embulk_guess:
  _parallel: true

  +guess_customers:
    sh>: embulk guess tasks/seed_customers.yml -o tasks/config_customers.yml

  +guess_pageviews:
    sh>: embulk guess tasks/seed_pageviews.yml -o tasks/config_pageviews.yml

+embulk_run:
  _parallel: true
  
  +config_customers:
    sh>: embulk run tasks/config_customers.yml

  +config_pageviews:
    sh>: embulk run tasks/config_pageviews.yml

# Data Ingestion

+create_tables:
  +create_customers:
    sh>: mysql -u${user} -p${password} ${database} < queries/create_customers.sql

  +update_customers:
    sh>: mysql -u${user} -p${password} ${database} < queries/update_customers.sql

  +create_pageviews:
    sh>: mysql -u${user} -p${password} ${database} < queries/create_pageviews.sql

# Data Analysis
+analysis:
  _parallel: true
  
  +count_pageviews:
    sh>: mysql -u${user} -p${password} ${database} < queries/count_pageviews.sql > ${q1}
  
  +top_3_users:
    sh>: mysql -u${user} -p${password} ${database} < queries/top_3_users.sql > ${q2}

# Print Results
+output:
  _parallel: true

  +q1:
    sh>: cat ${q1}
  +q2:
    sh>: cat ${q2}

# End of Workflow
+end:
  echo>: ${end_msg}

_error:
  echo>: ${error_msg}
```

## Running the Digdag workflow

```bash
# If this isn't your first time running the workflow, use the --rerun flag 
$ digdag secrets --local --set mysql.password=digdag
$ digdag run embulk_to_mysql.dig --rerun -O log/task
```

# Contact
If you want to contact me you can reach me at rukeine@gmail.com.

# License
This project uses the following license: MIT License.



```python
                       
           /\    _       _                           _                      *  
/\_/\_____/  \__| |_____| |_________________________| |___________________*___
[===]    / /\ \ | |  _  |  _  | _  \/ __/ -__|  \| \_  _/ _  \ \_/ | * _/| | |
 \./    /_/  \_\|_|  ___|_| |_|__/\_\ \ \____|_|\__| \__/__/\_\___/|_|\_\|_|_|
                  | /             |___/        
                  |/   
```