---
layout: post
title:  "Digdag PostgreSQL Tutorial"
date:   2020-07-26 11:11:11 -1800
categories: datascience
---

# Digdag PostgreSQL Tutorial
In this project, we'll create a `digdag` workflow that executes an `embulk script` for ingesting csv files to a `postgresql` database. We'll then write SQL queries to prepare and analyze the data.

## About Embulk and Digdag

Embulk and Digdag are open source libraries for data ingestion and data pipeline orchestration, respectively. These libraries were invented at Treasure Data and are foundational to the Treasure Data product.

![GitHub repo size](https://img.shields.io/github/repo-size/hakkeray/digdag)
![GitHub license](https://img.shields.io/github/license/hakkeray/digdag?color=black)

The `Digdag` project demonstrates how to use `SQL queries` with `digdag` and `embulk` open source libraries for ingesting and analyzing data. We'll load a postgresql database from CSV files and perform data analysis using automated workflows with digdag via SQL queries.

## Prerequisites

Before you begin, ensure you have met the following requirements:

* You have a `<Windows/Linux/Mac>` machine.
* You have access to `sudo` privileges
* You have installed `Java` version 8
* You have installed the digdag and embulk open source tools
* You have postgresql installed and configured

## Running the Digdag Project

To run this project locally, follow these steps:

In the command line/terminal:

```bash
$ git clone https://github.com/hakkeray/digdag
$ cd digdag/embulk_to_mysql
$ digdag run embulk_to_mysql.dig --rerun -O log/task
```

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

### Installing Java RE

Check which version of Java you're running. If you get an runtime error saying Java is not installed (when you go to run digdag or embulk) follow the steps below.

```bash
$ java -version
```

*Note: these are the steps for installing Java 9 from Oracle on an AWS remote server running Debian 9. If you're using a different environment you will need to adjust accordingly.

1. Download the tar file from ![Oracle](https://www.oracle.com/java/technologies/javase/javase9-archive-downloads.html): jdk-9.0.4_linux-x64_bin.tar.gz

2. Copy (`scp`) the tar file to the remote server
3. Unzip tar file into your JVM directory (you may need to create first)
4. Install Java
5. Set Java directory
6. Check version

```bash
$ sudo mkdir /usr/lib/jvm

$ sudo tar zxvf jre-8u261-linux-x64.tar.gz -C /usr/lib/jvm

$ sudo update-alternatives --install "/usr/bin/java" "java" "/usr/lib/jvm/jre1.8.0_261/bin/java" 1

$ sudo update-alternatives --set java /usr/lib/jvm/jre1.8.0_261/bin/java

$ java -version
java version "1.8.0_261"
Java(TM) SE Runtime Environment (build 1.8.0_261-b12)
Java HotSpot(TM) 64-Bit Server VM (build 25.261-b12, mixed mode)
```

 *For more in-depth doc on JAVA go here:*
https://docs.datastax.com/en/jdk-install/doc/jdk-install/installOracleJdkDeb.html

### Install `digdag`

```bash
$ sudo -s
$ curl -o ~/bin/digdag --create-dirs -L "https://dl.digdag.io/digdag-latest"
$ chmod +x ~/bin/digdag
$ echo 'export PATH="$HOME/bin:$PATH"' >> ~/.bashrc
```

### Check installation was successful

Check to make sure `digdag` is installed correctly:

```bash
$ digdag --help
```

### Create digdag project

```bash
$ digdag init embulk_to_pg
$ cd embulk_to_pg
```

### Install Embulk

```bash
curl --create-dirs -o ~/.embulk/bin/embulk -L "https://dl.embulk.org/embulk-latest.jar"
chmod +x ~/.embulk/bin/embulk
echo 'export PATH="$HOME/.embulk/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

### Install Postgres

```bash
$ sudo apt install postgresql postgresql-contrib
$ sudo -u postgres psql -c "SELECT version();"

```

The postgres user is created automatically when you install PostgreSQL. This user is the superuser for the PostgreSQL instance and it is equivalent to the MySQL root user.

#### Create Database

Log in to the PostgreSQL server as the postgres user

```bash
$ sudo su - postgres
$ psql
```
*Note: you can use the sudo command to access the prompt without switching users. The postgres user is typically used only from the local host and it is recommended not to set the password for this user:*

```bash
sudo -u postgres psql
```

Create Database and Grant Access Privileges
```bash
sudo su - postgres -c "createuser digdag"
sudo su - postgres -c "createdb td_coding_challenge"

sudo -u postgres psql

> grant all privileges on database johndb to john;
> \q
```

Enable remote access to PostgreSQL server

By default the PostgreSQL server listens only on the local interface 127.0.0.1. To enable remote access to your PostgreSQL server open the configuration file postgresql.conf and add listen_addresses = '*' in the CONNECTIONS AND AUTHENTICATION section.

```bash
sudo vim /etc/postgresql/9.6/main/postgresql.conf
```
```vim
/etc/postgresql/9.6/main/postgresql.conf
#------------------------------------------------------------------------------
# CONNECTIONS AND AUTHENTICATION
#------------------------------------------------------------------------------

# - Connection Settings -

listen_addresses = '*'     # what IP address(es) to listen on;

```
```bash
$ sudo service postgresql restart
```

Verify changes

```bash
ss -nlt | grep 5432
```
As you can see from the output above the PostgreSQL server is listening on all interfaces (0.0.0.0).

```bash
LISTEN   0         128                 0.0.0.0:5432             0.0.0.0:*
LISTEN   0         128                    [::]:5432                [::]:*
```

The last step is to configure the server to accept remote connections by editing the pg_hba.conf file.

```bash
/etc/postgresql/9.6/main/pg_hba.conf
```

```bash
# TYPE  DATABASE        USER            ADDRESS                 METHOD

# The user jane will be able to access all databases from all locations using a md5 password
host    all             jane            0.0.0.0/0                md5

# The user jane will be able to access only the janedb from all locations using a md5 password
host    janedb          jane            0.0.0.0/0                md5

# The user jane will be able to access all databases from a trusted location (192.168.1.134) without a password
host    all             jane            192.168.1.134            trust
```

## Install Plugin(s)

```bash
$ embulk gem install embulk-input-postgresql
$ embulk gem install embulk-output-postgresql
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
$ nano seed_customers.yml
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
  database: td_coding_challenge
  table: customers_tmp
  mode: insert
```

### Pageviews Embulk Script

```bash
$ nano seed_pageviews.yml
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
  database: td_coding_challenge
  table: pageviews_tmp
  mode: insert
```

---

## SQL Queries

### Customers Table

Creates a new table called `customers` that:
- Includes all columns from customers_tmp
- Parses the “user_agent” column to add a new column called ‘operating_system’ that contains one of the following values ("Windows", "Macintosh", "Linux", or "Other"). 

`create_customers.sql`

```sql
--create_customers.sql
CREATE TABLE customers 
SELECT c.user_id, c.first_name, c.last_name, c.job_title, p.user_agent AS operating_system 
FROM pageviews_tmp p 
JOIN customers_tmp c 
ON p.user_id = c.user_id 
GROUP BY user_id;
```

`update_customers.sql`

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

`create_pageviews.sql`
```sql
--create_pageviews.sql
CREATE TABLE pageviews 
SELECT * FROM pageviews_tmp
WHERE user_id IN 
(SELECT user_id
FROM customers
WHERE job_title NOT LIKE '%Sales%');
```

### Count Pageviews

Returns the total number of pageviews from users who are browsing with a Windows operating system or have “Engineer” in their job title.

`count_pageviews.sql`

```sql
--count_pageviews.sql--
SELECT COUNT(url) AS total_views 
FROM pageviews 
WHERE user_id 
IN 
(SELECT user_id 
FROM customers 
WHERE operating_system = 'Windows' 
OR job_title LIKE '%Engineer%');
```

Returns:

```bash
+-------------+
| total_views |
+-------------+
|        576  |
+-------------+
1 row in set (0.009 sec)
```

### Top 3 Users and Last Page Viewed

Returns top 3 user_id’s (ranked by total pageviews) who have viewed a web page with a “.gov” domain extension and the url of last page they viewed.

`top_3_users.sql`

```sql
--top_3_users.sql
WITH p2 AS(
SELECT user_id, max(timestamp) last_timestamp 
FROM pageviews 
WHERE user_id IN 
(SELECT user_id 
FROM pageviews 
WHERE url LIKE '%.gov%') 
GROUP BY user_id 
ORDER BY COUNT(url) DESC 
LIMIT 3) 
SELECT user_id, url last_page_viewed 
FROM pageviews 
WHERE user_id IN 
(SELECT user_id 
FROM p2 
WHERE timestamp=last_timestamp)
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
$ digdag init embulk_to_mysql.dig
$ cd embulk_to_mysql.dig
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
  postgresql:
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
    pg>: queries/create_customers.sql
    _preview: true

  +update_customers:
    pg>: queries/update_customers.sql
    _preview: true

  +create_pageviews:
    pg>: queries/create_pageviews.sql
    _preview: true

# Data Analysis
+analysis:
  _parallel: true
  
  +count_pageviews:
    pg>: queries/count_pageviews.sql > ${q1}
    _preview: true
  
  +top_3_users:
    pg>: queries/top_3_users.sql > ${q2}
    _preview: true

# End of Workflow
+end:
  echo>: ${end_msg}

_error:
  echo>: ${error_msg}
```

## Run Digdag workflow

```bash
# If this isn't your first time running the workflow, use the --rerun flag 
$ sudo -s
$ digdag secrets --local --set pg.password=digdag
$ digdag run embulk_to_pg.dig --rerun -O log/task
```

# Contact
If you want to contact me you can reach me at rukeine@gmail.com.

# License
This project uses the following license: MIT License.
#         _ __ _   _
#  /\_/\ | '__| | | |
#  [===] | |  | |_| |
#   \./  |_|   \__,_|

