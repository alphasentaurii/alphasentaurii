---
layout: post
title:  "Digdag PostgreSQL Tutorial"
date:   2020-07-06 07:07:07 -1800
categories: programming
tags: postgresql tutorial
author: Ru Keïn
---

The `Digdag Postgres` project demonstrates how to use `SQL queries` with `digdag` and `embulk` open source libraries for automation of ingesting and analyzing data using a PostgreSQL database.

## About Embulk and Digdag

Embulk and Digdag are open source libraries for data ingestion and data pipeline orchestration, respectively. These libraries were invented at Treasure Data and are foundational to the Treasure Data product.

![GitHub repo size](https://img.shields.io/github/repo-size/alphasentaurii/digdag-postgres)
![GitHub license](https://img.shields.io/github/license/alphasentaurii/digdag-postgres?color=black)

## Prerequisites

Before you begin, ensure you have met the following requirements:

* You have a `<Windows/Linux/Mac>` machine.
* You have access to `sudo` privileges

## Directory structure

This tutorial walks you through the creation of all necessary files and folders, as well as the installation and configuration steps for the pre-requisites such as JAVA 8 and postgreSQL.

The final directory structure will look like this (alternatively you can download and run the project with everything already done for you from my [digdag-postgres](https://github.com/alphasentaurii/digdag-postgres) repo on Github.):

.
├── README.md
└── embulk_to_pg
      └── embulk_to_pg.dig
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
$ sudo mkdir /usr/lib/jvm

$ sudo tar zxvf jre-8u261-linux-x64.tar.gz -C /usr/lib/jvm

$ sudo update-alternatives --install "/usr/bin/java" "java" "/usr/lib/jvm/jre1.8.0_261/bin/java" 1

$ sudo update-alternatives --set java /usr/lib/jvm/jre1.8.0_261/bin/java

$ java -version

java version "1.8.0_261"
Java(TM) SE Runtime Environment (build 1.8.0_261-b12)
Java HotSpot(TM) 64-Bit Server VM (build 25.261-b12, mixed mode)
```

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

### Install Embulk

```bash
$ curl --create-dirs -o ~/.embulk/bin/embulk -L "https://dl.embulk.org/embulk-latest.jar"
$ chmod +x ~/.embulk/bin/embulk
$ echo 'export PATH="$HOME/.embulk/bin:$PATH"' >> ~/.bashrc
$ source ~/.bashrc
```

## Install Plugins

```bash
$ embulk gem install embulk-input-postgresql
$ embulk gem install embulk-output-postgresql
```

### Install Postgres

```bash
$ sudo apt install postgresql postgresql-contrib
$ sudo -u postgres psql -c "SELECT version();"
```

The postgres user is created automatically when you install PostgreSQL. This user is the superuser for the PostgreSQL instance and it is equivalent to the MySQL root user.

### Create User and Database

Use the `sudo` command to access the prompt without switching users. The postgres user is typically used only from the local host and it is recommended not to set the password for this user:

```bash
root:~/digdag-postgres/embulk_to_pg# su - postgres
postgres:~$ createuser --interactive --pwprompt
...
Enter name of role to add: digdag
Enter password for new role: 
Enter it again: 
Shall the new role be a superuser? (y/n) y
...
$ sudo -u postgres psql
...
postgres> createdb td_coding_challenge
postgres> GRANT ALL privileges ON DATABASE td_coding_challenge TO digdag;
postgres> \q
```

Enable remote access to PostgreSQL server

By default the PostgreSQL server listens only on the local interface 127.0.0.1. To enable remote access to your PostgreSQL server open the configuration file postgresql.conf and add listen_addresses = '*' in the CONNECTIONS AND AUTHENTICATION section.

```bash
$ sudo vim /etc/postgresql/9.6/main/postgresql.conf

`i` # insert
`esc :w` # write changes
`:q` # close file
# ...
# /etc/postgresql/11/main/postgresql.conf
#------------------------------------------------------------------------------
# CONNECTIONS AND AUTHENTICATION
#------------------------------------------------------------------------------

# - Connection Settings -

listen_addresses = '*'     # what IP address(es) to listen on;

```

Restart pg

```bash
$ sudo service postgresql restart
```

Verify changes

```bash
$ ss -nlt | grep 5432
```
As you can see from the output above the PostgreSQL server is listening on all interfaces (0.0.0.0).

```bash
LISTEN   0         128                 0.0.0.0:5432             0.0.0.0:*
LISTEN   0         128                    [::]:5432                [::]:*
```

The last step is to configure the server to accept remote connections by editing the pg_hba.conf file.

```bash
$ sudo vim /etc/postgresql/11/main/pg_hba.conf

# TYPE  DATABASE        USER            ADDRESS                 METHOD
host    all             digdag          127.0.0.1/32            md5

# Data Ingestion

### Create digdag project

```bash
$ mkdir digdag-postgres
$ cd digdag-postgres
$ digdag init embulk_to_pg
$ cd embulk_to_pg
```

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
    'quote:': '"'
    escape: null
    null_string: 'NULL'
    skip_header_lines: 1
    columns:
    - {name: user_id, type: string}
    - {name: first_name, type: string}
    - {name: last_name, type: string}
    - {name: job_title, type: string}
    quote: '"'
    trim_if_not_quoted: false
    allow_extra_columns: false
    allow_optional_columns: false
out: 
  type: postgresql
  host: localhost
  user: digdag
  password: digdag
  database: treasure_data
  table: customers_tmp
  mode: insert_direct

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
    'quote:': '"'
    escape: null
    null_string: 'NULL'
    skip_header_lines: 1
    columns:
    - {name: user_id, type: string}
    - {name: url, type: string}
    - {name: user_agent, type: string}
    - {name: timestamp, type: string, format: varchar}
    quote: '"'
    trim_if_not_quoted: false
    allow_extra_columns: false
    allow_optional_columns: false
out:
  type: postgresql
  host: localhost
  user: digdag
  password: digdag
  database: treasure_data
  table: pageviews_tmp 
  mode: insert_direct
  column_options: 
    user_id: {value_type: string}
    url: {value_type: string}
    user_agent: {value_type: string}
    timestamp: {value_type: string, format: varchar}
```

---

## SQL Queries

### Customers Table

Creates a new table called `customers` that:
- Includes all columns from customers_tmp
- Parses the “user_agent” column to add a new column called ‘operating_system’ that contains one of the following values ("Windows", "Macintosh", "Linux", or "Other"). 

`create_customers.sql`

```sql
--create_customers.sql--
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
SELECT * FROM pageviews_tmp
WHERE user_id
IN (SELECT user_id
    FROM customers_tmp
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

We created the directory and initial digdag project already (see above).

```bash
$ nano embulk_to_pg.dig
```

```bash
# embulk_to_pg.dig
timezone: UTC

_export:
  workflow_name: "embulk_to_pg"
  start_msg:     "digdag ${workflow_name} start"
  end_msg:       "digdag ${workflow_name} finish"
  error_msg:     "digdag ${workflow_name} error"
  pg:
    host: 127.0.0.1
    port: 5432
    user: digdag
    password_override: password
    database: treasure_data
    strict_transaction: false

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
    create_table: customers

  +update_customers:
    pg>: queries/update_customers.sql

  +create_pageviews:
    pg>: queries/create_pageviews.sql
    create_table: pageviews

# Data Analysis
+analysis:
  
  +count_pageviews:
    pg>: queries/count_pageviews.sql
    store_last_results: all
  
  +print_q1:
    echo>: ${pg.last_results}

  +top_3_users:
    pg>: queries/top_3_users.sql
    store_last_results: all

  +print_q2:
    echo>: ${pg.last_results}

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
$ digdag run embulk_to_pg.dig -O log/task

# Note: If this isn't your first time running the workflow, use the --rerun flag:*
$ digdag run embulk_to_pg.dig --rerun -O log/task

# You can also have the workflow log to a text file instead of the command line
# Be patient as the workflow takes a while and it will appear as if nothing is happening until you see the success message print out. 

$ digdag run embulk_to_pg.dig --rerun -O log/task > log.txt
```
