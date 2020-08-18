---
layout: post
title:  "Neuroimaging API for Machine Learning"
date:   2020-08-10 11:11:11 -1800
categories: datascience
---

* Create a notebook instance
* Prepare the data
* Train the model to learn from the data
* Deploy the model
* Evaluate your ML model's performance *



Application to another application:

Send request (with some info/data)
Get response
data
service
Examples include:

Financial transactions
Posting to Twitter
Controlling IOT
Always a software-to-software interaction

Typical way of getting data (usually JSON or XML)

Access Permissions

User allowed to ask?
API Call/Request

Code used to make API call to implement complicated tasks/features
Methods: what questions can we ask?
Parameters: more info to be sent
Repsonse

Result of request


Client
Web Example

Defines what user sees, so it must:

Define the page visuals
Methods for responding to user interactions
Server
Web Example

Listens to requests (through HTTP):

Authenticates client/user
Looks at requests
Performs actions/processes needed
Sends response to client (possibly with more data)


# Example

We can use requests library to get web page form data, files, and parameters 

```python
import requests

# Getting response from request
resp = requests.get('neurovault.org/api/collections/?DOI=10.1016/j.neurobiolaging.2012.11.002')
type(resp)

# requests.models.Response

requests.codes.ok
# 200

print('Response Code:', resp.status_code)
print('Is it OK?', resp.status_code == requests.codes.ok)

# Response Code: 200
# Is it OK? True
```

# Response Components

```python
# Full HTML Doc
from pprint import pprint
# print(resp.text[:1000])

# Headers
pprint(resp.headers)

# Convert to a dictionary from `requests.structures.CaseInsensitiveDict`
headers = dict(resp.headers)
# {'Date': 'Tue, 05 Nov 2019 17:15:34 GMT', 'Expires': '-1', 'Cache-Control': 'private, max-age=0', 'Content-Type': 'text/html; charset=ISO-8859-1', 'P3P': 'CP="This is not a P3P policy! See g.co/p3phelp for more info."', 'Content-Encoding': 'gzip', 'Server': 'gws', 'X-XSS-Protection': '0', 'X-Frame-Options': 'SAMEORIGIN', 'Set-Cookie': '1P_JAR=2019-11-05-17; expires=Thu, 05-Dec-2019 17:15:34 GMT; path=/; domain=.google.com; SameSite=none, NID=190=Ua-DtcriEneGu6FdMGvevh3Ce6POTpJVN-YyZGXyhJ57WCdWL3KLwnsbhfXostgvG3viaO21MzjJ1p8hHEEEC-k-i7ecTzCXgIHuZC6Klcpypw4ArGSl7sBLNYpeJL_cto2Mt4O0NFWU9XAorz9sQ60eGVMCfvldn0RRPS2iB2c; expires=Wed, 06-May-2020 17:15:34 GMT; path=/; domain=.google.com; HttpOnly', 'Transfer-Encoding': 'chunked'}

print(headers['Date']) # Date response was sent
print(headers['Server']) # Server type
```

```python
# Passing parameters

credentials = {'user_name': ru, 'password': 'password'}
r = requests.get('', params = credentials)

# Note we can only do this becasue r.text() is in JSON format
results = r.json()

# Don't show your IP
results['origin'] = None

print(r.url)
display(results)

"""
http://httpbin.org/get?user_name=luigi&password=i%3C3peach
{'args': {'password': 'i<3peach', 'user_name': 'luigi'},
 'headers': {'Accept': '*/*',
  'Accept-Encoding': 'gzip, deflate',
  'Host': 'httpbin.org',
  'User-Agent': 'python-requests/2.21.0'},
 'origin': None,
 'url': 'https://httpbin.org/get?user_name=luigi&password=i<3peach'
"""
```

# HTTP Post

Allows multiple requests to be sent simultaneously

```python
from google.colab import drive
drive.mount('/content/drive')

```

Example:

```python
filepath_stroke = '/content/drive/My Drive/Colab Notebooks/data/train/imgA.png'
filepath_normal = '/content/drive/My Drive/Colab Notebooks/data/train/imgB.jpg'

url = 'http://httpbin.org/post'
file_list = [
    ('image', ('imgA.png', open(filepath_stroke, 'rb'), 'image/png')),
    ('image', ('imgB.png', open(filepath_normal, 'rb'), 'image/jpg'))
]

r = requests.post(url, files=file_list)
print(r.text)
```

# OAuth (Open Authorization)

Most common form of authorization for large datasets. Allows access without password (authentication separate from authorization).

- Get credentials & authorize application (before OAuth)
- Authorize permissions requested
- Redirect use back w/ authorization code
- Aquisition user "recieves" access token


```python

import requests
import json
import pandas as pd

tokens = {
        'lifx' : {
            'token_name': 'Lifx',
            'token': 'c33cf42e79aaf8afc8b647e13b07ff9fe668587c41c722ae6896462f835190ab',
        }
}

# Specific to today
token = tokens['lifx']['token']

headers = {
    "Authorization": "Bearer %s" % token,
}

response = requests.get('https://api.lifx.com/v1/lights/all', headers=headers)

lights = response.json()
display(lights)


"""

[{'brightness': 1,
  'color': {'hue': 0, 'kelvin': 3500, 'saturation': 0},
  'connected': True,
  'effect': 'OFF',
  'group': {'id': '3d5a822c35807538baf71c686df1f22e', 'name': 'Room 2'},
  'id': 'd073d533cbad',
  'label': "Mom's Room",
  'last_seen': '2019-11-05T17:24:16Z',
  'location': {'id': '00fbb492ce852678eda5704d3470229d', 'name': 'My Home'},
  'power': 'on',
  'product': {'capabilities': {'has_chain': False,
    'has_color': False,
    'has_ir': False,
    'has_matrix': False,
    'has_multizone': False,
    'has_variable_color_temp': True,
    'max_kelvin': 4000,
    'min_kelvin': 1500},
   'company': 'LIFX',
   'identifier': 'lifx_mini_day_and_dusk',
   'name': 'LIFX Mini Day and Dusk'},
  'seconds_since_seen': 0,
  'uuid': '02e3bd5b-ecb8-4bb0-b6e5-b648392f16ec'},
 {'brightness': 1,
  'color': {'hue': 0, 'kelvin': 3500, 'saturation': 0.003997863736934462},
  'connected': False,
  'effect': 'OFF',
  'group': {'id': 'c390740109f7aae4526905966a30a803', 'name': 'Room 1'},
  'id': 'd073d5348133',
  'label': "Matt's Room",
  'last_seen': '2019-05-25T23:56:09Z',
  'location': {'id': '00fbb492ce852678eda5704d3470229d', 'name': 'My Home'},
  'power': 'off',
  'product': {'capabilities': {'has_chain': False,
    'has_color': True,
    'has_ir': False,
    'has_matrix': False,
    'has_multizone': False,
    'has_variable_color_temp': True,
    'max_kelvin': 9000,
    'min_kelvin': 2500},
   'company': 'LIFX',
   'identifier': 'lifx_mini2',
   'name': 'LIFX Mini'},
  'seconds_since_seen': 14146087,
  'uuid': '027d2582-b8f5-45f6-835b-8c6660b6571c'}]
"""
```

```python
lights_df = pd.DataFrame.from_dict(lights)

for light in lights:
    print(light['label'])
```

```python

# Power ON
payload = {
  "states": [
    {
        "selector" : str(lights[1]['id']),
        "power": "on"
    }
  ]
}
response = requests.put('https://api.lifx.com/v1/lights/states', data=json.dumps(payload), headers=headers)
pprint(response.content)

# Power OFF
payload = {
  "states": [
    {
        "selector" : str(lights[1]['id']),
        "power": "off"
    }
  ]
}

response = requests.put('https://api.lifx.com/v1/lights/states', data=json.dumps(payload), headers=headers)
print(response.content)
```


# Using Amazon S3 Command Line Interface REST API

- create IAM user
- set permissions
- create bucket
- get object from bucket

* _Bucket_ – A top-level Amazon S3 folder.

* _Prefix_ – An Amazon S3 folder in a bucket.

* _Object_ – Any item that's hosted in an Amazon S3 bucket.

# Install AWS CLI
```bash
# Macintosh
$ curl "https://awscli.amazonaws.com/AWSCLIV2.pkg" -o "AWSCLIV2.pkg"
  % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                 Dload  Upload   Total   Spent    Left  Speed
100 20.9M  100 20.9M    0     0  16.9M      0  0:00:01  0:00:01 --:--:-- 16.9M

$ sudo installer -pkg AWSCLIV2.pkg -target / #/usr/local/bin/
installer: Package name is AWS Command Line Interface
installer: Installing at base path /
installer: The install was successful.


# verify installation
$ which aws
/usr/local/bin/aws # Mojave
/Users/hakkeray/.local/bin/aws # Sierra

# check version
$ aws --version
aws-cli/2.0.23 Python/3.7.4 Darwin/18.7.0 botocore/2.0.0

# Linux
$ curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
$ unzip awscliv2.zip
$ sudo ./aws/install

# verify
$ aws --version
aws-cli/2.0.23 Python/3.7.4 Linux/4.14.133-113.105.amzn2.x86_64 botocore/2.0.0
```
# Create bucket

```bash
# Create a bucket
$ aws s3 mb <target> [--options]
$ aws s3 mb s3://bucket-name
```

