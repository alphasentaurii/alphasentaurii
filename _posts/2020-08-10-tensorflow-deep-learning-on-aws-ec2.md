---
layout: post
title:  "Tensorflow Deep Learning on AWS EC2"
date:   2020-08-10  11:11:11 -1800
categories: datascience
---

Configure `AWS Deep Learning` EC2 Container image for `TensorFlow` model training on a `CPU instance` with `Python 3.6` and run a machine learning model.

# IAM Policy
Navigate to AWS IAM Console and select or create a user. Add the following permissions under `Attach Existing Policies Directly`:
- ECS_FullAccess

Add inline policy and save as `ECR`:

```json
{
       "Version": "2012-10-17",
       "Statement": [
              {
                     "Action": "ecr:*",
                     "Effect": "Allow",
                     "Resource": "*"
              }
       ]
}
```

# Create EC2 Instance on AWS console

Navigate to EC2 console and launch an Amazon EC2 instance.  
- Select AWS Deep Learning Base AMI

I use the Amazon Linux one here because I prefer Debian (it's pretty much Amazon's version of a Debian image) but you can also choose the Ubuntu version.

Select instance type:
- c5.large

Click *LAUNCH* !

Create or select a key pair. You can look up how to do this if you're not sure...

Grab the public DNS (IPv4) from the instance you just created and ssh into it from your local machine:

# SSH into EC2 Instance

```bash
STARFOX $> ssh -L localhost:8888:localhost:8888 -i sophie_rsa ec2-user@ec2-3-236-65-85.compute-1.amazonaws.com

Enter passphrase for key 'sophie_rsa': 
```

```bash
=============================================================================
       __|  __|_  )
       _|  (     /   Deep Learning AMI (Amazon Linux 2) Version 32.0
      ___|\___|___|
=============================================================================

Please use one of the following commands to start the required environment with the framework of your choice:
for MXNet(+Keras2) with Python3 (CUDA 10.1 and Intel MKL-DNN) ____________________________________ source activate mxnet_p36
for MXNet(+Keras2) with Python2 (CUDA 10.1 and Intel MKL-DNN) ____________________________________ source activate mxnet_p27
for MXNet(+AWS Neuron) with Python3 ___________________________________________________ source activate aws_neuron_mxnet_p36
for TensorFlow(+Keras2) with Python3 (CUDA 10.0 and Intel MKL-DNN) __________________________ source activate tensorflow_p36
for TensorFlow(+Keras2) with Python2 (CUDA 10.0 and Intel MKL-DNN) __________________________ source activate tensorflow_p27
for TensorFlow(+AWS Neuron) with Python3 _________________________________________ source activate aws_neuron_tensorflow_p36
for TensorFlow 2(+Keras2) with Python3 (CUDA 10.1 and Intel MKL-DNN) _______________________ source activate tensorflow2_p36
for TensorFlow 2(+Keras2) with Python2 (CUDA 10.1 and Intel MKL-DNN) _______________________ source activate tensorflow2_p27
for TensorFlow 2.3 with Python3.7 (CUDA 10.2 and Intel MKL-DNN) _____________________ source activate tensorflow2_latest_p37
for PyTorch 1.4 with Python3 (CUDA 10.1 and Intel MKL) _________________________________________ source activate pytorch_p36
for PyTorch 1.4 with Python2 (CUDA 10.1 and Intel MKL) _________________________________________ source activate pytorch_p27
for PyTorch 1.6 with Python3 (CUDA 10.1 and Intel MKL) ________________________________ source activate pytorch_latest_p36
for PyTorch (+AWS Neuron) with Python3 ______________________________________________ source activate aws_neuron_pytorch_p36
for Chainer with Python2 (CUDA 10.0 and Intel iDeep) ___________________________________________ source activate chainer_p27
for Chainer with Python3 (CUDA 10.0 and Intel iDeep) ___________________________________________ source activate chainer_p36
for base Python2 (CUDA 10.0) _______________________________________________________________________ source activate python2
for base Python3 (CUDA 10.0) _______________________________________________________________________ source activate python3

To automatically activate base conda environment upon login, run: 'conda config --set auto_activate_base true'
 
Official Conda User Guide: https://docs.conda.io/projects/conda/en/latest/user-guide/
AWS Deep Learning AMI Homepage: https://aws.amazon.com/machine-learning/amis/
Developer Guide and Release Notes: https://docs.aws.amazon.com/dlami/latest/devguide/what-is-dlami.html
Support: https://forums.aws.amazon.com/forum.jspa?forumID=263
For a fully managed experience, check out Amazon SageMaker at https://aws.amazon.com/sagemaker
When using INF1 type instances, please update regularly using the instructions at: https://github.com/aws/aws-neuron-sdk/tree/master/release-notes
=============================================================================

ec2-user ~$ 
```

# Login to Amazon ECR

```bash
ec2-user ~$ $(aws ecr get-login --region us-east-1 --no-include-email --registry-ids 763104351884)

WARNING! Using --password via the CLI is insecure. Use --password-stdin.
WARNING! Your password will be stored unencrypted in /home/ec2-user/.docker/config.json.
Configure a credential helper to remove this warning. See
https://docs.docker.com/engine/reference/commandline/login/#credentials-store

Login Succeeded

```

# Run `TensorFlow` training on `CPU instances` with `Python 3.6`.

We'll run AWS Deep Learning Container images on your EC2 instance using the command below. This command will automatically pull the Deep Learning Container image if it doesn’t exist locally.

```bash
~$ docker run -it 763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-training:1.13-cpu-py36-ubuntu16.04

~$ source activate tensorflow2_latest_p37

# For tensorflow 1:
~$ source activate tensorflow_p36
"""
Installing collected packages: tensorflow
Successfully installed tensorflow-1.15.3
Installation complete.
"""
```

# Example : Run MNIST CNN Model

```bash
~$ git clone https://github.com/fchollet/keras.git

~$ python keras/examples/mnist_cnn.py

"""
2020-08-09 06:19:25.692828: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcudart.so.10.1
Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz
11493376/11490434 [==============================] - 0s 0us/step
x_train shape: (60000, 28, 28, 1)
60000 train samples
10000 test samples
2020-08-09 06:19:40.000744: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcuda.so.1
2020-08-09 06:19:40.516676: E tensorflow/stream_executor/cuda/cuda_driver.cc:314] failed call to cuInit: CUDA_ERROR_NO_DEVICE: no CUDA-capable device is detected
2020-08-09 06:19:40.516729: I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:156] kernel driver does not appear to be running on this host (ip-172-31-79-109.ec2.internal): /proc/driver/nvidia/version does not exist
2020-08-09 06:19:40.517234: I tensorflow/core/platform/cpu_feature_guard.cc:142] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN)to use the following CPU instructions in performance-critical operations:  AVX2 AVX512F FMA
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
2020-08-09 06:19:40.523521: I tensorflow/core/platform/profile_utils/cpu_utils.cc:104] CPU Frequency: 2999995000 Hz
2020-08-09 06:19:40.523688: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x56095e591280 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-08-09 06:19:40.523705: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
2020-08-09 06:19:40.596967: W tensorflow/core/framework/cpu_allocator_impl.cc:81] Allocation of 188160000 exceeds 10% of free system memory.
"""

Epoch 1/12
272/469 [================>.............] - ETA: 38s - loss: 2.2911 - accuracy: 0.1317

```

# Terminate all your resources

Terminate all the resources so you don't get charged

On the Amazon EC2 Console, select Running Instances.
 Select the EC2 instance you created and choose Actions > Instance State > Terminate.
 c. Confirm termination
You will be asked to confirm your termination. Select Yes, Terminate.

Note: This process can take several seconds to complete. Once your instance has been terminated, the Instance State will change to terminated on your EC2 Console.

---

# Using SAGEMAKER

Create Sagemaker instance on AWS Console then open the Jupyter Notebook. Set to Conda/Python3.

# Prepare the Data
Preprocess the data that you need to train your machine learning model.

```python
# import libraries
import boto3, re, sys, math, json, os, sagemaker, urllib.request
from sagemaker import get_execution_role
import numpy as np                                
import pandas as pd                               
import matplotlib.pyplot as plt                   
from IPython.display import Image                 
from IPython.display import display               
from time import gmtime, strftime 
```

# Define AWS environment variables

```python
role = get_execution_role()
prefix = 'sagemaker/DEMO-xgboost-dm'
containers = {'us-west-2': '433757028032.dkr.ecr.us-west-2.amazonaws.com/xgboost:latest',
              'us-east-1': '811284229777.dkr.ecr.us-east-1.amazonaws.com/xgboost:latest',
              'us-east-2': '825641698319.dkr.ecr.us-east-2.amazonaws.com/xgboost:latest',
              'eu-west-1': '685385470294.dkr.ecr.eu-west-1.amazonaws.com/xgboost:latest'} # each region has its XGBoost container
my_region = boto3.session.Session().region_name # set the region of the instance
print("Success - the MySageMakerInstance is in the " + my_region + " region. You will use the " + containers[my_region] + " container for your SageMaker endpoint.")
```

# Create bucket from Notebook/Python

```python
bucket_name = 'mybucket'
s3 = boto3.resource('s3')
try:
    if my_region == 'us-east-1':
        s3.create_bucket(Bucket=bucket_name)
    else:
        s3.create_bucket(Bucket=bucket_name, CreateBucketConfiguration={ 'LocationConstraint': my_region })
    print('S3 bucket created successfully')
except Exception as e:
    print('S3 error: ', e)
```
# Download data to instance and load to dataframe

```python
try:
  urllib.request.urlretrieve ("https://d1.awsstatic.com/tmt/build-train-deploy-machine-learning-model-sagemaker/bank_clean.27f01fbbdf43271788427f3682996ae29ceca05d.csv", "bank_clean.csv")
  print('Success: downloaded bank_clean.csv.')
except Exception as e:
  print('Data load error: ',e)

try:
  model_data = pd.read_csv('./bank_clean.csv',index_col=0)
  print('Success: Data loaded into dataframe.')
except Exception as e:
    print('Data load error: ',e)
```

# Train Test Shuffle Split

Shuffle the data and split it into training data and test data.

```python
train_data, test_data = np.split(model_data.sample(frac=1, random_state=1729), [int(0.7 * len(model_data))])
print(train_data.shape, test_data.shape)
```

# Load Model

Need to reformat header and first column of training data from s3

```python
pd.concat([train_data['y_yes'], train_data.drop(['y_no', 'y_yes'], axis=1)], axis=1).to_csv('train.csv', index=False, header=False)
boto3.Session().resource('s3').Bucket(bucket_name).Object(os.path.join(prefix, 'train/train.csv')).upload_file('train.csv')
s3_input_train = sagemaker.s3_input(s3_data='s3://{}/{}/train'.format(bucket_name, prefix), content_type='csv')
```


# Tune Parameters (Gradient Optimization)
Using XGBoost estimator

```python

sess = sagemaker.Session()
xgb = sagemaker.estimator.Estimator(containers[my_region],role, train_instance_count=1, train_instance_type='ml.m4.xlarge',output_path='s3://{}/{}/output'.format(bucket_name, prefix),sagemaker_session=sess)
xgb.set_hyperparameters(max_depth=5,eta=0.2,gamma=4,min_child_weight=6,subsample=0.8,silent=0,objective='binary:logistic',num_round=100)
```

# Train Model
```python
xgb.fit({'train': s3_input_train})

Uploading - Uploading generated training model
Completed - Training job completed
Billable seconds: 54
```

# Deploy Model

Deploy the model on a server and create an endpoint that you can access.

```python
xgb_predictor = xgb.deploy(initial_instance_count=1,instance_type='ml.m4.xlarge')

```

# Make Predictions

```python
test_data_array = test_data.drop(['y_no', 'y_yes'], axis=1).values #load the data into an array
xgb_predictor.content_type = 'text/csv' # set the data type for an inference
xgb_predictor.serializer = csv_serializer # set the serializer type
predictions = xgb_predictor.predict(test_data_array).decode('utf-8') # predict!
predictions_array = np.fromstring(predictions[1:], sep=',') # and turn the prediction into an array
print(predictions_array.shape)
```

# Evaluate Performance
```python
cm = pd.crosstab(index=test_data['y_yes'], columns=np.round(predictions_array), rownames=['Observed'], colnames=['Predicted'])
tn = cm.iloc[0,0]; fn = cm.iloc[1,0]; tp = cm.iloc[1,1]; fp = cm.iloc[0,1]; p = (tp+tn)/(tp+tn+fp+fn)*100
print("\n{0:<20}{1:<4.1f}%\n".format("Overall Classification Rate: ", p))
print("{0:<15}{1:<15}{2:>8}".format("Predicted", "No Purchase", "Purchase"))
print("Observed")
print("{0:<15}{1:<2.0f}% ({2:<}){3:>6.0f}% ({4:<})".format("No Purchase", tn/(tn+fn)*100,tn, fp/(tp+fp)*100, fp))
print("{0:<16}{1:<1.0f}% ({2:<}){3:>7.0f}% ({4:<}) \n".format("Purchase", fn/(tn+fn)*100,fn, tp/(tp+fp)*100, tp))
```

# Terminate Resources

Delete endpoint, objects, bucket

```bash

sagemaker.Session().delete_endpoint(xgb_predictor.endpoint)
bucket_to_delete = boto3.resource('s3').Bucket(bucket_name)
bucket_to_delete.objects.all().delete()

```
