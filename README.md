# Deploy ML Dashboards on Kubernetes with Bodywork

![bodywork](https://bodywork-media.s3.eu-west-2.amazonaws.com/ml_dashboard_workflow.png)

This repository contains a Bodywork project that demonstrates how to run a ML workflow on Kubernetes, with Bodywork. The example ML workflow has two stages:

1. Run a batch job to train a model that is stored in a AWS S3 bucket.
2. Deploy a Plotly dashboard to present model performance information.

To run this project, follow the steps below.

## Get Access to a Kubernetes Cluster

In order to run this example project you will need access to a Kubernetes cluster. To setup a single-node test cluster on your local machine you can use [minikube](https://minikube.sigs.Kubernetes.io/docs/) or [docker-for-desktop](https://www.docker.com/products/docker-desktop). Check your access to Kubernetes by running,

```shell
$ kubectl cluster-info
```

Which should return the details of your cluster.

## Install the Bodywork Python Package

```shell
$ pip install bodywork
```

## Setup a Kubernetes Namespace for use with Bodywork

```shell
$ bodywork setup-namespace ml-workflow
```

## Inject Dashboard Credentials into Namespace

The dashboard uses basic authentication, which requires a username and password to be passed to it via environment variables. These can be securely injected into the containers running the app, using Bodywork's secret management capabilities,

```shell
bodywork secret create \
    --namespace=ml-workflow \
    --name=plotly-dash-credentials \
    --data DASH_USERNAME=bodywork DASH_PASSWORD=bodywork123
```

## Run the Workflow

To test the ML workflow, using a workflow-controller running on your local machine and interacting with your Kubernetes cluster, run,

```shell
$ bodywork deployment create \
    --namespace=scoring-service \
    --name=test-deployment \
    --git-repo-url=https://github.com/bodywork-ml/bodywork-ml-dashboard-project \
    --git-repo-branch=master \
    --local-workflow-controller
```

The workflow-controller logs will be streamed to your shell's standard output until the job has been successfully completed.

## Accessing the Dashboard

You can only reach the dashboard from outside the cluster, if you have [installed an ingress controller](https://bodywork.readthedocs.io/en/latest/kubernetes/#configuring-ingress) in your cluster (this is not as complex as it sounds). If an ingress controller is operational, then you can reach the dashboard with a browser at,

```http
http://YOUR_CLUSTERS_EXTERNAL_IP/ml-workflow/bodywork-ml-dashboard-project--stage-2-dashboard-app/dash/
```

See [here](https://bodywork.readthedocs.io/en/latest/kubernetes/#connecting-to-the-cluster) for instruction on how to retrieve `YOUR_CLUSTERS_EXTERNAL_IP`. You should see something that looks like,

![bodywork](https://bodywork-media.s3.eu-west-2.amazonaws.com/ml_dashboard_screenshot.png)

## Running the Workflow on a Schedule

If you're happy with the test results, you can schedule the workflow-controller to operate remotely on the cluster on a pre-defined schedule. For example, to setup the the workflow to run every hour, use the following command,

```shell
$ bodywork cronjob create \
    --namespace=ml-workflow \
    --name=train-with-metrics-dashboard \
    --schedule="0 * * * *" \
    --git-repo-url=https://github.com/bodywork-ml/bodywork-ml-dashboard-project \
    --git-repo-branch=master
```

Each scheduled workflow will attempt to re-run the batch-job, as defined by the state of this repository's `master` branch at the time of execution.

To get the execution history for all `train-with-metrics-dashboard` jobs use,

```shell
$ bodywork cronjob history \
    --namespace=ml-workflow \
    --name=train-with-metrics-dashboard
```

Which should return output along the lines of,

```text
JOB_NAME                                START_TIME                    COMPLETION_TIME               ACTIVE      SUCCEEDED       FAILED
train-with-metrics-dashboard-1605214260 2020-11-12 20:51:04+00:00     2020-11-12 20:52:34+00:00     0           1               0
```

Then to stream the logs from any given cronjob run (e.g. to debug and/or monitor for errors), use,

```shell
$ bodywork cronjob logs \
    --namespace=ml-workflow \
    --name=train-with-metrics-dashboard-1605214260
```

## Cleaning Up

To clean-up the deployment in its entirety, delete the namespace using kubectl - e.g. by running,

```shell
$ kubectl delete ns ml-workflow
```

## Make this Project Your Own

This repository is a [GitHub template repository](https://docs.github.com/en/free-pro-team@latest/github/creating-cloning-and-archiving-repositories/creating-a-repository-from-a-template) that can be automatically copied into your own GitHub account by clicking the `Use this template` button above.

After you've cloned the template project, use official [Bodywork documentation](https://bodywork.readthedocs.io/en/latest/) to help modify the project to meet your own requirements.
