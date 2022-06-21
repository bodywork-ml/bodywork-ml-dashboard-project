# Deploy ML Dashboards on Kubernetes with Bodywork

![bodywork](https://bodywork-media.s3.eu-west-2.amazonaws.com/ml_dashboard_workflow.png)

This repository contains a Bodywork project that demonstrates how to run a ML workflow on Kubernetes, with Bodywork. The example ML workflow has two stages:

1. Run a batch job to train a model that is stored in a AWS S3 bucket.
2. Deploy a Plotly dashboard to present model performance information.

To run this project, follow the steps below.

## Get Access to a Kubernetes Cluster

Use our [Quickstart Guide to Kubernetes for MLOps](https://bodywork.readthedocs.io/en/latest/kubernetes/#quickstart) to spin-up a local Minikube cluster in minutes.

## Install the Bodywork Python Package

```shell
$ pip install bodywork
```

## Load Dashboard Credentials into Cluster

The dashboard uses basic authentication, which requires a username and password to be passed to it via environment variables. These can be securely injected into the containers running the app, using Bodywork's secret management capabilities,

```shell
$ bodywork create secret plotly-dash-credentials \
    --group dev \
    --data DASH_USERNAME=bodywork \
    --data DASH_PASSWORD=bodywork123
```

## Run the ML Pipeline

```shell
$ bodywork create deployment https://github.com/bodywork-ml/bodywork-ml-dashboard-project.git
```

The orchestrator logs will be streamed to your terminal until the job has been successfully completed.

## Accessing the Dashboard

Once the deployment has completed, the dashboard server be ready for testing. Bodywork will create ingress routes to your endpoints using the following scheme:

```md
/PIPELINE_NAME/STAGE_NAME/
```

To open an access route to the cluster for testing, start a new terminal and run,

```text
$ kubectl -n ingress-nginx port-forward service/ingress-nginx-controller 8080:80
```

Then browse the dashboard at,

```http
http://localhost:8080/bodywork-ml-dashboard-project/stage-2-dashboard-app/dash/
```

You should see something that looks like,

![bodywork](https://bodywork-media.s3.eu-west-2.amazonaws.com/ml_dashboard_screenshot.png)

## Running the Workflow on a Schedule

If you're happy with the test results, you can schedule the workflow-controller to operate remotely on the cluster on a pre-defined schedule. For example, to setup the the workflow to run every hour, use the following command,

```shell
$ bodywork create cronjob https://github.com/bodywork-ml/bodywork-ml-dashboard-project \
    --name=train-with-metrics-dashboard \
    --schedule="0 * * * *" \
```

Each scheduled workflow will attempt to re-run the batch-job, as defined by the state of this repository's `master` branch at the time of execution.

To get the execution history for all `train-with-metrics-dashboard` jobs use,

```shell
$ bodywork get cronjob train-with-metrics-dashboard --history
```

## Cleaning Up

To delete the deployment in its entirety,

```shell
$ bodywork delete deployment train-with-metrics-dashboard
```

## Make this Project Your Own

This repository is a [GitHub template repository](https://docs.github.com/en/free-pro-team@latest/github/creating-cloning-and-archiving-repositories/creating-a-repository-from-a-template) that can be automatically copied into your own GitHub account by clicking the `Use this template` button above.

After you've cloned the template project, use official [Bodywork documentation](https://bodywork.readthedocs.io/en/latest/) to help modify the project to meet your own requirements.
