# Train GPT-2 in TFX 
Main work done by [nsheppert/gpt-2](https://github.com/nshepperd/gpt-2)

In this repository, we take the existing code and transform it into a TFX pipeline. The TFX pipeline can then be orchestrated with either Airflow or Kubeflow. Have a look at `examples` for getting to know how to use this module.

## Getting Started
```
git clone https://github.com/steven-mi/tfx-gpt2.git
cd tfx-gtp2
pipenv install
pipenv run bash
```

**Run pipeline with Apache Beam**
```bash
cd examples
python beam-example.py
```

**Run pipeline with Airflow**
```bash
airflow initdb
cp airflow-example.py $AIRFLOW_HOME/dags
cp -r data $AIRFLOW_HOME/dags

airflow scheduler -D
airflow webserver 
# go to localhost:6006 an start the DAG
```

**Run pipeline with Kubeflow**
```bash
cd examples
python kubeflow-example.py
# upload pipeline to Kubeflow
```

## TODO
- test kubeflow example
- create component for exporting model to TFServing
- resume training
