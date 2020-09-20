import os
import logging

import pandas as pd

from pymongo import MongoClient
from typing import Any, Dict, List, Text

from tfx import types

from tfx.components.base import base_component
from tfx.components.base import executor_spec
from tfx.components.base import base_executor

from tfx.types import standard_artifacts

from tfx.types.artifact_utils import get_single_uri
from tfx.utils.dsl_utils import external_input

from tfx.types.component_spec import ChannelParameter
from tfx.types.component_spec import ExecutionParameter


class Executor(base_executor.BaseExecutor):

    def Do(self, input_dict: Dict[Text, List[types.Artifact]],
           output_dict: Dict[Text, List[types.Artifact]],
           exec_properties: Dict[Text, Any]) -> None:
        client = MongoClient(host=exec_properties["ip"],
                             port=int(exec_properties["port"]),
                             username=exec_properties["username"],
                             password=exec_properties["password"])
        dbname = exec_properties["dbname"]
        db = client[dbname]
        colnames = exec_properties["colnames"]
        end_token = exec_properties["end_token"]
        merged_text_dir = get_single_uri(output_dict["merged_text_dir"])

        raw_text = ""
        for colname in colnames:
            logging.info("Get data from {}/{}".format(dbname, colname))
            documents = db[colname].find({}, {"text": 1, "_id": 0})
            for document in documents:
                raw_text += document["text"] + end_token

        # store raw text for encoding
        merged_text_path = os.path.join(merged_text_dir, "merged_text")
        with open(merged_text_path, "w") as text_file:
            text_file.write(raw_text)
        logging.info("Saving merged text to {}".format(merged_text_dir))


class MongoExportSpec(types.ComponentSpec):
    PARAMETERS = {
        'ip': ExecutionParameter(type=Text),
        'port': ExecutionParameter(type=Text),
        'username': ExecutionParameter(type=Text),
        'password': ExecutionParameter(type=Text),
        'dbname': ExecutionParameter(type=Text),
        'colnames': ExecutionParameter(type=List),
        'end_token': ExecutionParameter(type=Text),
    }

    INPUTS = {
    }

    OUTPUTS = {
        'merged_text_dir': ChannelParameter(type=standard_artifacts.ExternalArtifact),
    }


class MongoExport(base_component.BaseComponent):
    SPEC_CLASS = MongoExportSpec
    EXECUTOR_SPEC = executor_spec.ExecutorClassSpec(Executor)

    def __init__(self,
                 colnames: Text,
                 end_token: Text,
                 ip: Text = "mongo",
                 port: Text = "27017",
                 username: Text = os.environ['MONGO_ROOT_USER'],
                 password: Text = os.environ['MONGO_ROOT_PASSWORD'],
                 dbname: Text = os.environ['MONGO_DATABASE_NAME']):
        merged_text_dir = external_input("MongoExport")
        spec = MongoExportSpec(ip=ip,
                               port=port,
                               username=username,
                               password=password,
                               dbname=dbname,
                               colname=colnames,
                               end_token=end_token,
                               merged_text_dir=merged_text_dir)
        super(MongoExport, self).__init__(spec=spec)
