from azureml.core import Workspace
from azureml.core.model import Model

from azureml.core.webservice import AciWebservice, Webservice 
from azureml.core.model import InferenceConfig 


# Connect to your Azure ML workspace
workspace = Workspace.get(name="PAIWP",
                          subscription_id="8ce46f80-4a45-4e79-a0d2-f29438358d73",
                          resource_group="RG")

# Register the model
model = Model.register(model_path="model.pkl",  # Path to the .pkl file
                       model_name="my_model",  # Name of the model for reference in Azure ML
                       workspace=workspace)

aci_config = AciWebservice.deploy_configuration(cpu_cores=1, memory_gb=1) 

inference_config = InferenceConfig(runtime= "python", entry_script="score.py", conda_file="environment.yml") 

service = Model.deploy(workspace=workspace, name='service1', models=[model], inference_config=inference_config, deployment_config=aci_config) 
service.wait_for_deployment(show_output=True)