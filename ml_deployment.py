from azureml.core import Workspace

subscription_id = '8ce46f80-4a45-4e79-a0d2-f29438358d73'
resource_group = 'RG'
workspace_name = 'PAIWP'
workspace = Workspace(subscription_id, resource_group, workspace_name)