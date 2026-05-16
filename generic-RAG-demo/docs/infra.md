# Infrastructure as Code with Terraform

This page is dedicated to documenting the infrastructure as code (IaC).
We will be focusing on Azure and reusing as much standard components as possible.
For now, the implementation will be with public network access enabled.
Access will be controlled through Role Based Access Control (RBAC).
These roles will be assigned to Managed Identities (MI's).

Below a schematic overflow of the different infra components needed.



# Step 1
Make sure you have the details of a Service Principal with contributor rights on the subscription.
A [link](https://servicecentral.capgemini.com/sc?id=ticket&table=sc_req_item&sys_id=086f8372eba226980c57f705cad0cd17&view=sp) a example request.
If you don't have that info, ask Thomas van der Meer (me).

The documentation recommends to add the SP info as environment variables.

`bash
export ARM_CLIENT_ID="<APPID_VALUE>"
export ARM_CLIENT_SECRET="<PASSWORD_VALUE>"
export ARM_SUBSCRIPTION_ID="<SUBSCRIPTION_ID>"
export ARM_TENANT_ID="<TENANT_VALUE>"
`
With this set, deploying something went 'suprisingly' easy. Follow this [getting started](https://developer.hashicorp.com/terraform/tutorials/azure-get-started/azure-build) guide to get an idea of what you can do.

# Assign RBAC administrator role to Service principal
The SP will also do RBAC assignments and therefore needs the special role of RBAC Administrator. Add this role through the portal by assigning it to the Service principal.

# Add resources
All the resources as described in the architecture are added through the `main.tf` file.
The resources are basic azurerm resources and are as standard as possible.
Some notables:

### Public network access
For now, everything is with public network access.
No Vnets or any other network configuration is used.
If there is a requirement to do this at a client, this can make the design more difficult.

### Managed Identity
A User assigned managed identity (MI) is used to ensure access to the different resources.
The role assignments to the MI are done in the file called role-assignments.tf

### Secrets & keys
Secrets, keys and in general environment variables are stored in a key vault.
For example, the container app has a key vault reference to get the environment variable from the key vault.
That way, the environment variable is not exposed and stored safely.

### Gitea actions
To make sure that the infra is up to date a Gitea actions workflow (.gitea/workflows/terraform_deploy.yml) is built.
Here the terraform IaC runs when a merge to main is done.
The Service Principal details are saved as secrets in the Gitea Actions settings.
