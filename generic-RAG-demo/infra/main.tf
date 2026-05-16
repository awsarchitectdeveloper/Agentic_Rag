
# Configure the Azure provider
locals {
  roles_to_assign = [
    "Cognitive Services OpenAI User",
    "Cognitive Services User",
    "Search Service Contributor",
    "Search Index Data Contributor",
    "Container Apps Contributor",
    "Key Vault Secrets User",
    "Storage Blob Data Contributor",
    "AcrPull"
  ]
}
terraform {
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "4.36.0"
    }
  }
}

provider "azurerm" {
  features {
    key_vault {
      purge_soft_delete_on_destroy    = true
      recover_soft_deleted_key_vaults = true
    }
    }
}

data "azurerm_client_config" "current" {}

resource "random_string" "suffix" {
  length  = 5
  upper   = false
  special = false

}

locals {
  rg_name   = var.resource_group_name
  location  = var.location
  suffix    = random_string.suffix.result
}


resource "azurerm_resource_group" "rg" {
  name     = local.rg_name
  location = local.location
}

resource "azurerm_user_assigned_identity" "mi" {
  location            = local.location
  name                = "rag-mi"
  resource_group_name = local.rg_name
}

resource "azurerm_key_vault" "kv" {
  name                        = "rag-kv-${local.suffix}"
  location                    = local.location
  resource_group_name         = local.rg_name
  enabled_for_disk_encryption = true
  tenant_id                   = data.azurerm_client_config.current.tenant_id
  soft_delete_retention_days  = 7
  purge_protection_enabled    = true

  sku_name = "standard"

  access_policy = [
    {
  application_id = azurerm_user_assigned_identity.mi.client_id
  tenant_id = azurerm_user_assigned_identity.mi.tenant_id
  object_id = azurerm_user_assigned_identity.mi.principal_id
    certificate_permissions = [
        "Get",
        ]
    key_permissions = [
        "Get",
        ]
    secret_permissions = [
      "Get",
    ]
    storage_permissions = [
      "Get",
    ]
  }
  ]
}

resource "azurerm_log_analytics_workspace" "log" {
  name                = "rag-log"
  location            = local.location
  resource_group_name = local.rg_name
  sku                 = var.log_analytics_sku
  retention_in_days   = var.log_analytics_retention
}

resource "azurerm_container_registry" "acr" {
  # Azure Container Registry name must be 5-50 alphanumeric characters, lowercase only, no dashes
  name                = "ragcr${local.suffix}"
  resource_group_name = local.rg_name
  location            = local.location
  sku                 = var.container_registry_sku
}

resource "azurerm_container_app_environment" "container_env" {
  name                       = "rag-cae"
  location                   = local.location
  resource_group_name        = local.rg_name
  log_analytics_workspace_id = azurerm_log_analytics_workspace.log.id
}

resource "azurerm_container_app" "container_app" {
  name                         = "rag-ca"
  container_app_environment_id = azurerm_container_app_environment.container_env.id
  resource_group_name          = local.rg_name
  revision_mode                = "Single"

  ingress {
    target_port = 80
    traffic_weight {
        label = "stable"
        percentage = 100
        latest_revision = true
    }
  }

  identity {
    type = "UserAssigned"
    identity_ids = [
      azurerm_user_assigned_identity.mi.id,
    ]
  }
  registry {
    server = azurerm_container_registry.acr.login_server
    identity = azurerm_user_assigned_identity.mi.id
  }

  template {
    container {
      name   = "examplecontainerapp"
      image  = "mcr.microsoft.com/k8se/quickstart:latest"
      cpu    = 0.25
      memory = "0.5Gi"
    }
  }
}

resource "azurerm_storage_account" "sa" {
  name                     = "ragsa${local.suffix}"
  location                 = local.location
  resource_group_name      = local.rg_name
  account_tier             = var.storage_account_tier
  account_replication_type = var.storage_account_replication
  public_network_access_enabled = false
  allow_nested_items_to_be_public = false

  network_rules {
    default_action = "Deny"
    bypass         = ["AzureServices"]
  }
}

resource "azurerm_cognitive_account" "cog_account" {
  name                = "rag-cog"
  location            = var.cognitive_account_location
  resource_group_name = local.rg_name
  kind                = "OpenAI"
  sku_name            = var.cognitive_account_sku
  local_auth_enabled  = false
  custom_subdomain_name = "rag-cog-${local.suffix}"
}

resource "azurerm_cognitive_deployment" "llm_deployment" {
  name                 = "gpt-4-1-mini"
  cognitive_account_id = azurerm_cognitive_account.cog_account.id
  sku {
    name        = "Standard"
    capacity    = 100
  }
  model {
    format  = "OpenAI"
    name    = "gpt-4.1-mini"
    version = "2025-04-14"
  }
}

resource "azurerm_cognitive_deployment" "embedding_deployment" {
  name                 = "text-embedding-3-large"
  cognitive_account_id = azurerm_cognitive_account.cog_account.id
  sku {
    name        = "Standard"
    capacity    = 100
  }
    model {
    format  = "OpenAI"
    name    = "text-embedding-3-large"
    version = 1
  }
}

resource "azurerm_cognitive_deployment" "agent_deployment" {
  name                 = "gpt-5-mini"
  cognitive_account_id = azurerm_cognitive_account.cog_account.id
  sku {
    name        = "DataZoneStandard"
    capacity    = 20
  }
  model {
    format  = "OpenAI"
    name    = "gpt-5-mini"
    version = "2025-08-07"
  }
}

resource "azurerm_cognitive_account" "doc_intelligence" {
  name                  = "rag-document-intelligence"
  location              = local.location
  resource_group_name   = local.rg_name
  kind                  = "FormRecognizer"
  sku_name              = var.cognitive_account_sku
  local_auth_enabled    = false
  custom_subdomain_name = "rag1-document-intelligence"
}

resource "azurerm_search_service" "search" {
  name                = "rag-ai-${local.suffix}"
  resource_group_name = local.rg_name
  location            = local.location
  sku                 = var.search_service_sku
  replica_count       = 1
  partition_count     = 1
  local_authentication_enabled = false

  identity {
    type = "UserAssigned"
    identity_ids = [
      azurerm_user_assigned_identity.mi.id
    ]
  }
}

module "role_assignments" {
  source               = "./modules/role_assignment"
  for_each             = toset(local.roles_to_assign)
  role_definition_name = each.key
  scope                = "/subscriptions/${data.azurerm_client_config.current.subscription_id}/resourceGroups/${azurerm_resource_group.rg.name}"
  principal_id         = azurerm_user_assigned_identity.mi.principal_id
}
