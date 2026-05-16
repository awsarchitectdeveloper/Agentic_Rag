############################################################
# Input variables for Azure RAG Demo Infrastructure
#
# These variables allow customization of the deployment.
############################################################

variable "location" {
  description = "The location to deploy the Azure services in."
  type        = string
  default     = "westeurope"
}

variable "resource_group_name" {
  # Name of the Azure resource group
  description = "The name of the resource group."
  type        = string
  default     = "generic-rag"
}

variable "container_registry_sku" {
  # SKU for Azure Container Registry
  description = "SKU for the Azure Container Registry."
  type        = string
  default     = "Basic"
}

variable "log_analytics_sku" {
  # SKU for Log Analytics Workspace
  description = "SKU for Log Analytics Workspace."
  type        = string
  default     = "PerGB2018"
}

variable "log_analytics_retention" {
  # Retention period (days) for Log Analytics Workspace
  description = "Retention in days for Log Analytics Workspace."
  type        = number
  default     = 30
}

variable "storage_account_tier" {
  # Tier for Azure Storage Account
  description = "Storage account tier."
  type        = string
  default     = "Standard"
}

variable "storage_account_replication" {
  # Replication type for Azure Storage Account
  description = "Storage account replication type."
  type        = string
  default     = "LRS"
}

variable "cognitive_account_location" {
  # Location for Azure Cognitive Services
  description = "Location for Cognitive Services account."
  type        = string
  default     = "swedencentral"
}

variable "cognitive_account_sku" {
  # SKU for Azure Cognitive Services
  description = "SKU for Cognitive Services account."
  type        = string
  default     = "S0"
}

variable "search_service_sku" {
  # SKU for Azure Search Service
  description = "SKU for Azure Search Service."
  type        = string
  default     = "free"
}
