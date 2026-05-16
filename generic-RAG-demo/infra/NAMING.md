# Resource Naming Convention

This project follows Microsoft's recommended Azure resource abbreviations and a consistent naming scheme:

**Format:**

```
rag-<abbreviation>[-<env>][-<number>]
```

- `rag` = project prefix
- `<abbreviation>` = Microsoft recommended abbreviation for the resource type
- `<env>` = optional environment (e.g., dev, prod)
- `<number>` = optional sequence number if needed

**Examples:**

| Resource Type                | Abbreviation | Example Name         |
|-----------------------------|--------------|---------------------|
| Resource Group               | rg           | rag-rg              |
| Log Analytics Workspace      | log          | rag-log             |
| Container Registry           | cr           | rag-cr              |
| Container Apps Environment   | cae          | rag-cae             |
| Container App                | ca           | rag-ca              |
| Key Vault                    | kv           | rag-kv              |
| Storage Account              | sa           | ragsa01             |
| Cognitive Account            | cog          | rag-cog             |
| Search Service               | ai           | rag-ai              |

Refer to [Microsoft's official abbreviations](https://learn.microsoft.com/en-us/azure/cloud-adoption-framework/ready/azure-best-practices/resource-abbreviations) for more.
