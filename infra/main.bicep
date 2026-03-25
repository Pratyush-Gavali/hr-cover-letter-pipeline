// ── main.bicep ────────────────────────────────────────────────────────────────
// Azure infrastructure for the HR Cover Letter Intelligence Pipeline.
//
// Resources provisioned
// ─────────────────────
//   Storage Account (Blob, lifecycle policy)
//   Azure Container Apps environment + app
//   Azure Service Bus namespace + queue
//   Azure Cache for Redis
//   Azure Key Vault
//   Azure Container Registry
//   Log Analytics workspace
//
// Deploy
// ------
// az group create --name hr-pipeline-rg --location australiaeast
// az deployment group create \
//   --resource-group hr-pipeline-rg \
//   --template-file infra/main.bicep \
//   --parameters @infra/params.json

@description('Base name for all resources')
param baseName string = 'hrpipeline'

@description('Azure region')
param location string = resourceGroup().location

@description('Container image tag')
param imageTag string = 'latest'

@description('OpenAI endpoint')
param openAiEndpoint string

@description('Qdrant Cloud URL (external)')
param qdrantUrl string

@secure()
param qdrantApiKey string

var storageAccountName = '${baseName}sa'
var containerAppEnvName = '${baseName}-cae'
var containerAppName = '${baseName}-app'
var serviceBusName = '${baseName}-sb'
var redisName = '${baseName}-redis'
var keyVaultName = '${baseName}-kv'
var acrName = '${baseName}acr'
var logWorkspaceName = '${baseName}-logs'

// ── Log Analytics workspace ────────────────────────────────────────────────────
resource logWorkspace 'Microsoft.OperationalInsights/workspaces@2022-10-01' = {
  name: logWorkspaceName
  location: location
  properties: {
    sku: { name: 'PerGB2018' }
    retentionInDays: 30
  }
}

// ── Azure Container Registry ───────────────────────────────────────────────────
resource acr 'Microsoft.ContainerRegistry/registries@2023-01-01-preview' = {
  name: acrName
  location: location
  sku: { name: 'Basic' }
  properties: { adminUserEnabled: true }
}

// ── Storage Account + Blob lifecycle ──────────────────────────────────────────
resource storageAccount 'Microsoft.Storage/storageAccounts@2023-01-01' = {
  name: storageAccountName
  location: location
  sku: { name: 'Standard_LRS' }
  kind: 'StorageV2'
  properties: {
    accessTier: 'Hot'
    minimumTlsVersion: 'TLS1_2'
    allowBlobPublicAccess: false
    supportsHttpsTrafficOnly: true
    encryption: {
      services: {
        blob: { enabled: true, keyType: 'Account' }
      }
      keySource: 'Microsoft.Storage'
    }
  }
}

resource blobService 'Microsoft.Storage/storageAccounts/blobServices@2023-01-01' = {
  parent: storageAccount
  name: 'default'
}

resource coversContainer 'Microsoft.Storage/storageAccounts/blobServices/containers@2023-01-01' = {
  parent: blobService
  name: 'covers'
  properties: { publicAccess: 'None' }
}

// Lifecycle policy: hot->cool at 30d, cool->archive at 180d, delete at 2555d
resource lifecyclePolicy 'Microsoft.Storage/storageAccounts/managementPolicies@2023-01-01' = {
  parent: storageAccount
  name: 'default'
  properties: {
    policy: {
      rules: [
        {
          name: 'cover-letter-tiers'
          enabled: true
          type: 'Lifecycle'
          definition: {
            filters: {
              blobTypes: ['blockBlob']
              prefixMatch: ['covers/']
            }
            actions: {
              baseBlob: {
                tierToCool:    { daysAfterModificationGreaterThan: 30  }
                tierToArchive: { daysAfterModificationGreaterThan: 180 }
                delete:        { daysAfterModificationGreaterThan: 2555 }
              }
            }
          }
        }
      ]
    }
  }
}

// ── Azure Service Bus ──────────────────────────────────────────────────────────
resource serviceBus 'Microsoft.ServiceBus/namespaces@2022-10-01-preview' = {
  name: serviceBusName
  location: location
  sku: { name: 'Standard', tier: 'Standard' }
}

resource coverLettersQueue 'Microsoft.ServiceBus/namespaces/queues@2022-10-01-preview' = {
  parent: serviceBus
  name: 'cover-letters'
  properties: {
    maxDeliveryCount: 3
    defaultMessageTimeToLive: 'PT24H'
    deadLetteringOnMessageExpiration: true
    lockDuration: 'PT5M'
  }
}

// ── Azure Cache for Redis ──────────────────────────────────────────────────────
resource redis 'Microsoft.Cache/redis@2023-04-01' = {
  name: redisName
  location: location
  properties: {
    sku: { name: 'Basic', family: 'C', capacity: 0 }
    enableNonSslPort: false
    minimumTlsVersion: '1.2'
    redisConfiguration: {
      'maxmemory-policy': 'allkeys-lru'
      'maxmemory-reserved': '50'
    }
  }
}

// ── Azure Key Vault ────────────────────────────────────────────────────────────
resource keyVault 'Microsoft.KeyVault/vaults@2023-02-01' = {
  name: keyVaultName
  location: location
  properties: {
    sku: { name: 'standard', family: 'A' }
    tenantId: subscription().tenantId
    enableSoftDelete: true
    softDeleteRetentionInDays: 90
    enablePurgeProtection: true
    enableRbacAuthorization: true  // RBAC instead of legacy access policies
  }
}

// ── Container Apps environment ─────────────────────────────────────────────────
resource containerAppEnv 'Microsoft.App/managedEnvironments@2023-05-01' = {
  name: containerAppEnvName
  location: location
  properties: {
    appLogsConfiguration: {
      destination: 'log-analytics'
      logAnalyticsConfiguration: {
        customerId: logWorkspace.properties.customerId
        sharedKey: logWorkspace.listKeys().primarySharedKey
      }
    }
  }
}

// ── Container App ──────────────────────────────────────────────────────────────
resource containerApp 'Microsoft.App/containerApps@2023-05-01' = {
  name: containerAppName
  location: location
  identity: { type: 'SystemAssigned' }
  properties: {
    managedEnvironmentId: containerAppEnv.id
    configuration: {
      activeRevisionsMode: 'Single'
      ingress: {
        external: true
        targetPort: 8000
        transport: 'http'
        corsPolicy: {
          allowedOrigins: ['*']
          allowedMethods: ['GET', 'POST', 'OPTIONS']
        }
      }
      registries: [
        {
          server: acr.properties.loginServer
          username: acr.listCredentials().username
          passwordSecretRef: 'acr-password'
        }
      ]
      secrets: [
        {
          name: 'acr-password'
          value: acr.listCredentials().passwords[0].value
        }
        {
          name: 'qdrant-api-key'
          value: qdrantApiKey
        }
        {
          name: 'redis-conn-string'
          value: 'rediss://:${redis.listKeys().primaryKey}@${redis.properties.hostName}:6380'
        }
      ]
    }
    template: {
      containers: [
        {
          name: containerAppName
          image: '${acr.properties.loginServer}/hr-pipeline:${imageTag}'
          resources: {
            cpu: json('2.0')
            memory: '4Gi'
          }
          env: [
            { name: 'AZURE_BLOB_ACCOUNT_URL', value: 'https://${storageAccountName}.blob.core.windows.net' }
            { name: 'QDRANT_URL',             value: qdrantUrl }
            { name: 'QDRANT_API_KEY',         secretRef: 'qdrant-api-key' }
            { name: 'REDIS_URL',              secretRef: 'redis-conn-string' }
            { name: 'AZURE_KEY_VAULT_URL',    value: keyVault.properties.vaultUri }
            { name: 'AZURE_OPENAI_ENDPOINT',  value: openAiEndpoint }
            { name: 'SERVICEBUS_NAMESPACE',   value: '${serviceBusName}.servicebus.windows.net' }
            { name: 'LOG_LEVEL',              value: 'INFO' }
          ]
        }
      ]
      scale: {
        minReplicas: 1
        maxReplicas: 5
        rules: [
          {
            name: 'http-scaling'
            http: { metadata: { concurrentRequests: '10' } }
          }
        ]
      }
    }
  }
}

// ── RBAC assignments ───────────────────────────────────────────────────────────
// Container App managed identity gets minimal permissions

// Storage Blob Data Contributor (read + write covers container)
resource blobRole 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(storageAccount.id, containerApp.id, 'blob-contributor')
  scope: storageAccount
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions',
      'ba92f5b4-2d11-453d-a403-e96b0029c9fe')  // Storage Blob Data Contributor
    principalId: containerApp.identity.principalId
    principalType: 'ServicePrincipal'
  }
}

// Key Vault Secrets Officer (read + write secrets for PII maps)
resource kvRole 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(keyVault.id, containerApp.id, 'kv-secrets-officer')
  scope: keyVault
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions',
      'b86a8fe4-44ce-4948-aee5-eccb2c155cd7')  // Key Vault Secrets Officer
    principalId: containerApp.identity.principalId
    principalType: 'ServicePrincipal'
  }
}

// Service Bus Data Owner (send + receive on cover-letters queue)
resource sbRole 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(serviceBus.id, containerApp.id, 'sb-data-owner')
  scope: serviceBus
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions',
      '090c5cfd-751d-490a-894a-3ce6f1109419')  // Azure Service Bus Data Owner
    principalId: containerApp.identity.principalId
    principalType: 'ServicePrincipal'
  }
}

// ── Outputs ────────────────────────────────────────────────────────────────────
output appUrl string = 'https://${containerApp.properties.configuration.ingress.fqdn}'
output storageAccountName string = storageAccount.name
output keyVaultUri string = keyVault.properties.vaultUri
output serviceBusNamespace string = '${serviceBusName}.servicebus.windows.net'
output acrLoginServer string = acr.properties.loginServer
