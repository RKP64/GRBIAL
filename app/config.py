from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # API
    api_keys: str = "dev-key-change-me"
    cors_origins: str = "http://localhost:5173"
    log_level: str = "INFO"

    # Language model provider: auto | azure | bedrock | anthropic | openai
    llm_provider: str = "auto"
    # Where embeddings come from when the chat provider has none:
    # none | local | azure | bedrock | openai
    embedding_provider: str = "none"

    # Local sentence-transformers embeddings (no API key, no data leaves the host)
    local_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    local_embedding_device: str = ""          # "" = auto, or cpu / cuda / mps
    local_embedding_batch_size: int = 32
    local_embedding_normalize: bool = True

    # Azure OpenAI
    azure_openai_endpoint: str = ""
    azure_openai_api_key: str = ""
    azure_openai_api_version: str = "2024-10-21"
    azure_openai_deployment: str = ""
    azure_openai_embedding_deployment: str = ""

    # Amazon Bedrock
    aws_region: str = "us-east-1"
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""
    aws_session_token: str = ""
    aws_profile: str = ""
    bedrock_model_id: str = ""
    bedrock_embedding_model_id: str = ""
    bedrock_max_tokens: int = 4096
    bedrock_embedding_dimensions: int = 0        # 0 = model default
    bedrock_endpoint_url: str = ""               # VPC endpoint, if used

    # Anthropic
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-4-5"
    anthropic_max_tokens: int = 4096
    anthropic_base_url: str = ""

    # Fine-tuning
    training_provider_mode: str = "azure"      # azure | openai
    azure_finetune_api_version: str = "2024-10-21"
    training_base_models: str = ""             # comma-separated, offered in the console
    bedrock_training_base_models: str = ""
    bedrock_training_role_arn: str = ""
    bedrock_training_s3_uri: str = ""
    groq_api_key: str = ""
    # Together AI — open-weight fine-tuning, weights are retrievable.
    # US-hosted, so training data leaves India: pilots and synthetic data only.
    together_api_key: str = ""
    # Written as "Label|model-id" so the console shows the label only.
    together_base_models: str = (
        "Compact 8B|meta-llama/Meta-Llama-3.1-8B-Instruct-Reference,"
        "Standard 14B|Qwen/Qwen2.5-14B-Instruct,"
        "Code 32B|Qwen/Qwen2.5-Coder-32B-Instruct")
    # Yotta Shakti Studio — Indian jurisdiction, Tier IV Mumbai / Greater Noida.
    # Base URL is configurable because the tenant surface is not fixed.
    shakti_api_key: str = ""
    shakti_base_url: str = ""
    shakti_base_models: str = ("Compact 8B|meta-llama/Llama-3.1-8B-Instruct,"
                               "Standard 14B|Qwen/Qwen2.5-14B-Instruct")
    # Training on hardware you control
    training_worker_url: str = ""
    training_worker_token: str = ""
    training_worker_timeout_seconds: float = 60.0
    training_worker_base_models: str = ""

    # OpenAI-compatible endpoint
    openai_base_url: str = ""
    openai_api_key: str = ""
    openai_model: str = ""
    openai_embedding_model: str = ""

    # Extraction
    extraction_concurrency: int = 6
    extraction_timeout_seconds: int = 90
    rows_per_chunk: int = 50
    chunk_size: int = 2000
    chunk_overlap: int = 200

    # Storage
    graph_backend: str = "local"      # local | cloud | neo4j | dual
    data_dir: Path = Path("./data")

    # Sign-in. Set JWT_SECRET in any deployment where sessions must survive a
    # restart or span more than one replica.
    jwt_secret: str = ""

    # Entity resolution. Off by default: it adds a pass over every extracted
    # node, and on a domain with clean identifiers it earns nothing.
    resolve_entities: bool = False
    # Whether the narrow ambiguous band is put to a model. With this off those
    # pairs go straight to the review queue instead.
    resolve_adjudicate: bool = True

    # Users and audit go to MongoDB when this is set, files when it is not.
    mongo_connection_string: str = ""
    mongo_database: str = "KGPlatform"
    mongo_users_collection: str = "Users"
    mongo_audit_collection: str = "AuditLog"
    cosmos_gremlin_endpoint: str = ""
    cosmos_key: str = ""
    cosmos_database: str = "GraphDatabase"
    cosmos_collection: str = "KnowledgeGraph"

    # Neo4j. The same settings serve both deployment options — only the URI
    # differs. bolt://host:7687 for Neo4j on a virtual machine inside the
    # VNet (the production shape, since graph data never leaves the tenant),
    # neo4j+s://<id>.databases.neo4j.io for AuraDB (development and demo
    # only, as the data sits outside the tenant).
    neo4j_uri: str = ""
    neo4j_user: str = "neo4j"
    neo4j_password: str = ""
    neo4j_database: str = "neo4j"

    # Retrieval
    retriever: str = "keyword"
    capture_passages: bool = True     # keep source chunks for local passage search

    # Azure AI Search (hybrid document retrieval alongside the graph)
    azure_search_endpoint: str = ""
    azure_search_api_key: str = ""
    azure_search_index: str = ""
    azure_search_vector_field: str = "text_vector"
    azure_search_semantic_config: str = ""
    azure_search_timeout_seconds: float = 10.0

    @property
    def api_key_set(self) -> set[str]:
        return {k.strip() for k in self.api_keys.split(",") if k.strip()}

    @property
    def cors_list(self) -> list[str]:
        """Origins permitted to call this API from a browser.

        Set CORS_ORIGINS=* only where the API is not reachable from the public
        internet — with API-key auth, a wildcard origin means any page a user
        visits can call it with a key they have.
        """
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def azure_search_configured(self) -> bool:
        return bool(self.azure_search_endpoint and self.azure_search_api_key
                    and self.azure_search_index)

    @property
    def llm_configured(self) -> bool:
        """True when any provider has enough configuration to run."""
        azure = bool(self.azure_openai_endpoint and self.azure_openai_api_key
                     and self.azure_openai_deployment)
        bedrock = bool(self.bedrock_model_id and (
            self.aws_region or self.aws_profile
            or (self.aws_access_key_id and self.aws_secret_access_key)))
        anthropic = bool(self.anthropic_api_key)
        compatible = bool(self.openai_model)
        choice = (self.llm_provider or "auto").lower()
        if choice in ("azure", "azure_openai"):
            return azure
        if choice in ("bedrock", "aws"):
            return bedrock
        if choice == "anthropic":
            return anthropic
        if choice in ("openai", "openai_compatible"):
            return compatible
        return azure or bedrock or anthropic or compatible

    @property
    def embeddings_configured(self) -> bool:
        # A separate embedding provider always wins, because the chat provider
        # may not offer embeddings at all.
        secondary = (self.embedding_provider or "none").lower()
        if secondary != "none":
            if secondary == "local":
                return bool(self.local_embedding_model)
            if secondary in ("bedrock", "aws"):
                return bool(self.bedrock_embedding_model_id)
            if secondary in ("azure", "azure_openai"):
                return bool(self.azure_openai_embedding_deployment)
            if secondary in ("openai", "openai_compatible"):
                return bool(self.openai_embedding_model)
        choice = (self.llm_provider or "auto").lower()
        if choice == "anthropic":
            return False
        if choice in ("bedrock", "aws"):
            return bool(self.bedrock_embedding_model_id)
        if choice in ("openai", "openai_compatible"):
            return bool(self.openai_embedding_model)
        if choice in ("azure", "azure_openai"):
            return bool(self.azure_openai_embedding_deployment)
        return bool(self.azure_openai_embedding_deployment
                    or self.bedrock_embedding_model_id or self.openai_embedding_model)


@lru_cache
def get_settings() -> Settings:
    s = Settings()
    s.data_dir.mkdir(parents=True, exist_ok=True)
    return s
