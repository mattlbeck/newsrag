import yaml
from haystack.utils import Secret
from haystack.dataclasses import ChatMessage
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.generators.chat import HuggingFaceAPIChatGenerator
from haystack_integrations.components.generators.ollama import OllamaChatGenerator
from topics import HuggingfaceAPIJointEmbedder, SentenceTransformersJointEmbedder


class AppConfig:
    """Manage the configuration of the app from a YAML config file"""
    def __init__(self, config_path="config.yaml"):
        self.config = yaml.safe_load(open(config_path))

    def get_document_store(self):
        document_store = InMemoryDocumentStore()
        return document_store

    def get_generator_model(self):
        if self.config["inference_platform"] == "ollama":
            return OllamaChatGenerator(self.config["ollama_generator_model"], generation_kwargs={"num_ctx": 4096})
        elif self.config["inference_platform"] == "hg_api":
            return HuggingFaceAPIChatGenerator(api_type="serverless_inference_api",
                                        api_params={"model": self.config["hg_generator_model"]},
                                        token=Secret.from_env_var(["HG_API_KEY"]))
        else:
            raise ValueError("Inference platform", self.config["inference_platform"], "unknown")
        
    def get_joint_document_embedder(self, **kwargs):
        if self.config["embedder_platform"] == "local":
            return SentenceTransformersJointEmbedder(model=self.config["embedder_model"], **kwargs)
        
        elif self.config["embedder_platform"] == "hg_api":
            return HuggingfaceAPIJointEmbedder(api_type="serverless_inference_api",
                                        api_params={"model": self.config["embedder_model"]},
                                        token=Secret.from_env_var(["HG_API_KEY"]), **kwargs)
        