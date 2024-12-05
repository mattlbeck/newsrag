import os

import yaml
from haystack.components.embedders import (HuggingFaceAPITextEmbedder,
                                           SentenceTransformersTextEmbedder)
from haystack.components.generators.chat import HuggingFaceAPIChatGenerator
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.utils import Secret
from haystack_integrations.components.generators.ollama import \
    OllamaChatGenerator

from newsrag.topics import (HuggingfaceAPIJointEmbedder,
                            SentenceTransformersJointEmbedder)


class AppConfig:
    """
    Manage the configuration of the app from a YAML config file and env vars.
    
    env var counterparts follow the convention "APP_<UPPER_CASE_KEY>"
    """

    def __init__(self, config_path: str="config.yaml"):
        self.config = yaml.safe_load(open(config_path))

        # overwrite file config options with any stored as env vars
        for key in self.config.keys():
            env_key = "APP_" + key.upper()
            if env_key in os.environ:
                self.config[key] = os.environ[env_key]

        # add the API key, which should not be present in the config file
        self._hg_api_key = Secret.from_env_var(["HG_API_KEY"])
        print(self.config)

    def get_document_store(self):
        document_store = InMemoryDocumentStore()
        return document_store

    def get_generator_model(self):
        if self.config["inference_platform"] == "ollama":
            return OllamaChatGenerator(self.config["ollama_generator_model"], generation_kwargs={"num_ctx": 4096})
        elif self.config["inference_platform"] == "hg_api":
            return HuggingFaceAPIChatGenerator(api_type="serverless_inference_api",
                                        api_params={"model": self.config["hg_generator_model"]},
                                        token=self._hg_api_key)
        else:
            raise ValueError("Inference platform", self.config["inference_platform"], "unknown")
        
    def get_joint_document_embedder(self, **kwargs):
        if self.config["embedder_platform"] == "local":
            return SentenceTransformersJointEmbedder(model=self.config["embedder_model"], **kwargs)
        
        elif self.config["embedder_platform"] == "hg_api":
            print("using HG API embedder")
            return HuggingfaceAPIJointEmbedder(api_type="serverless_inference_api",
                                        api_params={"model": self.config["embedder_model"]},
                                        token=self._hg_api_key, **kwargs)
        
    def get_text_embedder(self, **kwargs):
        if self.config["embedder_platform"] == "local":
            return SentenceTransformersTextEmbedder(model=self.config["embedder_model"], **kwargs)
        
        elif self.config["embedder_platform"] == "hg_api":
            return HuggingFaceAPITextEmbedder(api_type="serverless_inference_api",
                                        api_params={"model": self.config["embedder_model"]},
                                        token=self._hg_api_key, **kwargs)
        