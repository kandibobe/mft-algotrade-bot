
import logging
import os

import hvac
from hvac.exceptions import Forbidden, InvalidPath

logger = logging.getLogger(__name__)

class VaultClient:
    """
    A client for interacting with HashiCorp Vault.
    """
    _client = None

    @classmethod
    def _get_client(cls):
        """
        Initializes and returns a singleton Vault client instance.

        Reads connection details from environment variables.
        """
        if cls._client is None:
            vault_addr = os.getenv("VAULT_ADDR")
            vault_token = os.getenv("VAULT_TOKEN")

            if not vault_addr or not vault_token:
                logger.debug("VAULT_ADDR or VAULT_TOKEN not set. VaultClient is disabled.")
                return None

            try:
                cls._client = hvac.Client(url=vault_addr, token=vault_token)
                if not cls._client.is_authenticated():
                    raise Exception("Vault authentication failed.")
                logger.info(f"Successfully connected to Vault at {vault_addr}")
            except Exception as e:
                logger.error(f"Failed to initialize Vault client: {e}")
                cls._client = None # Ensure client is None on failure

        return cls._client

    @classmethod
    def get_secret(cls, path: str, key: str = "value") -> str | None:
        """
        Retrieves a secret from Vault's KV version 2 engine.

        Args:
            path: The path to the secret in the KV store (e.g., 'exchange/binance').
                  The 'kv/' prefix is assumed.
            key: The specific key within the secret to retrieve.

        Returns:
            The secret value as a string, or None if not found or on error.
        """
        client = cls._get_client()
        if not client:
            return None

        try:
            # Defaulting to 'kv' mount point, which is standard
            response = client.secrets.kv.v2.read_secret_version(path=path)

            secret_value = response['data']['data'].get(key)

            if secret_value:
                logger.debug(f"Successfully retrieved secret from path: {path}")
            else:
                logger.warning(f"Secret key '{key}' not found at path: {path}")

            return secret_value
        except (InvalidPath, Forbidden) as e:
            logger.error(f"Permission or path error for Vault secret at {path}: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to retrieve secret from Vault at path {path}: {e}")
            return None

def is_vault_available() -> bool:
    """
    Check if the Vault client is configured and available.
    """
    return VaultClient._get_client() is not None
