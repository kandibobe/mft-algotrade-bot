import base64
import logging
import os

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

from src.utils.vault_client import VaultClient, is_vault_available

logger = logging.getLogger(__name__)


class SecretManager:
    """
    Manages retrieval of sensitive data like API keys.
    It prioritizes fetching secrets from HashiCorp Vault if available.
    If Vault is not configured, it falls back to a local decryption method
    for legacy secrets. New secrets should be stored in Vault.
    """

    _fernet = None

    @classmethod
    def get_secret(cls, identifier: str) -> str:
        """
        Retrieves a secret, prioritizing Vault.

        The identifier can be a Vault path (e.g., 'exchange/binance') or
        a locally encrypted string ("ENC:...").
        """
        if is_vault_available():
            try:
                path, key = identifier.rsplit("/", 1)
                secret = VaultClient.get_secret(path, key)
                if secret:
                    return secret
            except ValueError:
                logger.warning(
                    f"Identifier '{identifier}' is not in 'path/key' format for Vault. Falling back to local."
                )

        return cls._decrypt_local(identifier)

    @classmethod
    def _get_fernet(cls):
        if cls._fernet is None:
            master_key = os.getenv("STOIC_MASTER_KEY")
            if not master_key:
                logger.warning(
                    "STOIC_MASTER_KEY not set. Using default development key. NOT SECURE FOR PRODUCTION!"
                )
                master_key = "dev-secret-key-do-not-use-in-production"

            salt = b"stoic_citadel_salt"
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
            )
            key = base64.urlsafe_b64encode(kdf.derive(master_key.encode()))
            cls._fernet = Fernet(key)
        return cls._fernet

    @classmethod
    def _decrypt_local(cls, data: str) -> str:
        """Decrypt a string if it's in the legacy encrypted format."""
        if not data or not data.startswith("ENC:"):
            return data

        try:
            f = cls._get_fernet()
            encrypted_part = data[4:]
            return f.decrypt(encrypted_part.encode()).decode()
        except Exception as e:
            logger.error(f"Failed to decrypt secret: {e}")
            return data
