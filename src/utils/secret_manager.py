import os
import base64
import logging
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

logger = logging.getLogger(__name__)

class SecretManager:
    """
    Manages encryption and decryption of sensitive data like API keys.
    Uses a master key derived from an environment variable.
    """
    
    _fernet = None

    @classmethod
    def _get_fernet(cls):
        if cls._fernet is None:
            # Try to get master key from environment
            master_key = os.getenv("STOIC_MASTER_KEY")
            if not master_key:
                logger.warning("STOIC_MASTER_KEY not set. Using default development key. NOT SECURE FOR PRODUCTION!")
                master_key = "dev-secret-key-do-not-use-in-production"
            
            # Derive a stable 32-byte key from the master key
            salt = b'stoic_citadel_salt' # In a real app, this would be stored separately
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
    def encrypt(cls, data: str) -> str:
        """Encrypt a string."""
        if not data:
            return ""
        f = cls._get_fernet()
        # Add a prefix to identify encrypted data
        encrypted = f.encrypt(data.encode()).decode()
        return f"ENC:{encrypted}"

    @classmethod
    def decrypt(cls, data: str) -> str:
        """Decrypt a string if it's encrypted."""
        if not data or not data.startswith("ENC:"):
            return data
        
        try:
            f = cls._get_fernet()
            encrypted_part = data[4:]
            return f.decrypt(encrypted_part.encode()).decode()
        except Exception as e:
            logger.error(f"Failed to decrypt secret: {e}")
            return data # Return original if decryption fails

def main():
    """CLI tool for encrypting secrets."""
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m src.utils.secret_manager <text_to_encrypt>")
        return

    text = sys.argv[1]
    encrypted = SecretManager.encrypt(text)
    print(f"\nEncrypted value:\n{encrypted}\n")
    print("Copy this value to your config.json or .env file.")

if __name__ == "__main__":
    main()
