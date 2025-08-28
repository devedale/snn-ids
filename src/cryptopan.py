"""
Crypto-PAn Implementation for IP Address Anonymization
Prefix-preserving anonymization scheme (Xu et al., NDSS 2001)
"""

import hashlib
import hmac
import struct
from typing import Dict, Tuple, Optional
import json
import os
import numpy as np
import pandas as pd
from config import RANDOM_CONFIG

class CryptoPan:
    """
    Crypto-PAn implementation for prefix-preserving IP anonymization.
    Preserves subnet structure while anonymizing individual IP addresses.
    """
    
    def __init__(self, key: Optional[bytes] = None, save_mapping: bool = True):
        """
        Initialize Crypto-PAn with optional key.
        
        Args:
            key: 32-byte key for AES encryption. If None, generates random key.
            save_mapping: Whether to save anonymization mapping to file.
        """
        if key is None:
            # Generate deterministic key from global seed
            seed = RANDOM_CONFIG.get('seed', 42)
            key = hashlib.sha256(str(seed).encode()).digest()
        
        self.key = key
        self.save_mapping = save_mapping
        self.mapping_cache = {}  # Cache for performance
        self.reverse_mapping = {}  # For decryption
        
    def _pad_key(self, key: bytes) -> bytes:
        """Pad key to 32 bytes if necessary."""
        if len(key) >= 32:
            return key[:32]
        return key + b'\x00' * (32 - len(key))
    
    def _aes_encrypt(self, plaintext: bytes) -> bytes:
        """AES encryption using the key."""
        # Simplified AES implementation for Crypto-PAn
        # In production, use cryptography library
        key = self._pad_key(self.key)
        
        # For Crypto-PAn, we only need the first 4 bytes
        # This is a simplified version - real implementation would use proper AES
        hash_input = key + plaintext
        result = hashlib.sha256(hash_input).digest()
        return result[:4]
    
    def _ip_to_int(self, ip_str: str) -> int:
        """Convert IP string to 32-bit integer."""
        try:
            parts = ip_str.split('.')
            if len(parts) != 4:
                return 0
            
            ip_int = 0
            for i, part in enumerate(parts):
                octet = int(part)
                if not (0 <= octet <= 255):
                    return 0
                ip_int |= (octet << (24 - 8 * i))
            return ip_int
        except:
            return 0
    
    def _int_to_ip(self, ip_int: int) -> str:
        """Convert 32-bit integer to IP string."""
        try:
            octets = []
            for i in range(4):
                octet = (ip_int >> (24 - 8 * i)) & 0xFF
                octets.append(str(octet))
            return '.'.join(octets)
        except:
            return "0.0.0.0"
    
    def anonymize_ip(self, ip_str: str) -> str:
        """
        Anonymize IP address using Crypto-PAn.
        
        Args:
            ip_str: IP address string (e.g., "192.168.1.1")
            
        Returns:
            Anonymized IP address string
        """
        if ip_str in self.mapping_cache:
            return self.mapping_cache[ip_str]
        
        # Convert IP to integer
        ip_int = self._ip_to_int(ip_str)
        if ip_int == 0:
            return ip_str  # Return original if conversion fails
        
        # Crypto-PAn algorithm
        anonymized_int = self._cryptopan_algorithm(ip_int)
        
        # Convert back to IP string
        anonymized_ip = self._int_to_ip(anonymized_int)
        
        # Cache result
        self.mapping_cache[ip_str] = anonymized_ip
        self.reverse_mapping[anonymized_ip] = ip_str
        
        return anonymized_ip
    
    def _cryptopan_algorithm(self, ip_int: int) -> int:
        """
        Core Crypto-PAn algorithm implementation.
        
        Args:
            ip_int: 32-bit IP address as integer
            
        Returns:
            Anonymized 32-bit IP address as integer
        """
        result = 0
        
        # Process each bit from left to right (MSB to LSB)
        for i in range(32):
            # Extract current bit
            current_bit = (ip_int >> (31 - i)) & 1
            
            # Build the prefix up to current position
            prefix = (ip_int >> (31 - i)) << (31 - i)
            
            # Encrypt the prefix
            prefix_bytes = struct.pack('>I', prefix)
            encrypted = self._aes_encrypt(prefix_bytes)
            encrypted_int = struct.unpack('>I', encrypted + b'\x00\x00\x00\x00')[:1][0]
            
            # Extract the corresponding bit from encrypted result
            encrypted_bit = (encrypted_int >> (31 - i)) & 1
            
            # XOR with current bit to get anonymized bit
            anonymized_bit = current_bit ^ encrypted_bit
            
            # Set the bit in result
            result |= (anonymized_bit << (31 - i))
        
        return result
    
    def deanonymize_ip(self, anonymized_ip: str) -> str:
        """
        Reverse anonymization (requires mapping to be saved).
        
        Args:
            anonymized_ip: Anonymized IP address string
            
        Returns:
            Original IP address string
        """
        if anonymized_ip in self.reverse_mapping:
            return self.reverse_mapping[anonymized_ip]
        
        # If no mapping available, return as-is
        return anonymized_ip
    
    def save_mapping(self, output_dir: str, filename: str = "ip_anonymization_map.json"):
        """Save anonymization mapping to file."""
        if not self.save_mapping:
            return
        
        os.makedirs(output_dir, exist_ok=True)
        mapping_file = os.path.join(output_dir, filename)
        
        # Convert to serializable format
        serializable_mapping = {
            'forward': {str(k): str(v) for k, v in self.mapping_cache.items()},
            'reverse': {str(k): str(v) for k, v in self.reverse_mapping.items()},
            'key_hash': hashlib.sha256(self.key).hexdigest(),
            'total_ips': len(self.mapping_cache)
        }
        
        with open(mapping_file, 'w') as f:
            json.dump(serializable_mapping, f, indent=2)
        
        print(f"✅ IP anonymization mapping saved to: {mapping_file}")
    
    def load_mapping(self, mapping_file: str):
        """Load anonymization mapping from file."""
        try:
            with open(mapping_file, 'r') as f:
                mapping_data = json.load(f)
            
            self.mapping_cache = mapping_data.get('forward', {})
            self.reverse_mapping = mapping_data.get('reverse', {})
            
            print(f"✅ IP anonymization mapping loaded from: {mapping_file}")
            print(f"   Loaded {len(self.mapping_cache)} IP mappings")
            
        except Exception as e:
            print(f"⚠️ Could not load mapping file: {e}")
    
    def get_stats(self) -> Dict[str, any]:
        """Get anonymization statistics."""
        return {
            'total_ips_processed': len(self.mapping_cache),
            'key_hash': hashlib.sha256(self.key).hexdigest()[:16],
            'cache_hit_rate': len(self.mapping_cache) / max(1, len(self.mapping_cache))
        }