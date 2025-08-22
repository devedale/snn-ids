import socket
import struct
from hashlib import sha256

# AES semplificato tramite SHA-256 per evitare dipendenze
# Nota: per produzione è meglio una libreria AES vera, qui usiamo SHA-256 per creare un PRF deterministico

def pseudo_random_function(key: bytes, data: bytes) -> bytes:
    return sha256(key + data).digest()

def ip_to_int(ip: str) -> int:
    return struct.unpack("!I", socket.inet_aton(ip))[0]

def int_to_ip(i: int) -> str:
    return socket.inet_ntoa(struct.pack("!I", i))

def cryptopan_ip(ip: str, key: bytes) -> str:
    """
    Applica Crypto-PAn (semplificato) su un IPv4 stringa con una chiave segreta.
    key deve essere 16+ byte; useremo SHA-256 come PRF semplificata.
    """
    ip_int = ip_to_int(ip)
    anon_ip = 0

    for i in range(32):  # per ogni bit dell'IPv4
        prefix = (anon_ip >> (31 - i)) << (31 - i)
        prf_input = struct.pack("!I", prefix)
        prf_out = pseudo_random_function(key, prf_input)
        bit = (prf_out[0] >> 7) & 1  # prendiamo il bit più significativo
        orig_bit = (ip_int >> (31 - i)) & 1
        anon_bit = orig_bit ^ bit
        anon_ip |= (anon_bit << (31 - i))

    return int_to_ip(anon_ip)

# Esempio integrazione con DataFrame
def anonymize_ips_with_cryptopan(df, ip_columns, key):
    for col in ip_columns:
        df[col] = df[col].apply(lambda ip: cryptopan_ip(ip, key))
    return df

def deanonymize_ip(anonymized_ip: str, ip_map: dict) -> str:
    """
    Decrittografa un IP anonimizzato usando la mappa salvata.
    
    Args:
        anonymized_ip: L'IP anonimizzato
        ip_map: La mappa di anonimizzazione salvata
    
    Returns:
        L'IP originale o l'IP anonimizzato se non trovato
    """
    return ip_map.get("inverse_map", {}).get(anonymized_ip, anonymized_ip)

def deanonymize_ips_in_dataframe(df, ip_columns, ip_map):
    """
    Decrittografa tutti gli IP anonimizzati in un DataFrame.
    
    Args:
        df: Il DataFrame con IP anonimizzati
        ip_columns: Lista delle colonne contenenti IP
        ip_map: La mappa di anonimizzazione salvata
    
    Returns:
        DataFrame con IP decrittografati
    """
    df_copy = df.copy()
    for col in ip_columns:
        df_copy[col] = df_copy[col].apply(lambda ip: deanonymize_ip(str(ip), ip_map))
    return df_copy
