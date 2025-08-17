#!/usr/bin/env python3
"""
Parser universale per log multi-formato
Supporta: FortiGate (tlog/elog), CSV, CEF, Syslog RFC3164, Syslog RFC5424
"""

import re
import csv
import json
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LogFormat(Enum):
    """Tipi di formato log supportati"""
    FORTIGATE = "fortigate"
    CSV = "csv"
    CEF = "cef"
    SYSLOG_RFC3164 = "syslog_rfc3164"
    SYSLOG_RFC5424 = "syslog_rfc5424"
    UNKNOWN = "unknown"


@dataclass
class ParsedLog:
    """Struttura dati per un log parsato"""
    format_type: LogFormat
    timestamp: Optional[datetime]
    raw_message: str
    parsed_fields: Dict[str, Any]
    metadata: Dict[str, Any]


class FortiGateParser:
    """Parser per log FortiGate (formato chiave=valore)"""
    
    @staticmethod
    def parse(line: str) -> Dict[str, Any]:
        """
        Parsa una riga di log FortiGate
        Gestisce campi variabili nel formato chiave=valore
        """
        fields = {}
        
        # Pattern per catturare chiave=valore, gestendo valori quotati
        pattern = r'(\w+)=(?:"([^"]*)"|([^\s]+))'
        
        for match in re.finditer(pattern, line):
            key = match.group(1)
            # Usa il valore quotato se presente, altrimenti quello non quotato
            value = match.group(2) if match.group(2) is not None else match.group(3)
            
            # Mantieni tutti i valori come stringhe per preservare gli zero iniziali
            # e la formattazione originale
            fields[key] = value
        
        return fields


class CEFParser:
    """Parser per Common Event Format (CEF)"""
    
    @staticmethod
    def parse(line: str) -> Dict[str, Any]:
        """
        Parsa una riga CEF
        Formato: CEF:Version|Device Vendor|Device Product|Device Version|Device Event Class ID|Name|Severity|Extension
        """
        fields = {}
        
        if not line.startswith('CEF:'):
            return fields
        
        # Rimuovi il prefisso CEF:
        cef_content = line[4:]
        
        # Separa header e extension
        parts = cef_content.split('|')
        
        if len(parts) >= 7:
            fields['cef_version'] = parts[0]
            fields['device_vendor'] = parts[1]
            fields['device_product'] = parts[2]
            fields['device_version'] = parts[3]
            fields['device_event_class_id'] = parts[4]
            fields['name'] = parts[5]
            fields['severity'] = parts[6]
            
            # Parsa le estensioni se presenti
            if len(parts) > 7:
                extension = '|'.join(parts[7:])
                fields.update(CEFParser._parse_extension(extension))
        
        return fields
    
    @staticmethod
    def _parse_extension(extension: str) -> Dict[str, Any]:
        """Parsa la parte extension del CEF"""
        ext_fields = {}
        
        # Pattern per catturare chiave=valore nell'extension
        pattern = r'(\w+)=([^=]+?)(?=\s+\w+=|$)'
        
        for match in re.finditer(pattern, extension):
            key = match.group(1)
            value = match.group(2).strip()
            ext_fields[key] = value
        
        return ext_fields


class SyslogParser:
    """Parser per Syslog RFC3164 e RFC5424"""
    
    # Priority pattern per estrarre facility e severity
    PRIORITY_PATTERN = r'^<(\d+)>'
    
    # RFC3164 pattern: <PRI>MMM DD HH:MM:SS hostname tag: message
    RFC3164_PATTERN = r'^<(\d+)>(\w{3}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2})\s+(\S+)\s+([^:]+):\s*(.*)'
    
    # RFC5424 pattern: <PRI>VERSION TIMESTAMP HOSTNAME APP-NAME PROCID MSGID STRUCTURED-DATA MSG
    RFC5424_PATTERN = r'^<(\d+)>(\d+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s*(.*)'
    
    @staticmethod
    def parse_rfc3164(line: str) -> Dict[str, Any]:
        """Parsa syslog RFC3164"""
        fields = {}
        
        match = re.match(SyslogParser.RFC3164_PATTERN, line)
        if match:
            priority = int(match.group(1))
            fields['facility'] = priority >> 3
            fields['severity'] = priority & 0x07
            fields['timestamp_str'] = match.group(2)
            fields['hostname'] = match.group(3)
            fields['tag'] = match.group(4)
            fields['message'] = match.group(5)
        
        return fields
    
    @staticmethod
    def parse_rfc5424(line: str) -> Dict[str, Any]:
        """Parsa syslog RFC5424"""
        fields = {}
        
        match = re.match(SyslogParser.RFC5424_PATTERN, line)
        if match:
            priority = int(match.group(1))
            fields['facility'] = priority >> 3
            fields['severity'] = priority & 0x07
            fields['version'] = match.group(2)
            fields['timestamp_str'] = match.group(3)
            fields['hostname'] = match.group(4)
            fields['app_name'] = match.group(5)
            fields['procid'] = match.group(6)
            fields['msgid'] = match.group(7)
            fields['structured_data'] = match.group(8)
            fields['message'] = match.group(9) if match.group(9) else ""
        
        return fields


class CSVParser:
    """Parser per file CSV"""
    
    @staticmethod
    def parse_csv_file(file_path: str) -> List[Dict[str, Any]]:
        """Parsa un intero file CSV"""
        results = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                # Rileva automaticamente il delimitatore
                sample = f.read(1024)
                f.seek(0)
                
                sniffer = csv.Sniffer()
                delimiter = sniffer.sniff(sample).delimiter
                
                reader = csv.DictReader(f, delimiter=delimiter)
                
                for row in reader:
                    # Rimuovi spazi bianchi dalle chiavi e valori
                    cleaned_row = {k.strip(): v.strip() for k, v in row.items() if k}
                    results.append(cleaned_row)
        
        except Exception as e:
            logger.error(f"Errore nel parsing CSV {file_path}: {e}")
        
        return results


class UniversalLogParser:
    """Parser universale che rileva automaticamente il formato"""
    
    def __init__(self):
        self.fortigate_parser = FortiGateParser()
        self.cef_parser = CEFParser()
        self.syslog_parser = SyslogParser()
        self.csv_parser = CSVParser()
    
    def detect_format(self, line: str) -> LogFormat:
        """Rileva automaticamente il formato del log"""
        line_stripped = line.strip()
        
        # CEF
        if line_stripped.startswith('CEF:'):
            return LogFormat.CEF
        
        # Syslog (RFC3164 e RFC5424) - inizia con <priority>
        if re.match(r'^<\d+>', line_stripped):
            # RFC5424 ha version number dopo priority
            if re.match(r'^<\d+>\d+\s', line_stripped):
                return LogFormat.SYSLOG_RFC5424
            else:
                return LogFormat.SYSLOG_RFC3164
        
        # FortiGate - contiene pattern logver= all'inizio
        if 'logver=' in line_stripped and '=' in line_stripped:
            return LogFormat.FORTIGATE
        
        # CSV - se contiene virgole e sembra strutturato
        if ',' in line_stripped and len(line_stripped.split(',')) > 3:
            return LogFormat.CSV
        
        return LogFormat.UNKNOWN
    
    def parse_line(self, line: str, format_hint: Optional[LogFormat] = None) -> ParsedLog:
        """
        Parsa una singola riga di log
        
        Args:
            line: la riga da parsare
            format_hint: suggerimento sul formato (opzionale)
        """
        line = line.strip()
        if not line:
            return ParsedLog(
                format_type=LogFormat.UNKNOWN,
                timestamp=None,
                raw_message=line,
                parsed_fields={},
                metadata={}
            )
        
        # Rileva il formato se non specificato
        detected_format = format_hint or self.detect_format(line)
        
        parsed_fields = {}
        timestamp = None
        metadata = {'detected_format': detected_format.value}
        
        try:
            if detected_format == LogFormat.FORTIGATE:
                parsed_fields = self.fortigate_parser.parse(line)
                timestamp = self._extract_fortigate_timestamp(parsed_fields)
            
            elif detected_format == LogFormat.CEF:
                parsed_fields = self.cef_parser.parse(line)
                timestamp = self._extract_cef_timestamp(parsed_fields)
            
            elif detected_format == LogFormat.SYSLOG_RFC3164:
                parsed_fields = self.syslog_parser.parse_rfc3164(line)
                timestamp = self._extract_syslog_timestamp(parsed_fields)
            
            elif detected_format == LogFormat.SYSLOG_RFC5424:
                parsed_fields = self.syslog_parser.parse_rfc5424(line)
                timestamp = self._extract_syslog_timestamp(parsed_fields)
            
            else:
                metadata['error'] = 'Formato non riconosciuto'
        
        except Exception as e:
            metadata['error'] = str(e)
            logger.error(f"Errore nel parsing: {e}")
        
        return ParsedLog(
            format_type=detected_format,
            timestamp=timestamp,
            raw_message=line,
            parsed_fields=parsed_fields,
            metadata=metadata
        )
    
    def parse_file(self, file_path: str, format_hint: Optional[LogFormat] = None) -> List[ParsedLog]:
        """
        Parsa un intero file di log
        
        Args:
            file_path: percorso del file
            format_hint: suggerimento sul formato (opzionale)
        """
        results = []
        
        # Gestione speciale per CSV
        if format_hint == LogFormat.CSV or file_path.endswith('.csv'):
            csv_data = self.csv_parser.parse_csv_file(file_path)
            for row in csv_data:
                parsed_log = ParsedLog(
                    format_type=LogFormat.CSV,
                    timestamp=self._extract_csv_timestamp(row),
                    raw_message=str(row),
                    parsed_fields=row,
                    metadata={'source_file': file_path}
                )
                results.append(parsed_log)
            return results
        
        # Per altri formati, parsa riga per riga
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    parsed_log = self.parse_line(line, format_hint)
                    parsed_log.metadata.update({
                        'source_file': file_path,
                        'line_number': line_num
                    })
                    results.append(parsed_log)
        
        except Exception as e:
            logger.error(f"Errore nell'apertura del file {file_path}: {e}")
        
        return results
    
    def _extract_fortigate_timestamp(self, fields: Dict[str, Any]) -> Optional[datetime]:
        """Estrae timestamp dai campi FortiGate"""
        try:
            if 'date' in fields and 'time' in fields:
                datetime_str = f"{fields['date']} {fields['time']}"
                return datetime.strptime(datetime_str, "%Y-%m-%d %H:%M:%S")
            elif 'eventtime' in fields:
                # eventtime è in nanoseconds since epoch (ora è stringa)
                timestamp_ns = int(str(fields['eventtime']))
                return datetime.fromtimestamp(timestamp_ns / 1_000_000_000)
        except (ValueError, TypeError) as e:
            logger.debug(f"Errore nell'estrazione timestamp FortiGate: {e}")
        return None
    
    def _extract_cef_timestamp(self, fields: Dict[str, Any]) -> Optional[datetime]:
        """Estrae timestamp dai campi CEF"""
        # CEF può avere vari campi timestamp, cerchiamo i più comuni
        timestamp_fields = ['rt', 'start', 'end', 'deviceReceiptTime']
        
        for field in timestamp_fields:
            if field in fields:
                try:
                    # Assumiamo timestamp in millisecondi (ora è stringa)
                    ts = int(str(fields[field]))
                    return datetime.fromtimestamp(ts / 1000)
                except (ValueError, TypeError):
                    pass
        return None
    
    def _extract_syslog_timestamp(self, fields: Dict[str, Any]) -> Optional[datetime]:
        """Estrae timestamp dai campi syslog"""
        if 'timestamp_str' in fields:
            try:
                ts_str = fields['timestamp_str']
                # RFC5424 format
                if 'T' in ts_str:
                    return datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
                # RFC3164 format (MMM DD HH:MM:SS)
                else:
                    current_year = datetime.now().year
                    return datetime.strptime(f"{current_year} {ts_str}", "%Y %b %d %H:%M:%S")
            except ValueError as e:
                logger.debug(f"Errore nell'estrazione timestamp syslog: {e}")
        return None
    
    def _extract_csv_timestamp(self, row: Dict[str, Any]) -> Optional[datetime]:
        """Estrae timestamp dai campi CSV"""
        timestamp_fields = ['Event Time', 'Timestamp', 'Date', 'DateTime', 'Time']
        
        for field in timestamp_fields:
            if field in row and row[field]:
                try:
                    # Prova vari formati
                    ts_str = row[field].strip('"')
                    
                    # ISO format con T
                    if 'T' in ts_str:
                        return datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
                    
                    # Altri formati comuni
                    formats = [
                        "%Y-%m-%d %H:%M:%S",
                        "%Y/%m/%d %H:%M:%S",
                        "%d/%m/%Y %H:%M:%S",
                        "%Y-%m-%d",
                    ]
                    
                    for fmt in formats:
                        try:
                            return datetime.strptime(ts_str, fmt)
                        except ValueError:
                            continue
                
                except Exception as e:
                    logger.debug(f"Errore nell'estrazione timestamp CSV: {e}")
        
        return None


def main():
    """Funzione principale per test e dimostrazione"""
    from pathlib import Path
    
    # Crea directory output se non esiste
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    parser = UniversalLogParser()
    
    # Test con i file di esempio
    test_files = [
        "/home/wls_user/tesi/input/FGT80FTK22013405.root.tlog.txt",
        "/home/wls_user/tesi/input/FGT80FTK22013405.root.elog.txt",
        "/home/wls_user/tesi/input/aaa9b30098d31c056625c603cf1a98e6e10afe77_2025-06-30_2025-07-07.csv"
    ]
    
    for file_path in test_files:
        print(f"\n=== Parsing {file_path} ===")
        
        try:
            results = parser.parse_file(file_path)
            
            print(f"Parsed {len(results)} entries")
            
            # Mostra il primo risultato come esempio
            if results:
                first_result = results[0]
                print(f"Formato rilevato: {first_result.format_type.value}")
                print(f"Timestamp: {first_result.timestamp}")
                print(f"Campi principali: {list(first_result.parsed_fields.keys())[:10]}")
                
                # Mostra alcuni campi interessanti
                if first_result.format_type == LogFormat.FORTIGATE:
                    print(f"Device: {first_result.parsed_fields.get('devname', 'N/A')}")
                    print(f"Type: {first_result.parsed_fields.get('type', 'N/A')}")
                elif first_result.format_type == LogFormat.CSV:
                    print(f"Action Type: {first_result.parsed_fields.get('Action Type', 'N/A')}")
                    print(f"Computer Name: {first_result.parsed_fields.get('Computer Name', 'N/A')}")
        
        except Exception as e:
            print(f"Errore nel parsing di {file_path}: {e}")


if __name__ == "__main__":
    main()
