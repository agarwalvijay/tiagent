"""PDF parser for TI datasheets."""
import re
import fitz  # PyMuPDF
import pdfplumber
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path

# Optional import for parametrics parser
try:
    from backend.parsers.parametrics_parser import ParametricsParser
except ImportError:
    ParametricsParser = None


@dataclass
class DatasheetMetadata:
    """Structured metadata extracted from datasheet."""
    document_id: str
    revision: str
    part_numbers: List[str] = field(default_factory=list)
    device_type: Optional[str] = None
    architecture: Optional[str] = None
    core_freq_mhz: Optional[float] = None
    flash_kb: List[int] = field(default_factory=list)
    ram_kb: List[int] = field(default_factory=list)
    operating_temp_min_c: Optional[float] = None
    operating_temp_max_c: Optional[float] = None
    voltage_min_v: Optional[float] = None
    voltage_max_v: Optional[float] = None
    package_types: List[str] = field(default_factory=list)
    key_features: List[str] = field(default_factory=list)
    peripherals: List[str] = field(default_factory=list)
    applications: List[str] = field(default_factory=list)
    adc_bits: List[int] = field(default_factory=list)
    # Datasheet links
    pdf_datasheet_url: Optional[str] = None
    html_datasheet_url: Optional[str] = None
    product_page_url: Optional[str] = None
    datasheet_link: Optional[str] = None  # Best available link
    # Pricing and lifecycle
    price_usd: Optional[float] = None
    status: Optional[str] = None  # Lifecycle status (ACTIVE, PREVIEW, NRND, etc.)
    pin_count: Optional[str] = None  # Pin counts (e.g., "28,32,48,64")

    def to_dict(self) -> Dict:
        """Convert to dictionary for ChromaDB metadata."""
        return {
            "document_id": self.document_id,
            "revision": self.revision,
            "part_numbers": ",".join(self.part_numbers),
            "device_type": self.device_type or "",
            "architecture": self.architecture or "",
            "core_freq_mhz": self.core_freq_mhz or 0,
            "flash_kb_min": min(self.flash_kb) if self.flash_kb else 0,
            "flash_kb_max": max(self.flash_kb) if self.flash_kb else 0,
            "ram_kb_min": min(self.ram_kb) if self.ram_kb else 0,
            "ram_kb_max": max(self.ram_kb) if self.ram_kb else 0,
            "operating_temp_min_c": self.operating_temp_min_c or -999,
            "operating_temp_max_c": self.operating_temp_max_c or 999,
            "voltage_min_v": self.voltage_min_v or 0,
            "voltage_max_v": self.voltage_max_v or 0,
            "package_types": ",".join(self.package_types),
            "key_features": ",".join(self.key_features),
            "peripherals": ",".join(self.peripherals),
            "applications": ",".join(self.applications),
            "adc_bits": ",".join(map(str, self.adc_bits)),
            "pdf_datasheet_url": self.pdf_datasheet_url or "",
            "html_datasheet_url": self.html_datasheet_url or "",
            "product_page_url": self.product_page_url or "",
            "datasheet_link": self.datasheet_link or "",
            "price_usd": self.price_usd or 0,
            "status": self.status or "",
            "pin_count": self.pin_count or "",
        }


@dataclass
class DatasheetSection:
    """A section of the datasheet."""
    title: str
    level: int  # Hierarchical level (1, 2, 3, etc.)
    page_start: int
    page_end: Optional[int]
    content: str
    section_number: Optional[str] = None  # e.g., "1.2.3"


@dataclass
class DatasheetChunk:
    """A semantic chunk for vector storage."""
    chunk_id: str
    chunk_type: str  # overview, features, specs, peripheral, pins
    content: str
    metadata: Dict
    page_numbers: List[int]


class TIDatasheetParser:
    """Parser for Texas Instruments datasheets."""

    # Common TI section headers (expanded patterns)
    SECTION_PATTERNS = {
        "features": re.compile(r"^\d*\.?\s*Features?\s*(?:and\s+Benefits)?", re.IGNORECASE),
        "applications": re.compile(r"^\d*\.?\s*Applications?\s*(?:and\s+Description)?", re.IGNORECASE),
        "description": re.compile(r"^\d*\.?\s*(?:Device\s+)?Descriptions?\s*$", re.IGNORECASE),
        "device_comparison": re.compile(r"^\d*\.?\s*Device\s+Comparisons?\s*(?:Table)?", re.IGNORECASE),
        "pin_config": re.compile(r"^\d*\.?\s*Pin\s+(?:Configuration|Functions?|Assignments?|Descriptions?)", re.IGNORECASE),
        "specifications": re.compile(r"^\d*\.?\s*(?:Electrical\s+)?(?:Specifications?|Characteristics?)", re.IGNORECASE),
        "detailed_description": re.compile(r"^\d*\.?\s*Detailed\s+Descriptions?\s*$", re.IGNORECASE),
        "functional_block": re.compile(r"^\d*\.?\s*Functional\s+(?:Block\s+Diagram|Description)", re.IGNORECASE),
        "applications_impl": re.compile(r"^\d*\.?\s*Applications?,?\s+(?:Implementation|Information)", re.IGNORECASE),
        "power": re.compile(r"^\d*\.?\s*Power\s+(?:Consumption|Management|Supply)", re.IGNORECASE),
        "mechanical": re.compile(r"^\d*\.?\s*(?:Package|Mechanical)\s+(?:Information|Outline|Dimensions?)", re.IGNORECASE),
        "absolute_max": re.compile(r"^\d*\.?\s*Absolute\s+Maximum\s+Ratings?", re.IGNORECASE),
        "recommended_op": re.compile(r"^\d*\.?\s*Recommended\s+Operating\s+Conditions?", re.IGNORECASE),
        "thermal": re.compile(r"^\d*\.?\s*Thermal\s+(?:Information|Characteristics?)", re.IGNORECASE),
        "timing": re.compile(r"^\d*\.?\s*Timing\s+(?:Diagrams?|Requirements?|Characteristics?)", re.IGNORECASE),
    }

    def __init__(self, pdf_path: str, parametrics_parser: Optional['ParametricsParser'] = None):
        """
        Initialize parser with PDF path.

        Args:
            pdf_path: Path to the PDF datasheet
            parametrics_parser: Optional ParametricsParser for authoritative metadata
        """
        self.pdf_path = Path(pdf_path)
        self.doc = fitz.open(str(pdf_path))
        self.metadata: Optional[DatasheetMetadata] = None
        self.sections: List[DatasheetSection] = []
        self.parametrics_parser = parametrics_parser

    def parse(self) -> Tuple[DatasheetMetadata, List[DatasheetChunk]]:
        """Parse the entire datasheet and return metadata and chunks."""
        # Extract metadata from first few pages
        self.metadata = self._extract_metadata()

        # Extract sections
        self.sections = self._extract_sections()

        # Create chunks
        chunks = self._create_chunks()

        return self.metadata, chunks

    def _extract_metadata(self) -> DatasheetMetadata:
        """Extract structured metadata from the first 5 pages."""
        # Get PDF metadata
        pdf_meta = self.doc.metadata

        # Extract text from first 5 pages
        first_pages_text = ""
        for page_num in range(min(5, len(self.doc))):
            page = self.doc[page_num]
            first_pages_text += page.get_text()

        # Extract document ID and revision from title or keywords
        doc_id, revision = self._extract_doc_id_and_revision(pdf_meta, first_pages_text)

        metadata = DatasheetMetadata(
            document_id=doc_id,
            revision=revision
        )

        # Extract part numbers (always from PDF)
        metadata.part_numbers = self._extract_part_numbers(first_pages_text)

        # Fallback: Use document ID if no part numbers found
        if not metadata.part_numbers and metadata.document_id:
            metadata.part_numbers = [metadata.document_id]
            print(f"  ℹ️  Using document ID as part number: {metadata.document_id}")

        # Always extract from PDF first (some fields not in CSV)
        metadata.device_type = self._extract_device_type(first_pages_text)
        metadata.architecture = self._extract_architecture(first_pages_text)
        metadata.core_freq_mhz = self._extract_frequency(first_pages_text)
        metadata.flash_kb, metadata.ram_kb = self._extract_memory(first_pages_text)
        metadata.operating_temp_min_c, metadata.operating_temp_max_c = self._extract_temp_range(first_pages_text)
        metadata.voltage_min_v, metadata.voltage_max_v = self._extract_voltage_range(first_pages_text)
        metadata.peripherals = self._extract_peripherals(first_pages_text)
        metadata.applications = self._extract_applications(first_pages_text)
        metadata.key_features = self._extract_key_features(first_pages_text)
        metadata.adc_bits = self._extract_adc_bits(first_pages_text)

        # Override with parametrics data if available (more accurate for certain fields)
        if self.parametrics_parser and metadata.part_numbers:
            parametrics_data = self._get_parametrics_data(metadata.part_numbers)
            if parametrics_data:
                # Override with authoritative parametrics data
                self._enrich_from_parametrics(metadata, parametrics_data)

        return metadata

    def _get_parametrics_data(self, part_numbers: List[str]) -> Optional[Dict]:
        """
        Get parametrics data for the first matching part number.

        Args:
            part_numbers: List of part numbers extracted from PDF

        Returns:
            Parametrics data dict or None if not found
        """
        if not self.parametrics_parser:
            return None

        for part_num in part_numbers:
            data = self.parametrics_parser.get_part_data(part_num)
            if data:
                print(f"✓ Found parametrics data for {part_num}")
                return data

        print(f"✗ No parametrics data found for {', '.join(part_numbers)}")
        return None

    def _enrich_from_parametrics(self, metadata: DatasheetMetadata, param_data: Dict):
        """
        Enrich metadata with authoritative parametrics data.

        Args:
            metadata: The metadata object to enrich
            param_data: Parametrics data from CSV
        """
        # Core specs (use parametrics as authoritative)
        if param_data.get('core_freq_mhz'):
            freq_list = param_data['core_freq_mhz']
            metadata.core_freq_mhz = max(freq_list) if freq_list else None

        if param_data.get('flash_kb'):
            metadata.flash_kb = param_data['flash_kb']

        if param_data.get('ram_kb'):
            metadata.ram_kb = param_data['ram_kb']

        # Architecture
        if param_data.get('architecture'):
            metadata.architecture = param_data['architecture']

        # Temperature range
        if param_data.get('temp_range'):
            min_temp, max_temp = param_data['temp_range']
            metadata.operating_temp_min_c = min_temp
            metadata.operating_temp_max_c = max_temp

        # Package types
        if param_data.get('package_type'):
            pkg_types = param_data['package_type']
            if isinstance(pkg_types, str):
                metadata.package_types = [p.strip() for p in pkg_types.split(',')]

        # Device type (infer from description or CPU)
        if param_data.get('cpu'):
            metadata.device_type = "Microcontroller"

        # Features (combine parametrics features with PDF-extracted features)
        if param_data.get('features'):
            features_str = param_data['features']
            if isinstance(features_str, str):
                param_features = [f.strip() for f in features_str.split(',')]
                metadata.key_features.extend(param_features)

        # Peripherals (from parametrics)
        peripherals = []
        if param_data.get('uart_count'):
            peripherals.append(f"UART x{param_data['uart_count']}")
        if param_data.get('i2c_count'):
            peripherals.append(f"I2C x{param_data['i2c_count']}")
        if param_data.get('spi_count'):
            peripherals.append(f"SPI x{param_data['spi_count']}")
        if param_data.get('adc_type'):
            peripherals.append(param_data['adc_type'])

        if peripherals:
            metadata.peripherals.extend(peripherals)

        # Datasheet links (priority: PDF > HTML > Product Page)
        if param_data.get('pdf_url'):
            metadata.pdf_datasheet_url = param_data['pdf_url']
        if param_data.get('html_url'):
            metadata.html_datasheet_url = param_data['html_url']
        if param_data.get('product_url'):
            metadata.product_page_url = param_data['product_url']
        if param_data.get('datasheet_link'):
            metadata.datasheet_link = param_data['datasheet_link']

        # Pricing and lifecycle status
        if param_data.get('price_usd') is not None:
            metadata.price_usd = param_data['price_usd']
        if param_data.get('status'):
            metadata.status = param_data['status']

        # Pin count
        if param_data.get('pin_count'):
            pin_counts = param_data['pin_count']
            if isinstance(pin_counts, list):
                metadata.pin_count = ','.join(map(str, pin_counts))
            else:
                metadata.pin_count = str(pin_counts)

    def _extract_doc_id_and_revision(self, pdf_meta: Dict, text: str) -> Tuple[str, str]:
        """Extract document ID and revision."""
        # Try to find in keywords (TI format: SPRSPB1A)
        keywords = pdf_meta.get("keywords", "")
        doc_pattern = re.compile(r'\b([A-Z]{3,8}\d{2,5}[A-Z]?)\b')
        matches = doc_pattern.findall(keywords + " " + text[:500])

        if matches:
            doc_id = matches[0]
            # Extract revision (last letter)
            revision_match = re.search(r'Rev\.?\s*([A-Z])', text[:1000])
            revision = revision_match.group(1) if revision_match else doc_id[-1] if doc_id[-1].isalpha() else "A"
            doc_id_base = doc_id[:-1] if doc_id[-1].isalpha() else doc_id
            return doc_id_base, revision

        # Fallback: use filename
        filename = Path(self.pdf_path).stem
        return filename.upper(), "A"

    def _extract_part_numbers(self, text: str) -> List[str]:
        """
        Extract part numbers from text.
        Focuses on first 1000 characters (header area) where part numbers are typically shown.
        """
        part_numbers = set()

        # Focus on header area first (where blue text/part numbers typically are)
        header_text = text[:1000]  # First ~1000 chars usually contain header

        # Common TI part number patterns
        patterns = [
            # Specific product families
            r'\b(TMS320[A-Z]\d{4,5}[A-Z]*)\b',  # TMS320F28377D (DSPs)
            r'\b(F28[A-Z]\d{3,4}[A-Z]{1,2})\b', # F28E120SC, F28E120SB (C2000 MCUs)
            r'\b(MSPM0[A-Z]\d{4})\b',            # MSPM0G5187 (MSPM0 MCUs)
            r'\b(MSPM33[A-Z]\d{3,4}[A-Z]?)\b',  # MSPM33C321A (MSPM33 MCUs)
            r'\b(MSPM0[HG]\d{4})\b',             # MSPM0H3215, MSPM0G3507 (MSPM0 variants)
            r'\b(MSP32[A-Z]\d{3,4}[A-Z]\d?)\b', # MSP32G031C8 (MSP32 MCUs)
            r'\b(MSP430[A-Z]\d{3,4}[A-Z]?)\b',  # MSP430F5529 (MSP430 MCUs)
            r'\b(TDA\d+[A-Z]{1,3}(?:-Q1)?)\b',  # TDA4VH-Q1, TDA4AH-Q1 (Jacinto processors)
            r'\b(AM\d{3,4}[A-Z](?:-Q1)?)\b',    # AM3358-Q1, AM5728 (Sitara processors)
            r'\b(CC\d{4}[A-Z]?(?:-Q1)?)\b',     # CC2652R, CC2640-Q1 (Wireless MCUs)
            r'\b(LP\d{4}[A-Z](?:-Q1)?)\b',      # LP5907-Q1 (Power management)
            r'\b(TPS\d{4,5}[A-Z]?(?:-Q1)?)\b',  # TPS6594-Q1, TPS65217C (Power management)
            r'\b(LM\d{3,4}[A-Z]?(?:-Q1)?)\b',   # LM3481-Q1 (Power/Analog)
            r'\b(TLV\d{4}[A-Z]?(?:-Q1)?)\b',    # TLV3901, TLV6723 (Comparators/Op-Amps)
            r'\b(TCAN\d{4}[A-Z]?(?:-Q1)?)\b',   # TCAN5102, TCAN1473 (CAN transceivers)
            r'\b(TIOL\d{3}[A-Z]?(?:-Q1)?)\b',   # TIOL221 (IO-Link transceivers)
            r'\b(SN\d{5}[A-Z]{1,2})\b',         # SN74HC595 (Logic/Interface)
            r'\b(OPA\d{3,4}[A-Z]?(?:-Q1)?)\b',  # OPA2350 (Op-Amps)
            r'\b(INA\d{3}[A-Z]?(?:-Q1)?)\b',    # INA219 (Current/Power monitors)
            r'\b(F\d{5}[A-Z]{1,3}[A-Z]{3})\b',  # F28377DPTPSEP (some F-series)
        ]

        # First, extract from header area (highest priority - where blue text is)
        for pattern in patterns:
            matches = re.findall(pattern, header_text)
            part_numbers.update(matches)

        # If we didn't find enough in header, search full text
        if len(part_numbers) < 3:
            for pattern in patterns:
                matches = re.findall(pattern, text)
                part_numbers.update(matches)

        return list(part_numbers)[:15]  # Limit to 15 to catch variant families

    def _extract_device_type(self, text: str) -> Optional[str]:
        """Extract device type."""
        text_lower = text.lower()

        if "microcontroller" in text_lower or "mcu" in text_lower:
            if "mixed-signal" in text_lower or "mixed signal" in text_lower:
                return "Mixed-Signal Microcontroller"
            return "Microcontroller"
        elif "processor" in text_lower or "dsp" in text_lower:
            return "Processor"
        elif "analog" in text_lower:
            return "Analog"
        elif "power management" in text_lower:
            return "Power Management"

        return None

    def _extract_architecture(self, text: str) -> Optional[str]:
        """Extract CPU architecture."""
        # Common architectures
        arch_patterns = [
            (r'TMS320C\d{2}x', 'TMS320C28x'),
            (r'Arm®?\s*Cortex[®\-]M\d+\+?', None),  # Extract as-is
            (r'RISC-V', 'RISC-V'),
        ]

        for pattern, fixed_name in arch_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return fixed_name if fixed_name else match.group(0)

        return None

    def _extract_frequency(self, text: str) -> Optional[float]:
        """Extract core frequency in MHz."""
        # Pattern: "@ 200MHz" or "200 MHz" near CPU/core mentions
        freq_pattern = re.compile(r'@?\s*(\d+\.?\d*)\s*MHz', re.IGNORECASE)

        # Look for frequency near "CPU", "core", or "operating"
        for line in text.split('\n'):
            if any(keyword in line.lower() for keyword in ['cpu', 'core', 'operating frequency', 'clock']):
                match = freq_pattern.search(line)
                if match:
                    return float(match.group(1))

        return None

    def _extract_memory(self, text: str) -> Tuple[List[int], List[int]]:
        """Extract flash and RAM sizes in KB."""
        flash_sizes = set()
        ram_sizes = set()

        # Patterns for memory - separate KB and MB patterns for clarity
        flash_kb_pattern = re.compile(r'(\d+)\s*KB.*?[Ff]lash', re.IGNORECASE)
        flash_mb_pattern = re.compile(r'(\d+)\s*MB.*?[Ff]lash', re.IGNORECASE)
        ram_pattern = re.compile(r'(\d+)\s*KB.*?([SR]RAM|memory)', re.IGNORECASE)

        # Extract KB flash values
        for match in flash_kb_pattern.finditer(text):
            size = int(match.group(1))
            # Sanity check: flash is typically 16KB to 2048KB (2MB)
            if 8 <= size <= 2048:
                flash_sizes.add(size)

        # Extract MB flash values and convert to KB
        for match in flash_mb_pattern.finditer(text):
            size_mb = int(match.group(1))
            size_kb = size_mb * 1024
            # Sanity check: MB values should be reasonable (e.g., 1MB = 1024KB)
            if 1 <= size_mb <= 4:
                flash_sizes.add(size_kb)

        # Extract RAM values
        for match in ram_pattern.finditer(text):
            size = int(match.group(1))
            # Sanity check: RAM is typically 1KB to 512KB
            if 1 <= size <= 1024:
                ram_sizes.add(size)

        return sorted(list(flash_sizes)), sorted(list(ram_sizes))

    def _extract_temp_range(self, text: str) -> Tuple[Optional[float], Optional[float]]:
        """Extract operating temperature range."""
        # Pattern: "-40°C to 125°C" or "-55°C to 150°C"
        temp_pattern = re.compile(r'(-?\d+)°C\s+to\s+(-?\d+)°C', re.IGNORECASE)

        matches = temp_pattern.findall(text)
        if matches:
            # Find the most conservative range (widest)
            temps = [(float(t1), float(t2)) for t1, t2 in matches]
            min_temp = min(t[0] for t in temps)
            max_temp = max(t[1] for t in temps)
            return min_temp, max_temp

        return None, None

    def _extract_voltage_range(self, text: str) -> Tuple[Optional[float], Optional[float]]:
        """Extract operating voltage range."""
        # Pattern: "1.2V" or "3.3V" or "1.62V to 3.6V"
        voltage_pattern = re.compile(r'(\d+\.?\d*)\s*V\s+(?:to|-)\s+(\d+\.?\d*)\s*V', re.IGNORECASE)

        matches = voltage_pattern.findall(text[:2000])  # Search in first part
        if matches:
            voltages = [(float(v1), float(v2)) for v1, v2 in matches]
            min_v = min(v[0] for v in voltages)
            max_v = max(v[1] for v in voltages)
            return min_v, max_v

        # Single voltage mentions
        single_v_pattern = re.compile(r'(\d+\.?\d*)\s*V(?:\s+(?:I/O|core|supply))?', re.IGNORECASE)
        single_matches = single_v_pattern.findall(text[:2000])
        if single_matches:
            voltages = [float(v) for v in single_matches if 0.5 < float(v) < 30]
            if voltages:
                return min(voltages), max(voltages)

        return None, None

    def _extract_peripherals(self, text: str) -> List[str]:
        """Extract peripheral types."""
        peripherals = set()

        peripheral_keywords = [
            'ADC', 'DAC', 'PWM', 'UART', 'USART', 'I2C', 'SPI', 'CAN',
            'USB', 'Ethernet', 'GPIO', 'Timer', 'RTC', 'DMA', 'Comparator',
            'I2S', 'QSPI', 'LIN', 'FlexCAN'
        ]

        text_upper = text[:3000].upper()  # Search in first pages
        for peripheral in peripheral_keywords:
            if peripheral.upper() in text_upper:
                peripherals.add(peripheral)

        return sorted(list(peripherals))

    def _extract_applications(self, text: str) -> List[str]:
        """Extract target applications."""
        applications = []

        # Find "Applications" section
        app_match = re.search(r'Applications?:?\s*([\s\S]{0,500})', text, re.IGNORECASE)
        if app_match:
            app_text = app_match.group(1)
            # Extract bullet points or lines
            lines = [line.strip() for line in app_text.split('\n') if line.strip()]
            applications = [line.lstrip('•-▪ ') for line in lines[:10] if len(line) < 100]

        return applications

    def _extract_key_features(self, text: str) -> List[str]:
        """Extract key feature tags."""
        features = set()

        feature_keywords = {
            'Low Power': ['low power', 'low-power', 'ultra-low power'],
            'Radiation-Hardened': ['radiation', 'space-grade', 'SEL immune'],
            'AI': ['AI', 'NPU', 'neural', 'edge ai'],
            'Security': ['AES', 'encryption', 'secure', 'cryptographic'],
            'Dual-Core': ['dual-core', 'dual core', 'multi-core'],
            'High-Performance': ['high-performance', 'high performance'],
            'Automotive': ['automotive', 'AEC-Q100'],
            'Industrial': ['industrial'],
        }

        text_lower = text[:3000].lower()
        for feature, keywords in feature_keywords.items():
            if any(kw in text_lower for kw in keywords):
                features.add(feature)

        return sorted(list(features))

    def _extract_adc_bits(self, text: str) -> List[int]:
        """Extract ADC resolution in bits."""
        adc_bits = set()

        # Pattern: "12-bit ADC" or "16 bit ADC"
        pattern = re.compile(r'(\d+)[-\s]bit\s+ADC', re.IGNORECASE)
        matches = pattern.findall(text[:3000])

        for match in matches:
            bits = int(match)
            if 8 <= bits <= 32:  # Reasonable range
                adc_bits.add(bits)

        return sorted(list(adc_bits))

    def _extract_sections(self) -> List[DatasheetSection]:
        """Extract major sections from the datasheet."""
        sections = []
        current_section = None

        # Debug: Track font sizes and potential headers
        font_sizes_found = set()
        potential_headers = []

        # First, try to extract Table of Contents
        toc = self.doc.get_toc(simple=False)
        if toc:
            print(f"  📑 TOC found with {len(toc)} entries")

            for entry in toc:
                if len(entry) < 3:
                    continue
                level = entry[0]
                title = entry[1]
                page_num = entry[2]

                # Check if matches our section patterns
                for section_type, pattern in self.SECTION_PATTERNS.items():
                    if pattern.match(title):
                        # Close previous section
                        if current_section:
                            current_section.page_end = page_num - 2  # Page before new section
                            sections.append(current_section)

                        # Start new section
                        current_section = DatasheetSection(
                            title=title,
                            level=level,
                            page_start=page_num - 1,  # TOC uses 1-indexed pages
                            page_end=None,
                            content="",
                            section_number=self._extract_section_number(title)
                        )
                        break

            # Close last section
            if current_section:
                current_section.page_end = len(self.doc) - 1
                sections.append(current_section)

            # Extract content for each section
            for section in sections:
                section.content = self._extract_section_content(section.page_start, section.page_end or len(self.doc))

            if sections:
                print(f"  ✅ Extracted {len(sections)} sections from TOC")
                return sections
            else:
                print(f"  ⚠️  TOC found but no sections matched patterns")

        # Fallback: Parse text for section headers
        print(f"  🔍 No TOC found, analyzing text for section headers...")

        # First pass: find potential headers in first 20 pages
        for page_num in range(min(20, len(self.doc))):
            page = self.doc[page_num]
            blocks = page.get_text("dict")["blocks"]

            for block in blocks:
                if "lines" not in block:
                    continue

                for line in block["lines"]:
                    for span in line["spans"]:
                        text = span["text"].strip()
                        size = span["size"]
                        font_sizes_found.add(round(size, 1))

                        # Look for potential headers (any size > 10, length 3-100 chars)
                        if size > 10 and 3 < len(text) < 100:
                            # Check if matches known section pattern
                            for section_type, pattern in self.SECTION_PATTERNS.items():
                                if pattern.match(text):
                                    potential_headers.append({
                                        'text': text,
                                        'size': round(size, 1),
                                        'page': page_num,
                                        'type': section_type
                                    })
                                    break

        # Now actually extract sections using relaxed criteria
        if potential_headers:
            # Find the minimum font size used for headers
            header_sizes = [h['size'] for h in potential_headers]
            min_header_size = min(header_sizes) - 0.5  # Be slightly more permissive
            print(f"  🎯 Found {len(potential_headers)} section headers, using font threshold: {min_header_size}")

            # Re-scan all pages with this threshold
            for page_num in range(len(self.doc)):
                page = self.doc[page_num]
                blocks = page.get_text("dict")["blocks"]

                for block in blocks:
                    if "lines" not in block:
                        continue

                    for line in block["lines"]:
                        for span in line["spans"]:
                            text = span["text"].strip()
                            size = span["size"]

                            # Detect section headers with relaxed threshold
                            if size >= min_header_size and len(text) > 3:
                                # Check if matches known section pattern
                                for section_type, pattern in self.SECTION_PATTERNS.items():
                                    if pattern.match(text):
                                        # Close previous section
                                        if current_section:
                                            current_section.page_end = page_num - 1
                                            sections.append(current_section)

                                        # Start new section
                                        current_section = DatasheetSection(
                                            title=text,
                                            level=1,
                                            page_start=page_num,
                                            page_end=None,
                                            content="",
                                            section_number=self._extract_section_number(text)
                                        )
                                        break

            # Close last section
            if current_section:
                current_section.page_end = len(self.doc) - 1
                sections.append(current_section)

            # Extract content for each section
            for section in sections:
                section.content = self._extract_section_content(section.page_start, section.page_end or len(self.doc))

        if sections:
            print(f"  ✅ Extracted {len(sections)} sections from text analysis")

        return sections

    def _extract_section_number(self, text: str) -> Optional[str]:
        """Extract section number like '1.2.3'."""
        match = re.match(r'^((?:\d+\.)+\d*)', text)
        return match.group(1) if match else None

    def _extract_section_content(self, start_page: int, end_page: int) -> str:
        """Extract text content from a page range."""
        content = []
        for page_num in range(start_page, min(end_page + 1, len(self.doc))):
            page = self.doc[page_num]
            content.append(page.get_text())
        return "\n".join(content)

    def _create_chunks(self) -> List[DatasheetChunk]:
        """Create semantic chunks for vector storage."""
        chunks = []
        chunk_id_counter = 0

        # 1. Overview chunk (first 3 pages)
        overview_text = self._extract_section_content(0, 2)
        chunks.append(DatasheetChunk(
            chunk_id=f"{self.metadata.document_id}_overview_{chunk_id_counter}",
            chunk_type="overview",
            content=overview_text,
            metadata=self.metadata.to_dict(),
            page_numbers=list(range(3))
        ))
        chunk_id_counter += 1

        # 2. Section-based chunks (if sections were found)
        if self.sections:
            for section in self.sections:
                # Determine chunk type
                chunk_type = "general"
                if any(kw in section.title.lower() for kw in ['feature', 'application']):
                    chunk_type = "features"
                elif any(kw in section.title.lower() for kw in ['specification', 'electrical', 'characteristic']):
                    chunk_type = "specs"
                elif any(kw in section.title.lower() for kw in ['pin', 'terminal']):
                    chunk_type = "pins"
                elif 'description' in section.title.lower():
                    chunk_type = "description"

                # Split long sections into smaller chunks (max 2000 tokens ≈ 8000 chars)
                content_parts = self._split_content(section.content, max_chars=8000)

                for i, part in enumerate(content_parts):
                    chunks.append(DatasheetChunk(
                        chunk_id=f"{self.metadata.document_id}_{chunk_type}_{chunk_id_counter}",
                        chunk_type=chunk_type,
                        content=f"Section: {section.title}\n\n{part}",
                        metadata={**self.metadata.to_dict(), "section": section.title},
                        page_numbers=list(range(section.page_start, (section.page_end or section.page_start) + 1))
                    ))
                    chunk_id_counter += 1
        else:
            # Fallback: If no sections found, chunk by page ranges
            print(f"  ⚠️  No sections detected, using page-based chunking")
            pages_per_chunk = 3  # Small chunks to fit in OpenAI's 8192 token limit
            total_pages = len(self.doc)

            for start_page in range(3, total_pages, pages_per_chunk):  # Start after overview (page 3)
                end_page = min(start_page + pages_per_chunk - 1, total_pages - 1)
                content = self._extract_section_content(start_page, end_page)

                chunks.append(DatasheetChunk(
                    chunk_id=f"{self.metadata.document_id}_page_{chunk_id_counter}",
                    chunk_type="general",
                    content=f"Pages {start_page+1}-{end_page+1}:\n\n{content}",
                    metadata=self.metadata.to_dict(),
                    page_numbers=list(range(start_page, end_page + 1))
                ))
                chunk_id_counter += 1

        return chunks

    def _split_content(self, content: str, max_chars: int = 8000) -> List[str]:
        """Split content into chunks of max_chars, trying to break at paragraphs."""
        if len(content) <= max_chars:
            return [content]

        parts = []
        current = []
        current_len = 0

        paragraphs = content.split('\n\n')

        for para in paragraphs:
            para_len = len(para)
            if current_len + para_len > max_chars and current:
                parts.append('\n\n'.join(current))
                current = [para]
                current_len = para_len
            else:
                current.append(para)
                current_len += para_len

        if current:
            parts.append('\n\n'.join(current))

        return parts

    def close(self):
        """Close the PDF document."""
        self.doc.close()
