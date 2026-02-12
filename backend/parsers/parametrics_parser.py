"""
Parser for TI parametrics CSV files.

This parser extracts authoritative product specifications from TI's
parametric export files, providing more reliable metadata than PDF parsing.
"""

import pandas as pd
import re
from typing import Dict, List, Optional, Tuple
from pathlib import Path


class ParametricsParser:
    """Parse TI parametrics CSV files to extract product specifications."""

    def __init__(self, csv_path: str):
        """
        Initialize the parser with a CSV file.

        Args:
            csv_path: Path to the TI parametrics CSV export
        """
        self.csv_path = csv_path
        self.df = None
        self._load_csv()

    def _load_csv(self):
        """Load and validate the CSV file."""
        try:
            # Read CSV with mixed type handling
            self.df = pd.read_csv(self.csv_path, low_memory=False)

            # Validate required columns
            required_cols = ['Product or Part number']
            missing_cols = [col for col in required_cols if col not in self.df.columns]

            if missing_cols:
                raise ValueError(f"CSV missing required columns: {missing_cols}")

            print(f"Loaded parametrics CSV: {len(self.df)} products")

        except Exception as e:
            raise RuntimeError(f"Failed to load parametrics CSV: {e}")

    def get_part_data(self, part_number: str) -> Optional[Dict]:
        """
        Get parametric data for a specific part number.

        Args:
            part_number: The part number to look up (e.g., "MSPM0G5187")

        Returns:
            Dictionary with parametric data, or None if not found
        """
        if self.df is None:
            return None

        # Normalize part number for matching
        part_clean = part_number.strip().upper()

        # Try exact match first
        matches = self.df[self.df['Product or Part number'].str.upper() == part_clean]

        # If no exact match, try contains (for variants)
        if matches.empty:
            matches = self.df[self.df['Product or Part number'].str.upper().str.contains(part_clean, na=False)]

        if matches.empty:
            return None

        # Use first match
        row = matches.iloc[0]

        return self._extract_metadata(row)

    def _extract_metadata(self, row: pd.Series) -> Dict:
        """
        Extract and normalize metadata from a CSV row.

        Args:
            row: A pandas Series representing a product row

        Returns:
            Dictionary with normalized metadata
        """
        metadata = {
            'part_number': self._safe_get(row, 'Product or Part number'),
            'description': self._safe_get(row, 'Description'),
        }

        # Datasheet links (priority: PDF > HTML > Product Page)
        metadata['pdf_url'] = self._safe_get(row, 'PDF datasheet')
        metadata['html_url'] = self._safe_get(row, 'HTML datasheet')
        metadata['product_url'] = self._safe_get(row, 'Product Page')

        # Get best available link
        metadata['datasheet_link'] = (
            metadata['pdf_url'] or
            metadata['html_url'] or
            metadata['product_url']
        )

        # Core specs
        metadata['core_freq_mhz'] = self._extract_frequency(row)
        metadata['flash_kb'] = self._extract_flash(row)
        metadata['ram_kb'] = self._extract_ram(row)

        # CPU and architecture
        metadata['cpu'] = self._safe_get(row, 'CPU')
        metadata['architecture'] = self._extract_architecture(row)

        # Package information
        metadata['package_type'] = self._safe_get(row, 'Package type')
        metadata['pin_count'] = self._extract_pin_count(row)
        metadata['package_size'] = self._safe_get(row, 'Package size (L x W) (mm)')

        # Operating conditions
        metadata['temp_range'] = self._extract_temp_range(row)
        metadata['rating'] = self._safe_get(row, 'Rating')
        metadata['status'] = self._safe_get(row, 'Status')

        # Peripherals
        metadata['adc_type'] = self._safe_get(row, 'ADC type')
        metadata['uart_count'] = self._extract_number(row, 'UART')
        metadata['i2c_count'] = self._extract_number(row, 'Number of I2Cs')
        metadata['spi_count'] = self._extract_number(row, 'SPI')
        metadata['gpio_count'] = self._safe_get(row, 'Number of GPIOs')

        # Security and features
        metadata['security'] = self._safe_get(row, 'Security')
        metadata['features'] = self._safe_get(row, 'Features')

        # Price
        metadata['price_usd'] = self._extract_price(row)

        return metadata

    def _safe_get(self, row: pd.Series, column: str) -> Optional[str]:
        """Safely get a column value, handling missing columns and NaN."""
        if column not in row.index:
            return None

        value = row[column]

        # Handle NaN, None, empty strings
        if pd.isna(value) or value == '' or value == 'NaN':
            return None

        return str(value).strip()

    def _extract_frequency(self, row: pd.Series) -> Optional[List[int]]:
        """Extract core frequency in MHz."""
        freq_str = self._safe_get(row, 'Frequency (MHz)')

        if not freq_str:
            return None

        # Parse frequency values (could be single value or range)
        freqs = []
        for match in re.finditer(r'(\d+(?:\.\d+)?)', freq_str):
            freq = float(match.group(1))
            if freq > 0 and freq <= 1000:  # Sanity check
                freqs.append(int(freq))

        return freqs if freqs else None

    def _extract_flash(self, row: pd.Series) -> Optional[List[int]]:
        """Extract flash memory in KB."""
        flash_str = self._safe_get(row, 'Nonvolatile memory (kByte)')

        if not flash_str:
            return None

        flash_sizes = []
        for match in re.finditer(r'(\d+)', flash_str):
            size = int(match.group(1))
            if 8 <= size <= 8192:  # Sanity check: 8KB to 8MB
                flash_sizes.append(size)

        return flash_sizes if flash_sizes else None

    def _extract_ram(self, row: pd.Series) -> Optional[List[int]]:
        """Extract RAM in KB."""
        ram_str = self._safe_get(row, 'RAM (kByte)')

        if not ram_str:
            return None

        ram_sizes = []
        for match in re.finditer(r'(\d+)', ram_str):
            size = int(match.group(1))
            if 1 <= size <= 2048:  # Sanity check: 1KB to 2MB
                ram_sizes.append(size)

        return ram_sizes if ram_sizes else None

    def _extract_architecture(self, row: pd.Series) -> Optional[str]:
        """Extract CPU architecture from CPU field or description."""
        cpu = self._safe_get(row, 'CPU')

        if cpu:
            # Normalize architecture names
            if 'cortex-m0+' in cpu.lower() or 'cortex m0+' in cpu.lower():
                return 'Arm Cortex-M0+'
            elif 'cortex-m4' in cpu.lower():
                return 'Arm Cortex-M4'
            elif 'cortex-m33' in cpu.lower():
                return 'Arm Cortex-M33'
            elif 'cortex-a' in cpu.lower():
                return 'Arm Cortex-A'

            return cpu

        # Fallback to description
        desc = self._safe_get(row, 'Description')
        if desc:
            if 'cortex-m0+' in desc.lower():
                return 'Arm Cortex-M0+'
            elif 'cortex-m4' in desc.lower():
                return 'Arm Cortex-M4'

        return None

    def _extract_pin_count(self, row: pd.Series) -> Optional[List[int]]:
        """Extract pin counts."""
        pin_str = self._safe_get(row, 'Pin count')

        if not pin_str:
            return None

        pins = []
        for match in re.finditer(r'(\d+)', pin_str):
            count = int(match.group(1))
            if 4 <= count <= 500:  # Sanity check
                pins.append(count)

        return pins if pins else None

    def _extract_temp_range(self, row: pd.Series) -> Optional[Tuple[int, int]]:
        """Extract temperature range as (min, max) tuple."""
        temp_str = self._safe_get(row, 'Operating temperature range (°C)')

        if not temp_str:
            return None

        # Match patterns like "-40 to 125" or "-40 to 85"
        match = re.search(r'(-?\d+)\s*to\s*(-?\d+)', temp_str)
        if match:
            min_temp = int(match.group(1))
            max_temp = int(match.group(2))
            return (min_temp, max_temp)

        return None

    def _extract_price(self, row: pd.Series) -> Optional[float]:
        """Extract price in USD."""
        price_str = self._safe_get(row, 'Price|Quantity (USD)')

        if not price_str:
            return None

        # Extract first numeric value
        match = re.search(r'(\d+(?:\.\d+)?)', price_str)
        if match:
            return float(match.group(1))

        return None

    def _extract_number(self, row: pd.Series, column: str) -> Optional[int]:
        """Extract a simple numeric value from a column."""
        value_str = self._safe_get(row, column)

        if not value_str:
            return None

        match = re.search(r'(\d+)', value_str)
        if match:
            return int(match.group(1))

        return None

    def get_all_mspm0_parts(self) -> List[Dict]:
        """
        Get all MSPM0 parts from the parametrics data.

        Returns:
            List of dictionaries with part data
        """
        if self.df is None:
            return []

        # Filter for MSPM0 parts
        mspm0_df = self.df[self.df['Product or Part number'].str.contains('MSPM0', case=False, na=False)]

        parts = []
        for _, row in mspm0_df.iterrows():
            parts.append(self._extract_metadata(row))

        return parts

    def get_stats(self) -> Dict:
        """Get statistics about the parametrics data."""
        if self.df is None:
            return {}

        mspm0_count = len(self.df[self.df['Product or Part number'].str.contains('MSPM0', case=False, na=False)])

        return {
            'total_products': len(self.df),
            'mspm0_products': mspm0_count,
            'columns': len(self.df.columns),
            'column_names': list(self.df.columns)
        }
