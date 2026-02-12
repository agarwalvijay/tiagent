"""Extended tools for price comparison, advanced filtering, and pin compatibility."""
from typing import List, Dict, Any, Optional, Tuple
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
from backend.agent.tools import SearchTools


def compare_prices_tool(part_numbers: List[str], quantity: int = 1000) -> str:
    """
    Compare prices for multiple parts at a given quantity.

    Args:
        part_numbers: List of 2-5 part numbers to compare
        quantity: Order quantity (default: 1000 units)

    Returns:
        Formatted price comparison table
    """
    if len(part_numbers) < 2:
        return "Please provide at least 2 part numbers to compare prices."

    if len(part_numbers) > 5:
        return "Maximum 5 parts can be compared at once."

    tools = SearchTools()
    prices = {}
    specs = {}

    for part_num in part_numbers:
        chunks = tools.get_by_part_number(part_num)

        if not chunks:
            prices[part_num] = {"error": "Part not found"}
            continue

        # Get metadata from first chunk (overview)
        meta = chunks[0]['metadata']

        # Extract price (price_usd is typically for 1K quantity)
        price_str = meta.get('price_usd', '')
        try:
            # Parse price - might be a string like "0.97" or empty
            if price_str and price_str != '':
                unit_price = float(price_str)
            else:
                unit_price = None
        except (ValueError, TypeError):
            unit_price = None

        # Get key specs for context
        flash_min = meta.get('flash_kb_min', 0)
        flash_max = meta.get('flash_kb_max', 0)
        flash = flash_max if flash_max > 0 else flash_min

        ram_min = meta.get('ram_kb_min', 0)
        ram_max = meta.get('ram_kb_max', 0)
        ram = ram_max if ram_max > 0 else ram_min

        freq = meta.get('core_freq_mhz', 0)
        arch = meta.get('architecture', 'N/A')

        # Get datasheet link
        datasheet_link = (
            meta.get('pdf_datasheet_url') or
            meta.get('html_datasheet_url') or
            meta.get('product_page_url') or
            meta.get('datasheet_link', '')
        )

        prices[part_num] = {
            "unit_price": unit_price,
            "total_price": unit_price * quantity if unit_price else None,
            "flash_kb": flash,
            "ram_kb": ram,
            "freq_mhz": freq,
            "architecture": arch,
            "datasheet_link": datasheet_link
        }

    # Build output
    output = [f"## Price Comparison ({quantity:,} units)\n\n"]

    # Check if we have any valid prices
    valid_prices = [p for p in prices.values() if not isinstance(p.get('unit_price'), type(None)) and 'error' not in p]

    if not valid_prices:
        output.append("⚠️ No pricing information available for these parts.\n")
        output.append("\nNote: Pricing data may not be available for all parts in the database.\n")
        return "".join(output)

    # Create markdown table
    header = "| Part Number | Unit Price | Total Cost | Flash | RAM | Freq | Datasheet |\n"
    separator = "|-------------|------------|------------|-------|-----|------|-----------|"

    output.append(header)
    output.append(separator + "\n")

    # Sort by price (cheapest first)
    sorted_parts = sorted(
        [(pn, data) for pn, data in prices.items() if 'error' not in data],
        key=lambda x: x[1].get('unit_price', float('inf'))
    )

    for part_num, data in sorted_parts:
        unit_price = data.get('unit_price')
        total_price = data.get('total_price')

        if unit_price is None:
            price_display = "N/A"
            total_display = "N/A"
        else:
            price_display = f"${unit_price:.2f}"
            total_display = f"${total_price:,.2f}"

        flash_display = f"{int(data['flash_kb'])} KB" if data['flash_kb'] > 0 else "N/A"
        ram_display = f"{int(data['ram_kb'])} KB" if data['ram_kb'] > 0 else "N/A"
        freq_display = f"{int(data['freq_mhz'])} MHz" if data['freq_mhz'] > 0 else "N/A"

        link = data.get('datasheet_link', '')
        link_display = f"[View]({link})" if link else "—"

        row = f"| **{part_num}** | {price_display} | {total_display} | {flash_display} | {ram_display} | {freq_display} | {link_display} |\n"
        output.append(row)

    # Add cost analysis
    if len(valid_prices) >= 2:
        cheapest = sorted_parts[0]
        most_expensive = sorted_parts[-1]

        if cheapest[1]['unit_price'] and most_expensive[1]['unit_price']:
            savings = most_expensive[1]['total_price'] - cheapest[1]['total_price']
            savings_pct = (savings / most_expensive[1]['total_price']) * 100

            output.append(f"\n**Cost Analysis:**\n")
            output.append(f"- Cheapest: **{cheapest[0]}** at ${cheapest[1]['unit_price']:.2f}/unit\n")
            output.append(f"- Most expensive: **{most_expensive[0]}** at ${most_expensive[1]['unit_price']:.2f}/unit\n")
            output.append(f"- Savings: **${savings:,.2f}** ({savings_pct:.1f}%) by choosing {cheapest[0]}\n")

    # Add errors if any
    errors = [f"{pn}: {data['error']}" for pn, data in prices.items() if 'error' in data]
    if errors:
        output.append(f"\n**Errors:**\n")
        for error in errors:
            output.append(f"- {error}\n")

    return "".join(output)


def find_parts_by_specs_tool(
    min_flash_kb: Optional[int] = None,
    min_ram_kb: Optional[int] = None,
    min_freq_mhz: Optional[int] = None,
    max_price: Optional[float] = None,
    required_peripherals: Optional[List[str]] = None,
    package_type: Optional[str] = None,
    temp_min: Optional[int] = None,
    temp_max: Optional[int] = None,
    architecture: Optional[str] = None,
    max_results: int = 10
) -> str:
    """
    Find parts matching multiple specification criteria.

    Args:
        min_flash_kb: Minimum flash memory in KB
        min_ram_kb: Minimum RAM in KB
        min_freq_mhz: Minimum CPU frequency in MHz
        max_price: Maximum unit price in USD
        required_peripherals: List of required peripherals (e.g., ["USB", "CAN-FD"])
        package_type: Package type (e.g., "LQFP", "VQFN")
        temp_min: Minimum operating temperature in °C
        temp_max: Maximum operating temperature in °C
        architecture: CPU architecture (e.g., "Arm Cortex-M0+")
        max_results: Maximum number of results (default: 10)

    Returns:
        Formatted list of matching parts
    """
    tools = SearchTools()

    # Get all parts (we'll filter manually since ChromaDB filtering is limited)
    all_results = tools.collection.get(
        limit=10000,
        include=["metadatas", "documents"]
    )

    matching_parts = {}

    for i, metadata in enumerate(all_results["metadatas"]):
        part_nums = metadata.get("part_numbers", "")

        if not part_nums or part_nums in matching_parts:
            continue

        # Apply filters
        matches = True
        reasons = []

        # Flash filter
        if min_flash_kb is not None:
            flash_max = metadata.get('flash_kb_max', 0)
            if flash_max < min_flash_kb:
                matches = False
                continue
            reasons.append(f"{flash_max}KB flash")

        # RAM filter
        if min_ram_kb is not None:
            ram_max = metadata.get('ram_kb_max', 0)
            if ram_max < min_ram_kb:
                matches = False
                continue
            reasons.append(f"{ram_max}KB RAM")

        # Frequency filter
        if min_freq_mhz is not None:
            freq = metadata.get('core_freq_mhz', 0)
            if freq < min_freq_mhz:
                matches = False
                continue
            reasons.append(f"{freq}MHz")

        # Price filter
        if max_price is not None:
            price_str = metadata.get('price_usd', '')
            try:
                price = float(price_str) if price_str else float('inf')
                if price > max_price:
                    matches = False
                    continue
                reasons.append(f"${price:.2f}")
            except (ValueError, TypeError):
                pass

        # Package type filter
        if package_type is not None:
            packages = metadata.get('package_types', '').upper()
            # Extract base package type (e.g., "VQFN" from "VQFN-48")
            search_pkg = package_type.upper().split('-')[0]
            # Check if base package type is in the comma-separated list
            package_list = [p.strip() for p in packages.split(',')]
            if not any(search_pkg in pkg for pkg in package_list):
                matches = False
                continue
            reasons.append(f"{package_type}")

        # Temperature range filter
        if temp_min is not None or temp_max is not None:
            op_temp_min = metadata.get('operating_temp_min_c', -999)
            op_temp_max = metadata.get('operating_temp_max_c', 999)

            if temp_min is not None and op_temp_min > temp_min:
                matches = False
                continue
            if temp_max is not None and op_temp_max < temp_max:
                matches = False
                continue

            if temp_min is not None or temp_max is not None:
                reasons.append(f"{op_temp_min} to {op_temp_max}°C")

        # Architecture filter
        if architecture is not None:
            arch = metadata.get('architecture', '')
            if architecture.lower() not in arch.lower():
                matches = False
                continue
            reasons.append(f"{arch}")

        # Peripherals filter (check if all required peripherals are present)
        if required_peripherals:
            peripherals_str = metadata.get('peripherals', '').lower()
            key_features_str = metadata.get('key_features', '').lower()
            combined = peripherals_str + " " + key_features_str

            for peripheral in required_peripherals:
                if peripheral.lower() not in combined:
                    matches = False
                    break

            if not matches:
                continue

            reasons.append(f"Has: {', '.join(required_peripherals)}")

        if matches:
            matching_parts[part_nums] = {
                'metadata': metadata,
                'match_reasons': reasons,
                'document_id': all_results['ids'][i]
            }

        # Stop if we have enough
        if len(matching_parts) >= max_results * 2:  # Get extra for sorting
            break

    if not matching_parts:
        filters_desc = []
        if min_flash_kb: filters_desc.append(f"≥{min_flash_kb}KB flash")
        if min_ram_kb: filters_desc.append(f"≥{min_ram_kb}KB RAM")
        if min_freq_mhz: filters_desc.append(f"≥{min_freq_mhz}MHz")
        if max_price: filters_desc.append(f"≤${max_price}")
        if required_peripherals: filters_desc.append(f"with {', '.join(required_peripherals)}")
        if package_type: filters_desc.append(f"{package_type} package")
        if architecture: filters_desc.append(f"{architecture}")

        return f"No parts found matching: {', '.join(filters_desc)}"

    # Sort by price (if available), then by flash size
    sorted_parts = sorted(
        matching_parts.items(),
        key=lambda x: (
            float(x[1]['metadata'].get('price_usd', '999')) if x[1]['metadata'].get('price_usd') else 999,
            -x[1]['metadata'].get('flash_kb_max', 0)
        )
    )[:max_results]

    # Format output
    output = [f"## Parts Matching Specifications\n\n"]

    # Show filter criteria
    criteria = []
    if min_flash_kb: criteria.append(f"Flash ≥ {min_flash_kb} KB")
    if min_ram_kb: criteria.append(f"RAM ≥ {min_ram_kb} KB")
    if min_freq_mhz: criteria.append(f"Frequency ≥ {min_freq_mhz} MHz")
    if max_price: criteria.append(f"Price ≤ ${max_price}")
    if required_peripherals: criteria.append(f"Peripherals: {', '.join(required_peripherals)}")
    if package_type: criteria.append(f"Package: {package_type}")
    if temp_min or temp_max:
        temp_str = f"Temp: {temp_min if temp_min else '?'} to {temp_max if temp_max else '?'}°C"
        criteria.append(temp_str)
    if architecture: criteria.append(f"Architecture: {architecture}")

    if criteria:
        output.append(f"**Search Criteria:** {', '.join(criteria)}\n\n")

    output.append(f"**Found {len(matching_parts)} matching parts** (showing top {len(sorted_parts)}):\n\n")

    for i, (part_num, data) in enumerate(sorted_parts, 1):
        meta = data['metadata']

        # Get specs
        flash = meta.get('flash_kb_max', 0)
        ram = meta.get('ram_kb_max', 0)
        freq = meta.get('core_freq_mhz', 0)
        arch = meta.get('architecture', 'N/A')
        price_str = meta.get('price_usd', '')
        price = f"${float(price_str):.2f}" if price_str else "N/A"

        # Get datasheet link
        datasheet_link = (
            meta.get('pdf_datasheet_url') or
            meta.get('html_datasheet_url') or
            meta.get('product_page_url') or
            meta.get('datasheet_link', '')
        )

        link_text = f" - 📄 [Datasheet]({datasheet_link})" if datasheet_link else ""

        output.append(
            f"{i}. **{part_num}**{link_text}\n"
            f"   - Flash: {flash} KB, RAM: {ram} KB, Freq: {freq} MHz\n"
            f"   - Architecture: {arch}\n"
            f"   - Price: {price}\n"
            f"   - Match: {', '.join(data['match_reasons'])}\n\n"
        )

    return "".join(output)


def find_pin_compatible_tool(part_number: str, allow_better_specs: bool = True) -> str:
    """
    Find pin-compatible alternatives (drop-in replacements).

    Args:
        part_number: Reference part number
        allow_better_specs: If True, show parts with equal or better specs (default: True)

    Returns:
        List of pin-compatible alternatives
    """
    tools = SearchTools()

    # Get the reference part
    ref_chunks = tools.get_by_part_number(part_number)

    if not ref_chunks:
        return f"Part {part_number} not found in database."

    ref_meta = ref_chunks[0]['metadata']

    # Extract reference specs
    ref_package = ref_meta.get('package_types', '').split(',')[0].strip()  # Primary package
    ref_pins_str = ref_meta.get('pin_count', '')

    # Parse pin count (might be like "48,64" - get first)
    try:
        if ',' in str(ref_pins_str):
            ref_pins = int(str(ref_pins_str).split(',')[0].strip())
        else:
            ref_pins = int(ref_pins_str) if ref_pins_str else 0
    except (ValueError, TypeError):
        ref_pins = 0

    ref_voltage_min = ref_meta.get('voltage_min_v', 0)
    ref_voltage_max = ref_meta.get('voltage_max_v', 0)
    ref_flash = ref_meta.get('flash_kb_max', 0)
    ref_ram = ref_meta.get('ram_kb_max', 0)
    ref_freq = ref_meta.get('core_freq_mhz', 0)

    if not ref_package or ref_pins == 0:
        return f"Insufficient package information for {part_number} to find pin-compatible alternatives."

    # Search for alternatives
    all_results = tools.collection.get(
        limit=10000,
        include=["metadatas"]
    )

    alternatives = []

    for metadata in all_results["metadatas"]:
        alt_part = metadata.get("part_numbers", "")

        if not alt_part or alt_part == ref_meta.get("part_numbers"):
            continue

        # Check package type
        alt_packages = metadata.get('package_types', '').upper()
        if ref_package.upper() not in alt_packages:
            continue

        # Check pin count
        alt_pins_str = metadata.get('pin_count', '')
        try:
            if ',' in str(alt_pins_str):
                # Check if reference pin count is in the list
                alt_pins_list = [int(p.strip()) for p in str(alt_pins_str).split(',')]
                if ref_pins not in alt_pins_list:
                    continue
            else:
                alt_pins = int(alt_pins_str) if alt_pins_str else 0
                if alt_pins != ref_pins:
                    continue
        except (ValueError, TypeError):
            continue

        # Check voltage compatibility (should overlap)
        alt_voltage_min = metadata.get('voltage_min_v', 0)
        alt_voltage_max = metadata.get('voltage_max_v', 0)

        voltage_compatible = True
        if ref_voltage_min > 0 and alt_voltage_max > 0:
            # Check if ranges overlap
            if alt_voltage_max < ref_voltage_min or alt_voltage_min > ref_voltage_max:
                voltage_compatible = False

        if not voltage_compatible:
            continue

        # Get alternative specs
        alt_flash = metadata.get('flash_kb_max', 0)
        alt_ram = metadata.get('ram_kb_max', 0)
        alt_freq = metadata.get('core_freq_mhz', 0)

        # Apply "better specs" filter if requested
        if allow_better_specs:
            # Must have equal or better specs
            if alt_flash < ref_flash or alt_ram < ref_ram or alt_freq < ref_freq:
                continue
        else:
            # Must match exactly
            if alt_flash != ref_flash or alt_ram != ref_ram or alt_freq != ref_freq:
                continue

        # Calculate compatibility score
        score = 0
        differences = []

        if alt_flash > ref_flash:
            score += (alt_flash - ref_flash) / ref_flash * 10
            differences.append(f"+{alt_flash - ref_flash}KB flash")

        if alt_ram > ref_ram:
            score += (alt_ram - ref_ram) / ref_ram * 10
            differences.append(f"+{alt_ram - ref_ram}KB RAM")

        if alt_freq > ref_freq:
            score += (alt_freq - ref_freq) / ref_freq * 10
            differences.append(f"+{alt_freq - ref_freq}MHz")

        # Get price comparison
        ref_price_str = ref_meta.get('price_usd', '')
        alt_price_str = metadata.get('price_usd', '')

        price_diff = None
        if ref_price_str and alt_price_str:
            try:
                ref_price = float(ref_price_str)
                alt_price = float(alt_price_str)
                price_diff = alt_price - ref_price

                if price_diff < 0:
                    differences.append(f"${abs(price_diff):.2f} cheaper")
                elif price_diff > 0:
                    differences.append(f"${price_diff:.2f} more expensive")
            except (ValueError, TypeError):
                pass

        # Get datasheet link
        datasheet_link = (
            metadata.get('pdf_datasheet_url') or
            metadata.get('html_datasheet_url') or
            metadata.get('product_page_url') or
            metadata.get('datasheet_link', '')
        )

        alternatives.append({
            'part_number': alt_part,
            'flash': alt_flash,
            'ram': alt_ram,
            'freq': alt_freq,
            'price': alt_price_str,
            'differences': differences,
            'score': score,
            'datasheet_link': datasheet_link,
            'metadata': metadata
        })

    if not alternatives:
        return f"No pin-compatible alternatives found for {part_number} ({ref_package}-{ref_pins})."

    # Sort by score (lower is better - closer to original)
    sorted_alternatives = sorted(alternatives, key=lambda x: x['score'])[:10]

    # Format output
    output = [f"## Pin-Compatible Alternatives to {part_number}\n\n"]

    # Show reference specs
    ref_datasheet_link = (
        ref_meta.get('pdf_datasheet_url') or
        ref_meta.get('html_datasheet_url') or
        ref_meta.get('product_page_url') or
        ref_meta.get('datasheet_link', '')
    )

    ref_link = f" - 📄 [Datasheet]({ref_datasheet_link})" if ref_datasheet_link else ""

    output.append(f"**Reference:** {part_number}{ref_link}\n")
    output.append(f"- Package: {ref_package}-{ref_pins}\n")
    output.append(f"- Specs: {ref_flash}KB flash, {ref_ram}KB RAM, {ref_freq}MHz\n")
    output.append(f"- Voltage: {ref_voltage_min}-{ref_voltage_max}V\n\n")

    output.append(f"**Found {len(alternatives)} pin-compatible parts** (showing top {len(sorted_alternatives)}):\n\n")

    for i, alt in enumerate(sorted_alternatives, 1):
        link_text = f" - 📄 [Datasheet]({alt['datasheet_link']})" if alt['datasheet_link'] else ""

        price_display = f"${float(alt['price']):.2f}" if alt['price'] else "N/A"

        diff_text = f" ({', '.join(alt['differences'])})" if alt['differences'] else " (Identical specs)"

        output.append(
            f"{i}. **{alt['part_number']}**{link_text}\n"
            f"   - Specs: {alt['flash']}KB flash, {alt['ram']}KB RAM, {alt['freq']}MHz\n"
            f"   - Price: {price_display}\n"
            f"   - Difference: {diff_text}\n\n"
        )

    return "".join(output)


def estimate_battery_life_tool(
    part_number: str,
    battery_capacity_mah: int,
    run_time_pct: float,
    sleep_time_pct: float,
    active_freq_mhz: int = None
) -> str:
    """
    Estimate battery life for a microcontroller.

    Args:
        part_number: Part number to analyze
        battery_capacity_mah: Battery capacity in mAh (e.g., 240 for CR2032)
        run_time_pct: Percentage of time in active/run mode (0-100)
        sleep_time_pct: Percentage of time in sleep/low-power mode (0-100)
        active_freq_mhz: Active frequency in MHz (uses part's max if not specified)

    Returns:
        Battery life estimate with breakdown
    """
    if run_time_pct + sleep_time_pct > 100:
        return "Error: run_time_pct + sleep_time_pct cannot exceed 100%"

    tools = SearchTools()
    chunks = tools.get_by_part_number(part_number)

    if not chunks:
        return f"Part {part_number} not found in database."

    meta = chunks[0]['metadata']

    # Get part specs
    freq = active_freq_mhz or meta.get('core_freq_mhz', 0)
    flash = meta.get('flash_kb_max', 0)
    ram = meta.get('ram_kb_max', 0)

    # Try to extract power consumption from content
    # Look for power-related chunks
    power_info = []
    active_current_ua = None
    sleep_current_ua = None
    standby_current_ua = None

    for chunk in chunks:
        content_lower = chunk['content'].lower()
        
        # Look for active/run mode current
        if 'run' in content_lower or 'active' in content_lower:
            # Try to find current values (µA or mA)
            import re
            # Look for patterns like "34µA/MHz" or "100µA at 24MHz"
            ua_per_mhz_match = re.search(r'(\d+\.?\d*)\s*[µu]a\s*/\s*mhz', content_lower)
            if ua_per_mhz_match and not active_current_ua:
                active_current_ua = float(ua_per_mhz_match.group(1)) * freq

            # Look for absolute current values
            ua_match = re.search(r'(\d+\.?\d*)\s*[µu]a', content_lower)
            ma_match = re.search(r'(\d+\.?\d*)\s*ma', content_lower)
            
            if ma_match and not active_current_ua:
                active_current_ua = float(ma_match.group(1)) * 1000
            elif ua_match and not active_current_ua and 'mhz' in content_lower:
                # Might be a specific frequency mentioned
                freq_match = re.search(r'(\d+)\s*mhz', content_lower)
                if freq_match:
                    active_current_ua = float(ua_match.group(1))

        # Look for sleep/stop/standby mode
        if any(mode in content_lower for mode in ['sleep', 'stop', 'standby', 'low power']):
            import re
            ua_match = re.search(r'(\d+\.?\d*)\s*[µu]a', content_lower)
            na_match = re.search(r'(\d+\.?\d*)\s*na', content_lower)
            
            if 'standby' in content_lower and not standby_current_ua:
                if ua_match:
                    standby_current_ua = float(ua_match.group(1))
                elif na_match:
                    standby_current_ua = float(na_match.group(1)) / 1000
            elif ('sleep' in content_lower or 'stop' in content_lower) and not sleep_current_ua:
                if ua_match:
                    sleep_current_ua = float(ua_match.group(1))
                elif na_match:
                    sleep_current_ua = float(na_match.group(1)) / 1000

    # Use best available sleep current (prefer standby if available)
    low_power_current_ua = standby_current_ua or sleep_current_ua

    # If we couldn't extract power data, provide estimates or warning
    if not active_current_ua:
        # Rough estimate based on architecture and frequency
        arch = meta.get('architecture', '').lower()
        if 'cortex-m0' in arch:
            active_current_ua = 100 * freq  # ~100µA/MHz for M0+
        else:
            active_current_ua = 150 * freq  # ~150µA/MHz for other architectures
        power_info.append(f"⚠️ Active current estimated (not found in datasheet)")

    if not low_power_current_ua:
        low_power_current_ua = 1.5  # Typical STANDBY current for MSPM0
        power_info.append(f"⚠️ Sleep current estimated (not found in datasheet)")

    # Calculate average current
    run_fraction = run_time_pct / 100
    sleep_fraction = sleep_time_pct / 100
    idle_fraction = 1 - run_fraction - sleep_fraction

    avg_current_ua = (
        active_current_ua * run_fraction +
        low_power_current_ua * sleep_fraction +
        low_power_current_ua * idle_fraction  # Assume idle = sleep
    )

    # Calculate battery life
    avg_current_ma = avg_current_ua / 1000
    battery_life_hours = battery_capacity_mah / avg_current_ma
    battery_life_days = battery_life_hours / 24
    battery_life_years = battery_life_days / 365.25

    # Get datasheet link
    datasheet_link = (
        meta.get('pdf_datasheet_url') or
        meta.get('html_datasheet_url') or
        meta.get('product_page_url') or
        meta.get('datasheet_link', '')
    )

    # Format output
    output = [f"## Battery Life Estimate for {part_number}\n\n"]

    if datasheet_link:
        output.append(f"📄 [View Datasheet]({datasheet_link})\n\n")

    output.append(f"**Configuration:**\n")
    output.append(f"- Battery: {battery_capacity_mah} mAh\n")
    output.append(f"- Active time: {run_time_pct}% at {freq}MHz\n")
    output.append(f"- Sleep time: {sleep_time_pct}%\n")
    output.append(f"- Idle time: {(100 - run_time_pct - sleep_time_pct):.1f}%\n\n")

    output.append(f"**Power Consumption:**\n")
    output.append(f"- Active: ~{active_current_ua:.0f} µA at {freq}MHz\n")
    output.append(f"- Sleep/Standby: ~{low_power_current_ua:.1f} µA\n")
    output.append(f"- **Average: {avg_current_ua:.1f} µA ({avg_current_ma:.3f} mA)**\n\n")

    output.append(f"**Estimated Battery Life:**\n")
    
    if battery_life_years >= 1:
        output.append(f"- **{battery_life_years:.1f} years** ({battery_life_days:.0f} days)\n")
    elif battery_life_days >= 1:
        output.append(f"- **{battery_life_days:.1f} days** ({battery_life_hours:.0f} hours)\n")
    else:
        output.append(f"- **{battery_life_hours:.1f} hours**\n")

    if power_info:
        output.append(f"\n**Notes:**\n")
        for note in power_info:
            output.append(f"- {note}\n")
        output.append(f"- For accurate estimates, consult the datasheet's power consumption section\n")

    # Add optimization tips
    if run_time_pct > 20:
        output.append(f"\n**Optimization Tips:**\n")
        output.append(f"- Reduce active time to extend battery life significantly\n")
        if freq > 32:
            output.append(f"- Consider running at lower frequency when possible\n")
        output.append(f"- Use sleep modes aggressively between operations\n")

    return "".join(output)


def find_cheaper_alternative_tool(
    part_number: str,
    must_have_features: Optional[List[str]] = None,
    max_price_reduction_pct: float = 50
) -> str:
    """
    Find cheaper alternatives to a given part.

    Args:
        part_number: Reference part number
        must_have_features: Features that must be preserved (e.g., ["USB", "CAN-FD"])
        max_price_reduction_pct: Maximum acceptable price reduction (default: 50%)

    Returns:
        List of cheaper alternatives with cost savings
    """
    tools = SearchTools()
    
    # Get reference part
    ref_chunks = tools.get_by_part_number(part_number)
    if not ref_chunks:
        return f"Part {part_number} not found in database."

    ref_meta = ref_chunks[0]['metadata']

    # Get reference specs
    ref_price_str = ref_meta.get('price_usd', '')
    if not ref_price_str:
        return f"No pricing information available for {part_number}."

    try:
        ref_price = float(ref_price_str)
    except (ValueError, TypeError):
        return f"Invalid pricing data for {part_number}."

    ref_flash = ref_meta.get('flash_kb_max', 0)
    ref_ram = ref_meta.get('ram_kb_max', 0)
    ref_freq = ref_meta.get('core_freq_mhz', 0)
    ref_arch = ref_meta.get('architecture', '')
    ref_features = ref_meta.get('key_features', '').lower()
    ref_peripherals = ref_meta.get('peripherals', '').lower()

    # Search for alternatives
    all_results = tools.collection.get(
        limit=10000,
        include=["metadatas"]
    )

    alternatives = []

    for metadata in all_results["metadatas"]:
        alt_part = metadata.get("part_numbers", "")

        if not alt_part or alt_part == ref_meta.get("part_numbers"):
            continue

        # Get alternative price
        alt_price_str = metadata.get('price_usd', '')
        if not alt_price_str:
            continue

        try:
            alt_price = float(alt_price_str)
        except (ValueError, TypeError):
            continue

        # Must be cheaper
        if alt_price >= ref_price:
            continue

        # Calculate savings
        savings = ref_price - alt_price
        savings_pct = (savings / ref_price) * 100

        # Check if savings is within acceptable range
        if savings_pct > max_price_reduction_pct:
            continue

        # Get alternative specs
        alt_flash = metadata.get('flash_kb_max', 0)
        alt_ram = metadata.get('ram_kb_max', 0)
        alt_freq = metadata.get('core_freq_mhz', 0)
        alt_arch = metadata.get('architecture', '')

        # Check architecture match (prefer same architecture)
        arch_match = alt_arch.lower() == ref_arch.lower() if ref_arch and alt_arch else False

        # Check must-have features
        if must_have_features:
            alt_features = metadata.get('key_features', '').lower()
            alt_peripherals = metadata.get('peripherals', '').lower()
            combined_alt = alt_features + " " + alt_peripherals

            missing_features = []
            for feature in must_have_features:
                if feature.lower() not in combined_alt:
                    missing_features.append(feature)

            if missing_features:
                continue  # Skip this alternative

        # Identify tradeoffs
        tradeoffs = []
        if alt_flash < ref_flash:
            tradeoffs.append(f"-{ref_flash - alt_flash}KB flash")
        if alt_ram < ref_ram:
            tradeoffs.append(f"-{ref_ram - alt_ram}KB RAM")
        if alt_freq < ref_freq:
            tradeoffs.append(f"-{ref_freq - alt_freq}MHz")
        if not arch_match and ref_arch:
            tradeoffs.append(f"Different arch: {alt_arch}")

        # Benefits
        benefits = []
        if alt_flash > ref_flash:
            benefits.append(f"+{alt_flash - ref_flash}KB flash")
        if alt_ram > ref_ram:
            benefits.append(f"+{alt_ram - ref_ram}KB RAM")
        if alt_freq > ref_freq:
            benefits.append(f"+{alt_freq - ref_freq}MHz")

        # Get datasheet link
        datasheet_link = (
            metadata.get('pdf_datasheet_url') or
            metadata.get('html_datasheet_url') or
            metadata.get('product_page_url') or
            metadata.get('datasheet_link', '')
        )

        alternatives.append({
            'part_number': alt_part,
            'price': alt_price,
            'savings': savings,
            'savings_pct': savings_pct,
            'flash': alt_flash,
            'ram': alt_ram,
            'freq': alt_freq,
            'arch': alt_arch,
            'tradeoffs': tradeoffs,
            'benefits': benefits,
            'datasheet_link': datasheet_link,
            'arch_match': arch_match
        })

    if not alternatives:
        return f"No cheaper alternatives found for {part_number} within acceptable price range."

    # Sort by savings (highest first)
    sorted_alternatives = sorted(alternatives, key=lambda x: x['savings_pct'], reverse=True)[:10]

    # Format output
    output = [f"## Cheaper Alternatives to {part_number}\n\n"]

    # Show reference
    ref_datasheet = (
        ref_meta.get('pdf_datasheet_url') or
        ref_meta.get('html_datasheet_url') or
        ref_meta.get('product_page_url') or
        ref_meta.get('datasheet_link', '')
    )

    if ref_datasheet:
        output.append(f"**Reference:** {part_number} - 📄 [Datasheet]({ref_datasheet})\n")
    else:
        output.append(f"**Reference:** {part_number}\n")
    
    output.append(f"- Price: ${ref_price:.2f}\n")
    output.append(f"- Specs: {ref_flash}KB flash, {ref_ram}KB RAM, {ref_freq}MHz\n")
    output.append(f"- Architecture: {ref_arch}\n\n")

    if must_have_features:
        output.append(f"**Must-Have Features:** {', '.join(must_have_features)}\n\n")

    output.append(f"**Found {len(sorted_alternatives)} cheaper alternatives:**\n\n")

    for i, alt in enumerate(sorted_alternatives, 1):
        link_text = f" - 📄 [Datasheet]({alt['datasheet_link']})" if alt['datasheet_link'] else ""

        output.append(f"{i}. **{alt['part_number']}**{link_text}\n")
        output.append(f"   - **Price: ${alt['price']:.2f}** (Save ${alt['savings']:.2f} / {alt['savings_pct']:.1f}%)\n")
        output.append(f"   - Specs: {alt['flash']}KB flash, {alt['ram']}KB RAM, {alt['freq']}MHz\n")

        if alt['benefits']:
            output.append(f"   - Benefits: {', '.join(alt['benefits'])}\n")

        if alt['tradeoffs']:
            output.append(f"   - Tradeoffs: {', '.join(alt['tradeoffs'])}\n")

        output.append("\n")

    # Show potential annual savings for volume production
    best_savings = sorted_alternatives[0]['savings']
    output.append(f"**💰 Cost Savings Analysis:**\n")
    output.append(f"- Per unit: ${best_savings:.2f}\n")
    output.append(f"- 1,000 units: ${best_savings * 1000:,.2f}\n")
    output.append(f"- 10,000 units: ${best_savings * 10000:,.2f}\n")
    output.append(f"- 100,000 units: ${best_savings * 100000:,.2f}\n")

    return "".join(output)


def check_lifecycle_status_tool(part_numbers: List[str]) -> str:
    """
    Check lifecycle status of parts (ACTIVE, PREVIEW, NRND, etc.).

    Args:
        part_numbers: List of part numbers to check

    Returns:
        Lifecycle status for each part
    """
    tools = SearchTools()
    results = []

    for part_num in part_numbers:
        chunks = tools.get_by_part_number(part_num)

        if not chunks:
            results.append({
                'part_number': part_num,
                'status': 'NOT_FOUND',
                'error': 'Part not found in database'
            })
            continue

        meta = chunks[0]['metadata']

        status = meta.get('status') or meta.get('Status', 'UNKNOWN')
        rating = meta.get('rating') or meta.get('Rating', 'N/A')
        
        # Get datasheet link
        datasheet_link = (
            meta.get('pdf_datasheet_url') or
            meta.get('html_datasheet_url') or
            meta.get('product_page_url') or
            meta.get('datasheet_link', '')
        )

        results.append({
            'part_number': part_num,
            'status': status,
            'rating': rating,
            'datasheet_link': datasheet_link
        })

    # Format output
    output = [f"## Lifecycle Status Check\n\n"]

    # Group by status
    active_parts = [r for r in results if r['status'] == 'ACTIVE']
    preview_parts = [r for r in results if r['status'] == 'PREVIEW']
    nrnd_parts = [r for r in results if 'NRND' in r['status'].upper()]
    other_parts = [r for r in results if r['status'] not in ['ACTIVE', 'PREVIEW', 'UNKNOWN', 'NOT_FOUND'] and 'NRND' not in r['status'].upper()]
    unknown_parts = [r for r in results if r['status'] in ['UNKNOWN', 'NOT_FOUND']]

    if active_parts:
        output.append(f"### ✅ ACTIVE ({len(active_parts)} parts)\n")
        output.append("These parts are in full production and recommended for new designs:\n\n")
        for r in active_parts:
            link = f" - 📄 [Datasheet]({r['datasheet_link']})" if r['datasheet_link'] else ""
            output.append(f"- **{r['part_number']}**{link}\n")
            output.append(f"  - Rating: {r['rating']}\n")
        output.append("\n")

    if preview_parts:
        output.append(f"### 🔍 PREVIEW ({len(preview_parts)} parts)\n")
        output.append("These parts are in pre-production. Contact TI for availability:\n\n")
        for r in preview_parts:
            link = f" - 📄 [Datasheet]({r['datasheet_link']})" if r['datasheet_link'] else ""
            output.append(f"- **{r['part_number']}**{link}\n")
        output.append("\n")

    if nrnd_parts:
        output.append(f"### ⚠️ NRND ({len(nrnd_parts)} parts)\n")
        output.append("Not Recommended for New Designs. Plan for alternatives:\n\n")
        for r in nrnd_parts:
            link = f" - 📄 [Datasheet]({r['datasheet_link']})" if r['datasheet_link'] else ""
            output.append(f"- **{r['part_number']}**{link}\n")
        output.append("\n")

    if other_parts:
        output.append(f"### ℹ️ OTHER STATUS ({len(other_parts)} parts)\n")
        for r in other_parts:
            link = f" - 📄 [Datasheet]({r['datasheet_link']})" if r['datasheet_link'] else ""
            output.append(f"- **{r['part_number']}**: {r['status']}{link}\n")
        output.append("\n")

    if unknown_parts:
        output.append(f"### ❓ UNKNOWN ({len(unknown_parts)} parts)\n")
        for r in unknown_parts:
            if 'error' in r:
                output.append(f"- **{r['part_number']}**: {r['error']}\n")
            else:
                output.append(f"- **{r['part_number']}**: Status unknown\n")
        output.append("\n")

    # Add recommendations
    if nrnd_parts or unknown_parts:
        output.append(f"**Recommendations:**\n")
        if nrnd_parts:
            output.append(f"- Consider migrating from NRND parts to active alternatives\n")
        if unknown_parts:
            output.append(f"- Verify status of unknown parts on TI.com\n")

    return "".join(output)


def create_competitor_kill_sheet_tool(
    competitor_part: str,
    competitor_specs: Optional[Dict[str, Any]] = None,
    use_case: Optional[str] = None
) -> str:
    """
    Create a competitive analysis comparing TI parts to competitor offerings.
    
    Args:
        competitor_part: Competitor part number (e.g., "STM32L476", "ATmega328")
        competitor_specs: Optional specs dict with keys like:
            - architecture: str (e.g., "ARM Cortex-M4")
            - freq_mhz: int
            - flash_kb: int
            - ram_kb: int
            - peripherals: List[str]
            - price: float
        use_case: Optional application description
    
    Returns:
        Competitive kill sheet with TI advantages
    """
    tools = SearchTools()
    
    # Parse competitor part to identify likely specs
    competitor_info = _parse_competitor_part(competitor_part)
    
    # Merge with provided specs
    if competitor_specs:
        competitor_info.update(competitor_specs)
    
    # Find matching TI parts
    min_flash = competitor_info.get('flash_kb', 64)
    min_ram = competitor_info.get('ram_kb', 16)
    min_freq = competitor_info.get('freq_mhz', 48)
    architecture = competitor_info.get('architecture')
    
    # Get all potential TI alternatives
    all_results = tools.collection.get(
        limit=10000,
        include=["metadatas"]
    )
    
    candidates = []
    
    for metadata in all_results["metadatas"]:
        part_num = metadata.get("part_numbers", "")
        if not part_num:
            continue
        
        # Get TI part specs
        ti_flash = metadata.get('flash_kb_max', 0)
        ti_ram = metadata.get('ram_kb_max', 0)
        ti_freq = metadata.get('core_freq_mhz', 0)
        ti_arch = metadata.get('architecture', '')
        ti_price_str = metadata.get('price_usd', '')
        
        # Must meet minimum requirements
        if ti_flash < min_flash * 0.75:  # Allow 25% tolerance
            continue
        if ti_ram < min_ram * 0.75:
            continue
        if ti_freq < min_freq * 0.75:
            continue
        
        # Calculate match score
        score = 0
        advantages = []
        
        # Flash advantage
        if ti_flash >= min_flash:
            flash_advantage = ((ti_flash - min_flash) / min_flash) * 100
            if flash_advantage > 0:
                advantages.append(f"+{int(flash_advantage)}% more flash")
                score += flash_advantage * 0.5
        
        # RAM advantage
        if ti_ram >= min_ram:
            ram_advantage = ((ti_ram - min_ram) / min_ram) * 100
            if ram_advantage > 0:
                advantages.append(f"+{int(ram_advantage)}% more RAM")
                score += ram_advantage * 0.5
        
        # Frequency advantage
        if ti_freq >= min_freq:
            freq_advantage = ((ti_freq - min_freq) / min_freq) * 100
            if freq_advantage > 0:
                advantages.append(f"+{int(freq_advantage)}% faster")
                score += freq_advantage * 0.3
        
        # Price advantage
        competitor_price = competitor_info.get('price', 1.0)
        if ti_price_str:
            try:
                ti_price = float(ti_price_str)
                if ti_price < competitor_price:
                    price_savings = ((competitor_price - ti_price) / competitor_price) * 100
                    advantages.append(f"{int(price_savings)}% cheaper")
                    score += price_savings * 2  # Price is important!
            except (ValueError, TypeError):
                ti_price = None
        else:
            ti_price = None
        
        # Get additional info
        datasheet_link = (
            metadata.get('pdf_datasheet_url') or
            metadata.get('html_datasheet_url') or
            metadata.get('product_page_url') or
            metadata.get('datasheet_link', '')
        )
        
        peripherals = metadata.get('peripherals', '')
        features = metadata.get('key_features', '')
        
        candidates.append({
            'part_number': part_num,
            'flash_kb': ti_flash,
            'ram_kb': ti_ram,
            'freq_mhz': ti_freq,
            'architecture': ti_arch,
            'price': ti_price,
            'score': score,
            'advantages': advantages,
            'datasheet_link': datasheet_link,
            'peripherals': peripherals,
            'features': features,
            'metadata': metadata
        })
    
    if not candidates:
        return f"No TI alternatives found matching {competitor_part} specifications."
    
    # Sort by score (best matches first)
    sorted_candidates = sorted(candidates, key=lambda x: x['score'], reverse=True)[:5]
    
    # Build kill sheet
    output = [f"# 🎯 TI Competitive Analysis: {competitor_part} Alternatives\n\n"]
    
    # Competitor overview
    output.append(f"## Competitor Product: {competitor_part}\n\n")
    output.append(f"**Estimated Specifications:**\n")
    if competitor_info.get('vendor'):
        output.append(f"- Vendor: {competitor_info['vendor']}\n")
    if competitor_info.get('architecture'):
        output.append(f"- Architecture: {competitor_info['architecture']}\n")
    output.append(f"- Flash: ~{min_flash}KB\n")
    output.append(f"- RAM: ~{min_ram}KB\n")
    output.append(f"- Frequency: ~{min_freq}MHz\n")
    if competitor_info.get('price'):
        output.append(f"- Estimated Price: ${competitor_info['price']:.2f}\n")
    output.append("\n")
    
    # TI alternatives
    output.append(f"## 🔥 Recommended TI Alternatives\n\n")
    
    for i, ti_part in enumerate(sorted_candidates, 1):
        link_text = f" - 📄 [Datasheet]({ti_part['datasheet_link']})" if ti_part['datasheet_link'] else ""
        
        output.append(f"### {i}. **{ti_part['part_number']}**{link_text}\n\n")
        
        # Specifications
        output.append(f"**Specifications:**\n")
        output.append(f"- Architecture: {ti_part['architecture']}\n")
        output.append(f"- Flash: {ti_part['flash_kb']}KB\n")
        output.append(f"- RAM: {ti_part['ram_kb']}KB\n")
        output.append(f"- Frequency: {ti_part['freq_mhz']}MHz\n")
        if ti_part['price']:
            output.append(f"- Price: ${ti_part['price']:.2f}\n")
        output.append("\n")
        
        # Advantages
        if ti_part['advantages']:
            output.append(f"**✅ Key Advantages:**\n")
            for adv in ti_part['advantages']:
                output.append(f"- {adv}\n")
            output.append("\n")
    
    # Why choose TI section
    output.append(f"## 💡 Why Choose TI?\n\n")
    
    # Generic TI advantages
    output.append(f"**1. Industry-Leading Low Power**\n")
    output.append(f"- TI MSPM0 series offers industry-leading standby current (as low as 88nA)\n")
    output.append(f"- Multiple low-power modes for optimal battery life\n")
    output.append(f"- Smart peripherals that operate independently in sleep modes\n\n")
    
    output.append(f"**2. Superior Development Ecosystem**\n")
    output.append(f"- Free Code Composer Studio IDE with advanced debugging\n")
    output.append(f"- Extensive SDK and middleware libraries\n")
    output.append(f"- LaunchPad development boards at low cost\n")
    output.append(f"- Active E2E support community\n\n")
    
    output.append(f"**3. Better Value**\n")
    if sorted_candidates[0]['price'] and competitor_info.get('price'):
        savings_per_unit = competitor_info['price'] - sorted_candidates[0]['price']
        if savings_per_unit > 0:
            output.append(f"- Save ${savings_per_unit:.2f}/unit vs {competitor_part}\n")
            output.append(f"- Volume savings:\n")
            output.append(f"  - 10K units: ${savings_per_unit * 10000:,.0f}\n")
            output.append(f"  - 100K units: ${savings_per_unit * 100000:,.0f}\n")
            output.append(f"  - 1M units: ${savings_per_unit * 1000000:,.0f}\n")
    output.append(f"- More memory and performance for the price\n")
    output.append(f"- Long-term availability guarantee\n\n")
    
    output.append(f"**4. Advanced Features**\n")
    # Highlight unique TI features from best match
    best_features = sorted_candidates[0]['features'].split(',')[:5]
    for feature in best_features:
        if feature.strip():
            output.append(f"- {feature.strip()}\n")
    output.append("\n")
    
    output.append(f"**5. Security & Safety**\n")
    output.append(f"- Hardware-enforced security features\n")
    output.append(f"- Cryptographic acceleration\n")
    output.append(f"- Functional safety options available\n")
    output.append(f"- Secure boot and firmware updates\n\n")
    
    # Migration considerations
    output.append(f"## 🔄 Migration Considerations\n\n")
    output.append(f"**Easy Migration Path:**\n")
    output.append(f"- Both use ARM Cortex architecture - familiar toolchain\n")
    output.append(f"- Standard ARM CMSIS support\n")
    output.append(f"- Similar peripheral set for easy code porting\n")
    output.append(f"- TI provides migration guides and examples\n\n")
    
    output.append(f"**TI Support:**\n")
    output.append(f"- Free migration assistance from TI field engineers\n")
    output.append(f"- Reference designs and application notes\n")
    output.append(f"- Sample kits available for evaluation\n\n")
    
    # Call to action
    output.append(f"## 📞 Next Steps\n\n")
    output.append(f"1. **Request Samples:** Get free samples of recommended TI parts\n")
    output.append(f"2. **Download SDK:** Access free development tools and code examples\n")
    output.append(f"3. **Contact FAE:** Speak with a TI Field Application Engineer for personalized support\n")
    output.append(f"4. **Evaluate:** Test TI parts with LaunchPad development boards\n\n")
    
    output.append(f"---\n")
    output.append(f"*Ready to make the switch? TI parts offer better performance, lower power, and superior value.*\n")
    
    return "".join(output)


def _parse_competitor_part(part_number: str) -> Dict[str, Any]:
    """
    Parse competitor part number to estimate specs.
    
    This is a heuristic-based parser for common competitor families.
    """
    part_upper = part_number.upper()
    info = {}
    
    # STM32 family
    if 'STM32' in part_upper:
        info['vendor'] = 'STMicroelectronics'
        
        # STM32L = Low power
        if 'STM32L' in part_upper:
            info['architecture'] = 'ARM Cortex-M4'
            info['freq_mhz'] = 80
            info['flash_kb'] = 256
            info['ram_kb'] = 64
            info['price'] = 2.5  # Estimate
            
        # STM32F = Mainstream
        elif 'STM32F' in part_upper:
            info['architecture'] = 'ARM Cortex-M4'
            info['freq_mhz'] = 100
            info['flash_kb'] = 256
            info['ram_kb'] = 64
            info['price'] = 2.0
            
        # STM32G = Mainstream low-power
        elif 'STM32G' in part_upper:
            info['architecture'] = 'ARM Cortex-M4'
            info['freq_mhz'] = 170
            info['flash_kb'] = 512
            info['ram_kb'] = 128
            info['price'] = 2.8
    
    # AVR/ATmega
    elif 'ATMEGA' in part_upper or 'AVR' in part_upper:
        info['vendor'] = 'Microchip (Atmel)'
        info['architecture'] = '8-bit AVR'
        info['freq_mhz'] = 16
        info['flash_kb'] = 32
        info['ram_kb'] = 2
        info['price'] = 1.5
    
    # NXP LPC
    elif 'LPC' in part_upper:
        info['vendor'] = 'NXP'
        info['architecture'] = 'ARM Cortex-M0+'
        info['freq_mhz'] = 50
        info['flash_kb'] = 256
        info['ram_kb'] = 64
        info['price'] = 2.0
    
    # Nordic nRF
    elif 'NRF5' in part_upper or 'NRF52' in part_upper:
        info['vendor'] = 'Nordic Semiconductor'
        info['architecture'] = 'ARM Cortex-M4'
        info['freq_mhz'] = 64
        info['flash_kb'] = 512
        info['ram_kb'] = 64
        info['price'] = 3.0
    
    # Silicon Labs EFM32
    elif 'EFM32' in part_upper:
        info['vendor'] = 'Silicon Labs'
        info['architecture'] = 'ARM Cortex-M4'
        info['freq_mhz'] = 80
        info['flash_kb'] = 256
        info['ram_kb'] = 32
        info['price'] = 2.5
    
    # Renesas RA/RX
    elif 'RA4' in part_upper or 'RA6' in part_upper:
        info['vendor'] = 'Renesas'
        info['architecture'] = 'ARM Cortex-M4'
        info['freq_mhz'] = 100
        info['flash_kb'] = 512
        info['ram_kb'] = 128
        info['price'] = 2.8
    
    # Default for unknown
    else:
        info['vendor'] = 'Unknown'
        info['architecture'] = 'ARM Cortex-M'
        info['freq_mhz'] = 64
        info['flash_kb'] = 128
        info['ram_kb'] = 32
        info['price'] = 2.0
    
    return info


def synthesize_use_case_solution_tool(
    use_case: str,
    constraints: Optional[Dict[str, Any]] = None
) -> str:
    """
    Create a narrative solution synthesis for a use case.
    
    Provides a story-driven recommendation with architecture guidance,
    not just technical specs.
    
    Args:
        use_case: Application description (e.g., "battery-powered soil moisture sensor")
        constraints: Optional constraints (budget, size, battery_life_years, etc.)
    
    Returns:
        Narrative solution with architecture, reasoning, and implementation guidance
    """
    tools = SearchTools()
    
    # Parse use case to extract key requirements
    use_case_lower = use_case.lower()
    
    # Identify application domain
    domain = _identify_application_domain(use_case_lower)
    
    # Extract technical requirements from use case
    requirements = _extract_requirements_from_use_case(use_case_lower, constraints)
    
    # Find suitable MCUs
    candidates = []
    all_results = tools.collection.get(
        limit=10000,
        include=["metadatas", "documents"]
    )
    
    for i, metadata in enumerate(all_results["metadatas"]):
        part_num = metadata.get("part_numbers", "")
        if not part_num:
            continue
        
        # Score based on use case fit
        score = _score_use_case_fit(metadata, requirements, use_case_lower)
        
        if score > 0:
            candidates.append({
                'part_number': part_num,
                'score': score,
                'metadata': metadata,
                'content': all_results["documents"][i] if i < len(all_results["documents"]) else ""
            })
    
    if not candidates:
        return f"No suitable parts found for use case: {use_case}"
    
    # Get top recommendations
    sorted_candidates = sorted(candidates, key=lambda x: x['score'], reverse=True)[:3]
    primary_solution = sorted_candidates[0]
    
    # Build narrative solution
    output = [f"# 🎯 Solution Architecture: {domain['name']}\n\n"]
    
    # Executive Summary
    output.append(f"## Executive Summary\n\n")
    output.append(_generate_executive_summary(use_case, primary_solution, requirements, domain))
    output.append("\n")
    
    # Solution Architecture
    output.append(f"## 🏗️ Recommended Solution Architecture\n\n")
    output.append(_generate_architecture_narrative(primary_solution, sorted_candidates[1:], requirements, domain))
    output.append("\n")
    
    # Why This Solution
    output.append(f"## 💡 Why This Solution?\n\n")
    output.append(_generate_solution_reasoning(primary_solution, requirements, domain))
    output.append("\n")
    
    # Key Benefits
    output.append(f"## ✅ Key Benefits\n\n")
    output.append(_generate_benefits(primary_solution, requirements, domain))
    output.append("\n")
    
    # Implementation Guidance
    output.append(f"## 🛠️ Implementation Guidance\n\n")
    output.append(_generate_implementation_guidance(primary_solution, requirements, domain))
    output.append("\n")
    
    # Alternative Considerations
    if len(sorted_candidates) > 1:
        output.append(f"## 🔄 Alternative Considerations\n\n")
        output.append(_generate_alternatives(sorted_candidates[1:], primary_solution))
        output.append("\n")
    
    return "".join(output)


def _identify_application_domain(use_case: str) -> Dict[str, Any]:
    """Identify the application domain from use case description."""
    domains = {
        'iot_sensor': {
            'name': 'IoT Sensor Application',
            'keywords': ['sensor', 'iot', 'wireless', 'battery', 'monitoring'],
            'priorities': ['low_power', 'wireless', 'small_size'],
            'typical_battery': 'CR2032 or AA',
            'target_life': '5-10 years'
        },
        'industrial': {
            'name': 'Industrial Control Application',
            'keywords': ['industrial', 'control', 'automation', 'plc', 'motor'],
            'priorities': ['reliability', 'real_time', 'communication'],
            'typical_battery': 'Mains powered',
            'target_life': '10-20 years'
        },
        'wearable': {
            'name': 'Wearable Device',
            'keywords': ['wearable', 'fitness', 'health', 'watch', 'tracker'],
            'priorities': ['ultra_low_power', 'small_size', 'ble'],
            'typical_battery': 'Coin cell',
            'target_life': '1-2 years'
        },
        'smart_home': {
            'name': 'Smart Home Application',
            'keywords': ['smart home', 'thermostat', 'lighting', 'door', 'lock'],
            'priorities': ['low_power', 'wireless', 'security'],
            'typical_battery': 'AA or mains',
            'target_life': '2-5 years'
        },
        'medical': {
            'name': 'Medical/Healthcare Application',
            'keywords': ['medical', 'health', 'patient', 'vital', 'diagnostic'],
            'priorities': ['reliability', 'security', 'low_power'],
            'typical_battery': 'Rechargeable',
            'target_life': '5+ years'
        }
    }
    
    # Match keywords
    for domain_key, domain_info in domains.items():
        if any(keyword in use_case for keyword in domain_info['keywords']):
            return domain_info
    
    # Default
    return {
        'name': 'General Purpose Application',
        'keywords': [],
        'priorities': ['performance', 'versatility'],
        'typical_battery': 'Various',
        'target_life': 'Application dependent'
    }


def _extract_requirements_from_use_case(use_case: str, constraints: Optional[Dict]) -> Dict[str, Any]:
    """Extract technical requirements from use case description."""
    import re
    
    reqs = constraints.copy() if constraints else {}
    
    # Battery life
    if 'multi-year' in use_case or 'coin cell' in use_case or 'coin-cell' in use_case:
        reqs['battery_life_target'] = '5+ years'
        reqs['ultra_low_power'] = True
    elif 'battery' in use_case:
        reqs['low_power'] = True
    
    # Wireless
    if 'ble' in use_case or 'bluetooth' in use_case:
        reqs['ble_required'] = True
    if 'wifi' in use_case or 'wi-fi' in use_case:
        reqs['wifi_required'] = True
    
    # Interfaces
    if 'usb' in use_case:
        reqs['usb_required'] = True
    if 'can' in use_case:
        reqs['can_required'] = True
    
    # Size constraints
    if 'small' in use_case or 'compact' in use_case or 'tiny' in use_case:
        reqs['small_package'] = True
    
    # Volume
    volume_match = re.search(r'(\d+)k?\s*units', use_case)
    if volume_match:
        reqs['volume'] = volume_match.group(1)
    
    return reqs


def _score_use_case_fit(metadata: Dict, requirements: Dict, use_case: str) -> float:
    """Score how well a part fits the use case."""
    score = 0.0
    
    # Ultra-low-power requirement
    if requirements.get('ultra_low_power'):
        if 'MSPM0' in metadata.get('part_numbers', ''):
            score += 50  # MSPM0 is ultra-low-power
    
    # Low power requirement
    if requirements.get('low_power'):
        if 'MSPM0' in metadata.get('part_numbers', ''):
            score += 30
    
    # BLE requirement
    if requirements.get('ble_required'):
        features = metadata.get('key_features', '').lower()
        if 'ble' in features or 'bluetooth' in features:
            score += 40
    
    # USB requirement
    if requirements.get('usb_required'):
        features = metadata.get('key_features', '').lower()
        peripherals = metadata.get('peripherals', '').lower()
        if 'usb' in features or 'usb' in peripherals:
            score += 30
    
    # CAN requirement
    if requirements.get('can_required'):
        features = metadata.get('key_features', '').lower()
        if 'can' in features:
            score += 30
    
    # Small package
    if requirements.get('small_package'):
        package = metadata.get('package_types', '').upper()
        if 'VQFN' in package or 'QFN' in package:
            score += 20
    
    # Memory (more is better for IoT)
    flash = metadata.get('flash_kb_max', 0)
    if flash >= 128:
        score += 15
    if flash >= 256:
        score += 10
    
    # Active status
    if metadata.get('Status', '').upper() == 'ACTIVE':
        score += 10
    
    return score


def _generate_executive_summary(use_case: str, solution: Dict, requirements: Dict, domain: Dict) -> str:
    """Generate executive summary."""
    meta = solution['metadata']
    part = solution['part_number']
    
    flash = meta.get('flash_kb_max', 0)
    ram = meta.get('ram_kb_max', 0)
    freq = meta.get('core_freq_mhz', 0)
    
    summary = []
    summary.append(f"For **{use_case}**, we recommend the **{part}** as your primary MCU.\n\n")
    
    # Key highlights
    summary.append(f"**Solution Highlights:**\n")
    
    if requirements.get('ultra_low_power') or requirements.get('battery_life_target'):
        summary.append(f"- ⚡ **{domain['target_life']} battery life** on {domain['typical_battery']}\n")
    
    summary.append(f"- 🎯 **{flash}KB flash, {ram}KB RAM** - ample for your application\n")
    
    if 'MSPM0' in part:
        summary.append(f"- 🔋 **Industry-leading low power** - 88nA shutdown, <2µA standby\n")
    
    features = meta.get('key_features', '').split(',')[:3]
    if features and features[0]:
        summary.append(f"- 🚀 **Integrated peripherals**: {', '.join([f.strip() for f in features if f.strip()])}\n")
    
    price = meta.get('price_usd', '')
    if price:
        try:
            price_val = float(price)
            summary.append(f"- 💰 **Cost-effective**: ~${price_val:.2f}/unit at volume\n")
        except:
            pass
    
    return "".join(summary)


def _generate_architecture_narrative(primary: Dict, alternatives: List, requirements: Dict, domain: Dict) -> str:
    """Generate architecture narrative."""
    meta = primary['metadata']
    part = primary['part_number']
    
    arch = []
    
    # Primary MCU
    datasheet_link = (
        meta.get('pdf_datasheet_url') or
        meta.get('html_datasheet_url') or
        meta.get('product_page_url', '')
    )
    
    link_text = f" - 📄 [Datasheet]({datasheet_link})" if datasheet_link else ""
    
    arch.append(f"### Core Microcontroller: **{part}**{link_text}\n\n")
    
    arch.append(f"**Why this MCU:**\n")
    arch.append(f"- **Processing**: {meta.get('architecture', 'ARM Cortex')} at {meta.get('core_freq_mhz', 0)}MHz\n")
    arch.append(f"- **Memory**: {meta.get('flash_kb_max', 0)}KB flash for your application code + OTA updates\n")
    arch.append(f"- **RAM**: {meta.get('ram_kb_max', 0)}KB SRAM for data buffers and processing\n")
    
    # Peripherals narrative
    peripherals = meta.get('peripherals', '').split(',')
    if peripherals and peripherals[0]:
        arch.append(f"- **Integrated peripherals**: {', '.join([p.strip() for p in peripherals[:5] if p.strip()])}\n")
    
    arch.append("\n")
    
    # Power architecture
    if requirements.get('ultra_low_power') or requirements.get('low_power'):
        arch.append(f"**Power Architecture:**\n")
        arch.append(f"- Active mode: ~100µA/MHz (optimized for efficiency)\n")
        arch.append(f"- Sleep mode: ~1-2µA (with RTC and RAM retention)\n")
        arch.append(f"- Shutdown: ~88nA (wake on GPIO/RTC)\n")
        arch.append(f"- **Result**: {domain['target_life']} on {domain['typical_battery']}\n\n")
    
    # Integration benefits
    arch.append(f"**Integration Benefits:**\n")
    features = meta.get('key_features', '').lower()
    
    if 'aes' in features or 'security' in features:
        arch.append(f"- **Built-in security**: Hardware AES encryption, secure boot\n")
    
    if 'dma' in features:
        arch.append(f"- **DMA channels**: Zero-CPU data transfers for efficiency\n")
    
    if 'adc' in features or 'adc' in meta.get('peripherals', '').lower():
        arch.append(f"- **Integrated ADC**: {meta.get('adc_type', '12-bit')} for sensor interfacing\n")
    
    arch.append(f"- **Single-chip solution**: Reduces PCB footprint vs discrete components\n")
    
    return "".join(arch)


def _generate_solution_reasoning(solution: Dict, requirements: Dict, domain: Dict) -> str:
    """Generate reasoning for why this solution."""
    reasoning = []
    
    reasoning.append(f"**1. Optimized for {domain['name']}**\n\n")
    
    for priority in domain['priorities'][:2]:
        if priority == 'low_power':
            reasoning.append(f"- Ultra-low-power architecture extends battery life\n")
        elif priority == 'small_size':
            reasoning.append(f"- Compact package minimizes PCB footprint\n")
        elif priority == 'reliability':
            reasoning.append(f"- Industrial-grade reliability and long lifecycle\n")
        elif priority == 'wireless':
            reasoning.append(f"- Wireless-ready with integrated peripherals\n")
    
    reasoning.append(f"\n**2. Complete Integration**\n\n")
    reasoning.append(f"- All essential peripherals on-chip reduces external components\n")
    reasoning.append(f"- Lower BOM cost and simpler design vs discrete solutions\n")
    reasoning.append(f"- Faster time-to-market with reference designs\n")
    
    reasoning.append(f"\n**3. Proven Ecosystem**\n\n")
    reasoning.append(f"- Free development tools (Code Composer Studio)\n")
    reasoning.append(f"- Extensive SDK and middleware libraries\n")
    reasoning.append(f"- LaunchPad evaluation boards available\n")
    reasoning.append(f"- Active E2E support community\n")
    
    return "".join(reasoning)


def _generate_benefits(solution: Dict, requirements: Dict, domain: Dict) -> str:
    """Generate key benefits."""
    benefits = []
    
    meta = solution['metadata']
    
    # Battery life
    if requirements.get('battery_life_target'):
        benefits.append(f"**Extended Battery Life**: {domain['target_life']} operation\n")
        benefits.append(f"- Reduces maintenance costs and improves user experience\n\n")
    
    # Cost savings
    price = meta.get('price_usd', '')
    if price:
        try:
            price_val = float(price)
            benefits.append(f"**Cost Optimization**\n")
            benefits.append(f"- Competitive pricing: ${price_val:.2f}/unit\n")
            if requirements.get('volume'):
                vol = int(requirements['volume'].replace('k', '000'))
                benefits.append(f"- Total BOM impact at {vol:,} units: ${price_val * vol:,.0f}\n\n")
        except:
            pass
    
    # PCB footprint
    benefits.append(f"**Reduced PCB Footprint**\n")
    benefits.append(f"- Integrated peripherals eliminate external components\n")
    benefits.append(f"- Estimate: 20-30% smaller PCB vs discrete solution\n")
    benefits.append(f"- Lower manufacturing costs and smaller product form factor\n\n")
    
    # Time to market
    benefits.append(f"**Faster Time-to-Market**\n")
    benefits.append(f"- Reference designs available for quick prototyping\n")
    benefits.append(f"- Pre-certified software stacks reduce development time\n")
    benefits.append(f"- Comprehensive documentation and examples\n")
    
    return "".join(benefits)


def _generate_implementation_guidance(solution: Dict, requirements: Dict, domain: Dict) -> str:
    """Generate implementation guidance."""
    guidance = []
    
    part = solution['part_number']
    
    guidance.append(f"**Step 1: Evaluation**\n")
    guidance.append(f"- Order LaunchPad development board for {part}\n")
    guidance.append(f"- Download SDK and sample applications\n")
    guidance.append(f"- Prototype your sensor interfaces and power modes\n\n")
    
    guidance.append(f"**Step 2: Power Budget**\n")
    if requirements.get('battery_life_target'):
        guidance.append(f"- Model your duty cycle (active vs sleep time)\n")
        guidance.append(f"- Calculate average current consumption\n")
        guidance.append(f"- Validate {domain['target_life']} target\n")
        guidance.append(f"- *Tip: Use our battery life estimation tool*\n\n")
    
    guidance.append(f"**Step 3: Schematic Design**\n")
    guidance.append(f"- Use TI reference design as starting point\n")
    guidance.append(f"- Follow power supply decoupling guidelines\n")
    guidance.append(f"- Plan for debug interfaces (JTAG/SWD)\n\n")
    
    guidance.append(f"**Step 4: Software Architecture**\n")
    guidance.append(f"- Leverage TI drivers and middleware\n")
    guidance.append(f"- Implement low-power state machine\n")
    guidance.append(f"- Plan for OTA firmware updates\n\n")
    
    guidance.append(f"**Resources:**\n")
    guidance.append(f"- 📚 TI Resource Explorer (inside CCS)\n")
    guidance.append(f"- 💬 E2E Community Forums\n")
    guidance.append(f"- 📞 Field Application Engineer support\n")
    guidance.append(f"- 🎓 TI Training modules\n")
    
    return "".join(guidance)


def _generate_alternatives(alternatives: List, primary: Dict) -> str:
    """Generate alternative considerations."""
    alts = []
    
    primary_part = primary['part_number']
    
    alts.append(f"If **{primary_part}** doesn't perfectly fit your needs:\n\n")
    
    for i, alt in enumerate(alternatives[:2], 1):
        alt_meta = alt['metadata']
        alt_part = alt['part_number']
        
        datasheet_link = (
            alt_meta.get('pdf_datasheet_url') or
            alt_meta.get('html_datasheet_url') or
            alt_meta.get('product_page_url', '')
        )
        
        link_text = f" - 📄 [Datasheet]({datasheet_link})" if datasheet_link else ""
        
        alts.append(f"**Alternative {i}: {alt_part}**{link_text}\n")
        
        # Why consider this alternative
        if alt_meta.get('flash_kb_max', 0) > primary['metadata'].get('flash_kb_max', 0):
            alts.append(f"- Consider if you need more flash ({alt_meta.get('flash_kb_max')}KB)\n")
        elif alt_meta.get('flash_kb_max', 0) < primary['metadata'].get('flash_kb_max', 0):
            alts.append(f"- Consider for lower cost with less flash ({alt_meta.get('flash_kb_max')}KB)\n")
        
        alts.append("\n")
    
    return "".join(alts)
