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

        prices[part_num] = {
            'unit_price': unit_price,
            'quantity_price': unit_price * quantity if unit_price else None,
            'flash': meta.get('flash_kb_max', 'N/A'),
            'ram': meta.get('ram_kb_max', 'N/A'),
            'frequency': meta.get('core_freq_mhz', 'N/A'),
            'architecture': meta.get('architecture', 'N/A'),
            'package': meta.get('package', 'N/A'),
            'datasheet_link': meta.get('pdf_datasheet_url') or meta.get('html_datasheet_url') or meta.get('datasheet_link', ''),
        }

    # Format output as a table
    output = []
    output.append(f"## Price Comparison ({quantity} units)\n\n")
    output.append("| Part Number | Unit Price | Total ({} units) | Flash | RAM | Frequency | Architecture |\n".format(quantity))
    output.append("|-------------|-----------|-----------------|-------|-----|-----------|-------------|\n")

    for part_num, data in prices.items():
        if 'error' in data:
            output.append(f"| {part_num} | Error | {data['error']} | - | - | - | - |\n")
            continue

        unit_price = f"${data['unit_price']:.2f}" if data['unit_price'] else "N/A"
        total_price = f"${data['quantity_price']:,.2f}" if data['quantity_price'] else "N/A"

        link_text = f"[{part_num}]({data['datasheet_link']})" if data['datasheet_link'] else part_num

        output.append(
            f"| {link_text} | {unit_price} | {total_price} | "
            f"{data['flash']}KB | {data['ram']}KB | {data['frequency']}MHz | "
            f"{data['architecture']} |\n"
        )

    # Find cheapest option
    valid_prices = {k: v for k, v in prices.items() if 'error' not in v and v['unit_price']}
    if valid_prices:
        cheapest = min(valid_prices.items(), key=lambda x: x[1]['unit_price'])
        most_expensive = max(valid_prices.items(), key=lambda x: x[1]['unit_price'])

        savings = most_expensive[1]['unit_price'] - cheapest[1]['unit_price']
        savings_pct = (savings / most_expensive[1]['unit_price']) * 100

        output.append(f"\n**💰 Cost Analysis:**\n")
        output.append(f"- Cheapest: {cheapest[0]} at ${cheapest[1]['unit_price']:.2f}\n")
        output.append(f"- Most expensive: {most_expensive[0]} at ${most_expensive[1]['unit_price']:.2f}\n")
        output.append(f"- Savings: ${savings:.2f} per unit ({savings_pct:.1f}%)\n")
        output.append(f"- Total savings for {quantity} units: ${savings * quantity:,.2f}\n")

    return "".join(output)


def find_parts_by_specs_tool(
    min_flash_kb: int = None,
    min_ram_kb: int = None,
    min_freq_mhz: int = None,
    max_price: float = None,
    required_peripherals: List[str] = None,
    package_type: str = None,
    temp_min: int = None,
    temp_max: int = None,
    architecture: str = None,
    max_results: int = 10
) -> str:
    """
    Find parts matching specification criteria including package, flash, RAM, price.

    Args:
        min_flash_kb: Minimum flash memory in KB
        min_ram_kb: Minimum RAM in KB
        min_freq_mhz: Minimum CPU frequency in MHz
        max_price: Maximum unit price in USD
        required_peripherals: List of required peripherals
        package_type: Package type (e.g., "LQFP", "VQFN", "QFN")
        temp_min: Minimum operating temperature in °C
        temp_max: Maximum operating temperature in °C
        architecture: CPU architecture
        max_results: Maximum number of results

    Returns:
        List of matching parts with specifications
    """
    tools = SearchTools()

    # Build ChromaDB filters
    conditions = []

    if architecture:
        conditions.append({"architecture": architecture})

    if min_freq_mhz:
        conditions.append({"core_freq_mhz": {"$gte": min_freq_mhz}})

    if min_flash_kb:
        conditions.append({"flash_kb_max": {"$gte": min_flash_kb}})

    if min_ram_kb:
        conditions.append({"ram_kb_max": {"$gte": min_ram_kb}})

    if max_price:
        conditions.append({"price_usd": {"$lte": max_price}})

    if package_type:
        # Package matching: exact or contains
        conditions.append({"package": {"$contains": package_type.upper()}})

    if temp_min:
        conditions.append({"temp_min_c": {"$lte": temp_min}})

    if temp_max:
        conditions.append({"temp_max_c": {"$gte": temp_max}})

    # Combine conditions
    if not conditions:
        return "Please specify at least one filter criterion."

    filters = {"$and": conditions} if len(conditions) > 1 else conditions[0]

    # Query with filters
    results = tools.collection.query(
        query_texts=["microcontroller specifications"],  # Generic query
        n_results=max_results * 3,  # Get more for filtering
        where=filters
    )

    if not results['ids'] or not results['ids'][0]:
        return f"No parts found matching the specified criteria."

    # Deduplicate by part number
    seen_parts = set()
    unique_results = []

    for i in range(len(results['ids'][0])):
        meta = results['metadatas'][0][i]
        part_nums = meta.get('part_numbers', '')

        if part_nums and part_nums not in seen_parts:
            seen_parts.add(part_nums)
            unique_results.append(meta)

            if len(unique_results) >= max_results:
                break

    # Filter by peripherals if specified
    if required_peripherals:
        filtered_results = []
        for meta in unique_results:
            features = (meta.get('key_features', '') + ' ' +
                       meta.get('Peripherals', '')).lower()

            has_all = all(periph.lower() in features for periph in required_peripherals)
            if has_all:
                filtered_results.append(meta)

        unique_results = filtered_results

    if not unique_results:
        return f"No parts found matching all criteria (including peripherals: {required_peripherals})"

    # Format output
    output = [f"## Found {len(unique_results)} parts matching criteria\n\n"]

    # Show search criteria
    criteria = []
    if min_flash_kb:
        criteria.append(f"Flash ≥ {min_flash_kb}KB")
    if min_ram_kb:
        criteria.append(f"RAM ≥ {min_ram_kb}KB")
    if min_freq_mhz:
        criteria.append(f"Frequency ≥ {min_freq_mhz}MHz")
    if max_price:
        criteria.append(f"Price ≤ ${max_price}")
    if package_type:
        criteria.append(f"Package: {package_type}")
    if required_peripherals:
        criteria.append(f"Peripherals: {', '.join(required_peripherals)}")
    if architecture:
        criteria.append(f"Architecture: {architecture}")

    if criteria:
        output.append(f"**Criteria:** {', '.join(criteria)}\n\n")

    for i, meta in enumerate(unique_results, 1):
        part_num = meta.get('part_numbers', 'Unknown')
        flash = meta.get('flash_kb_max', 'N/A')
        ram = meta.get('ram_kb_max', 'N/A')
        freq = meta.get('core_freq_mhz', 'N/A')
        price = meta.get('price_usd', 'N/A')
        package = meta.get('package', 'N/A')
        arch = meta.get('architecture', 'N/A')

        datasheet_link = (
            meta.get('pdf_datasheet_url') or
            meta.get('html_datasheet_url') or
            meta.get('datasheet_link', '')
        )

        link_text = f" - 📄 [Datasheet]({datasheet_link})" if datasheet_link else ""

        output.append(f"{i}. **{part_num}**{link_text}\n")
        output.append(f"   - Flash: {flash}KB | RAM: {ram}KB | Frequency: {freq}MHz\n")
        output.append(f"   - Package: {package} | Architecture: {arch}\n")

        if price and price != 'N/A':
            output.append(f"   - Price: ${price}\n")

        features = meta.get('key_features', '')
        if features:
            feature_list = [f.strip() for f in features.split(',')[:3] if f.strip()]
            if feature_list:
                output.append(f"   - Features: {', '.join(feature_list)}\n")

        output.append("\n")

    return "".join(output)


def find_pin_compatible_tool(part_number: str, allow_better_specs: bool = True) -> str:
    """
    Find pin-compatible alternatives (drop-in replacements).

    Args:
        part_number: Reference part number
        allow_better_specs: If True, show parts with equal or better specs

    Returns:
        List of pin-compatible alternatives
    """
    tools = SearchTools()

    # Get reference part
    ref_chunks = tools.get_by_part_number(part_number)
    if not ref_chunks:
        return f"Part {part_number} not found in database."

    ref_meta = ref_chunks[0]['metadata']
    ref_package = ref_meta.get('package', '')
    ref_pin_count = ref_meta.get('pin_count', 0)

    if not ref_package:
        return f"Package information not available for {part_number}"

    # Search for parts with same package and pin count
    filters = {
        "$and": [
            {"package": ref_package},
            {"pin_count": ref_pin_count}
        ]
    }

    results = tools.collection.query(
        query_texts=[f"pin compatible with {part_number}"],
        n_results=20,
        where=filters
    )

    if not results['ids'] or not results['ids'][0]:
        return f"No pin-compatible alternatives found for {part_number} ({ref_package}, {ref_pin_count} pins)"

    # Filter and sort alternatives
    alternatives = []
    ref_flash = ref_meta.get('flash_kb_max', 0)
    ref_ram = ref_meta.get('ram_kb_max', 0)
    ref_freq = ref_meta.get('core_freq_mhz', 0)

    for i in range(len(results['ids'][0])):
        meta = results['metadatas'][0][i]
        alt_part = meta.get('part_numbers', '')

        # Skip the reference part itself
        if alt_part == part_number:
            continue

        alt_flash = meta.get('flash_kb_max', 0)
        alt_ram = meta.get('ram_kb_max', 0)
        alt_freq = meta.get('core_freq_mhz', 0)

        # Check if specs are acceptable
        if allow_better_specs:
            # Allow equal or better specs
            if alt_flash >= ref_flash and alt_ram >= ref_ram and alt_freq >= ref_freq:
                alternatives.append({
                    'part_number': alt_part,
                    'flash': alt_flash,
                    'ram': alt_ram,
                    'freq': alt_freq,
                    'price': meta.get('price_usd', 'N/A'),
                    'metadata': meta
                })
        else:
            # Exact match only
            if alt_flash == ref_flash and alt_ram == ref_ram and alt_freq == ref_freq:
                alternatives.append({
                    'part_number': alt_part,
                    'flash': alt_flash,
                    'ram': alt_ram,
                    'freq': alt_freq,
                    'price': meta.get('price_usd', 'N/A'),
                    'metadata': meta
                })

    if not alternatives:
        return f"No pin-compatible alternatives found for {part_number} with {'equal or better' if allow_better_specs else 'exact'} specs"

    # Format output
    output = []
    output.append(f"## Pin-Compatible Alternatives for {part_number}\n\n")
    output.append(f"**Reference:** {ref_package}, {ref_pin_count} pins\n")
    output.append(f"**Specs:** {ref_flash}KB flash, {ref_ram}KB RAM, {ref_freq}MHz\n\n")
    output.append(f"**Found {len(alternatives)} pin-compatible alternatives:**\n\n")

    for i, alt in enumerate(alternatives, 1):
        alt_meta = alt['metadata']
        datasheet_link = (
            alt_meta.get('pdf_datasheet_url') or
            alt_meta.get('html_datasheet_url') or
            alt_meta.get('datasheet_link', '')
        )

        link_text = f" - 📄 [Datasheet]({datasheet_link})" if datasheet_link else ""

        output.append(f"{i}. **{alt['part_number']}**{link_text}\n")
        output.append(f"   - Specs: {alt['flash']}KB flash, {alt['ram']}KB RAM, {alt['freq']}MHz\n")

        if alt['price'] != 'N/A':
            output.append(f"   - Price: ${alt['price']}\n")

        # Show what's better
        improvements = []
        if alt['flash'] > ref_flash:
            improvements.append(f"+{alt['flash'] - ref_flash}KB flash")
        if alt['ram'] > ref_ram:
            improvements.append(f"+{alt['ram'] - ref_ram}KB RAM")
        if alt['freq'] > ref_freq:
            improvements.append(f"+{alt['freq'] - ref_freq}MHz")

        if improvements:
            output.append(f"   - Improvements: {', '.join(improvements)}\n")

        output.append("\n")

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
        battery_capacity_mah: Battery capacity in mAh
        run_time_pct: Percentage of time in active mode (0-100)
        sleep_time_pct: Percentage of time in sleep mode (0-100)
        active_freq_mhz: Active frequency in MHz (uses max if not specified)

    Returns:
        Battery life estimate with power consumption breakdown
    """
    tools = SearchTools()

    # Validate percentages
    if run_time_pct + sleep_time_pct != 100:
        return f"Error: run_time_pct ({run_time_pct}) + sleep_time_pct ({sleep_time_pct}) must equal 100"

    # Get part info
    chunks = tools.get_by_part_number(part_number)
    if not chunks:
        return f"Part {part_number} not found in database."

    meta = chunks[0]['metadata']

    # Extract power consumption data from features or electrical characteristics
    # Look for typical current values in the text
    all_text = ' '.join([chunk['document'] for chunk in chunks[:5]])  # Check first 5 chunks

    # Try to extract power consumption values (this is simplified)
    # In reality, would need better parsing of electrical characteristics

    # Use some typical values if not found
    # These are estimates - real values should come from datasheets
    max_freq = meta.get('core_freq_mhz', 48)
    freq_to_use = active_freq_mhz if active_freq_mhz else max_freq

    # Rough estimates (would be better to parse from datasheet)
    # Active current scales roughly with frequency
    active_current_ma = (freq_to_use / 48) * 10  # Assume ~10mA at 48MHz
    sleep_current_ua = 5  # Assume ~5µA in sleep mode
    sleep_current_ma = sleep_current_ua / 1000

    # Calculate average current
    run_fraction = run_time_pct / 100
    sleep_fraction = sleep_time_pct / 100

    avg_current_ma = (active_current_ma * run_fraction) + (sleep_current_ma * sleep_fraction)

    # Calculate battery life
    battery_life_hours = battery_capacity_mah / avg_current_ma
    battery_life_days = battery_life_hours / 24
    battery_life_years = battery_life_days / 365

    # Format output
    output = []
    output.append(f"## Battery Life Estimate: {part_number}\n\n")

    # Battery info
    output.append(f"**Battery Configuration:**\n")
    output.append(f"- Capacity: {battery_capacity_mah}mAh\n")

    # Common battery types for reference
    battery_types = {
        240: "CR2032 coin cell",
        600: "CR123A",
        1000: "AAA alkaline",
        2500: "AA alkaline",
        3000: "18650 Li-ion"
    }

    closest_battery = min(battery_types.keys(), key=lambda x: abs(x - battery_capacity_mah))
    if abs(closest_battery - battery_capacity_mah) < 200:
        output.append(f"- Similar to: {battery_types[closest_battery]}\n")

    output.append("\n")

    # Operating profile
    output.append(f"**Operating Profile:**\n")
    output.append(f"- Active mode: {run_time_pct}% @ {freq_to_use}MHz\n")
    output.append(f"- Sleep mode: {sleep_time_pct}%\n\n")

    # Power consumption
    output.append(f"**Power Consumption (Estimates):**\n")
    output.append(f"- Active current: ~{active_current_ma:.2f}mA @ {freq_to_use}MHz\n")
    output.append(f"- Sleep current: ~{sleep_current_ua}µA ({sleep_current_ma:.3f}mA)\n")
    output.append(f"- Average current: ~{avg_current_ma:.3f}mA\n\n")

    # Battery life
    output.append(f"**⏱️ Estimated Battery Life:**\n")
    output.append(f"- **{battery_life_hours:.1f} hours**\n")
    output.append(f"- **{battery_life_days:.1f} days**\n")

    if battery_life_years >= 1:
        output.append(f"- **{battery_life_years:.1f} years** 🎉\n")

    output.append("\n")

    # Optimization tips
    output.append(f"**💡 Optimization Tips:**\n")
    if run_time_pct > 50:
        output.append(f"- Reduce active time: Currently {run_time_pct}% active\n")
    if freq_to_use > 8:
        output.append(f"- Lower frequency when possible: Currently {freq_to_use}MHz\n")
    output.append(f"- Use lowest power sleep modes available\n")
    output.append(f"- Disable unused peripherals\n")

    output.append(f"\n**⚠️ Note:** Power consumption estimates are approximate. ")
    output.append(f"Consult datasheet for actual values and conduct real-world testing.\n")

    return "".join(output)


def find_cheaper_alternative_tool(
    part_number: str,
    must_have_features: List[str] = None,
    max_price_reduction_pct: float = 50
) -> str:
    """
    Find cheaper alternatives to a given part.

    Args:
        part_number: Reference part number
        must_have_features: Features that must be preserved
        max_price_reduction_pct: Maximum acceptable price reduction

    Returns:
        List of cheaper alternatives with analysis
    """
    tools = SearchTools()

    # Get reference part
    ref_chunks = tools.get_by_part_number(part_number)
    if not ref_chunks:
        return f"Part {part_number} not found in database."

    ref_meta = ref_chunks[0]['metadata']
    ref_price_str = ref_meta.get('price_usd', '')

    try:
        ref_price = float(ref_price_str) if ref_price_str else None
    except (ValueError, TypeError):
        ref_price = None

    if not ref_price:
        return f"Price information not available for {part_number}"

    # Get reference specs
    ref_flash = ref_meta.get('flash_kb_max', 0)
    ref_ram = ref_meta.get('ram_kb_max', 0)
    ref_freq = ref_meta.get('core_freq_mhz', 0)
    ref_arch = ref_meta.get('architecture', '')
    ref_device_type = ref_meta.get('device_type', '')

    # Search for similar parts
    query_text = f"{ref_device_type} {ref_arch} microcontroller"

    results = tools.collection.query(
        query_texts=[query_text],
        n_results=50,
        where={"device_type": ref_device_type} if ref_device_type else None
    )

    if not results['ids'] or not results['ids'][0]:
        return f"No alternatives found for {part_number}"

    # Find cheaper alternatives
    alternatives = []

    for i in range(len(results['ids'][0])):
        meta = results['metadatas'][0][i]
        alt_part = meta.get('part_numbers', '')

        # Skip reference part
        if alt_part == part_number:
            continue

        alt_price_str = meta.get('price_usd', '')
        try:
            alt_price = float(alt_price_str) if alt_price_str else None
        except (ValueError, TypeError):
            continue

        if not alt_price or alt_price >= ref_price:
            continue

        # Check if it's cheaper
        price_reduction_pct = ((ref_price - alt_price) / ref_price) * 100

        if price_reduction_pct > max_price_reduction_pct:
            continue  # Too cheap, probably missing features

        alt_flash = meta.get('flash_kb_max', 0)
        alt_ram = meta.get('ram_kb_max', 0)
        alt_freq = meta.get('core_freq_mhz', 0)

        # Check must-have features
        if must_have_features:
            features_text = (meta.get('key_features', '') + ' ' +
                           meta.get('Peripherals', '')).lower()
            has_all = all(feat.lower() in features_text for feat in must_have_features)
            if not has_all:
                continue

        # Calculate benefits and tradeoffs
        benefits = []
        tradeoffs = []

        if alt_flash == ref_flash:
            benefits.append(f"Same flash ({alt_flash}KB)")
        elif alt_flash > ref_flash:
            benefits.append(f"More flash (+{alt_flash - ref_flash}KB)")
        else:
            tradeoffs.append(f"Less flash (-{ref_flash - alt_flash}KB)")

        if alt_ram == ref_ram:
            benefits.append(f"Same RAM ({alt_ram}KB)")
        elif alt_ram > ref_ram:
            benefits.append(f"More RAM (+{alt_ram - ref_ram}KB)")
        else:
            tradeoffs.append(f"Less RAM (-{ref_ram - alt_ram}KB)")

        if alt_freq == ref_freq:
            benefits.append(f"Same frequency ({alt_freq}MHz)")
        elif alt_freq > ref_freq:
            benefits.append(f"Higher frequency (+{alt_freq - ref_freq}MHz)")
        else:
            tradeoffs.append(f"Lower frequency (-{ref_freq - alt_freq}MHz)")

        alternatives.append({
            'part_number': alt_part,
            'price': alt_price,
            'savings': ref_price - alt_price,
            'savings_pct': price_reduction_pct,
            'flash': alt_flash,
            'ram': alt_ram,
            'freq': alt_freq,
            'benefits': benefits,
            'tradeoffs': tradeoffs,
            'datasheet_link': (
                meta.get('pdf_datasheet_url') or
                meta.get('html_datasheet_url') or
                meta.get('datasheet_link', '')
            )
        })

    if not alternatives:
        return f"No cheaper alternatives found for {part_number} with the specified constraints"

    # Sort by price (cheapest first)
    sorted_alternatives = sorted(alternatives, key=lambda x: x['price'])

    # Limit to top 5
    sorted_alternatives = sorted_alternatives[:5]

    # Format output
    output = []
    output.append(f"## Cheaper Alternatives for {part_number}\n\n")

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
        output.append(f"### ✅ ACTIVE ({len(active_parts)})\n\n")
        for r in active_parts:
            link = f" - 📄 [Datasheet]({r['datasheet_link']})" if r['datasheet_link'] else ""
            output.append(f"- **{r['part_number']}**{link}\n")
        output.append("\n")

    if preview_parts:
        output.append(f"### 🆕 PREVIEW ({len(preview_parts)})\n\n")
        for r in preview_parts:
            link = f" - 📄 [Datasheet]({r['datasheet_link']})" if r['datasheet_link'] else ""
            output.append(f"- **{r['part_number']}**{link} (New product, check availability)\n")
        output.append("\n")

    if nrnd_parts:
        output.append(f"### ⚠️ NRND - Not Recommended for New Designs ({len(nrnd_parts)})\n\n")
        for r in nrnd_parts:
            link = f" - 📄 [Datasheet]({r['datasheet_link']})" if r['datasheet_link'] else ""
            output.append(f"- **{r['part_number']}**{link} (Being phased out - find alternative)\n")
        output.append("\n")

    if other_parts:
        output.append(f"### ℹ️ Other Status ({len(other_parts)})\n\n")
        for r in other_parts:
            link = f" - 📄 [Datasheet]({r['datasheet_link']})" if r['datasheet_link'] else ""
            output.append(f"- **{r['part_number']}** - Status: {r['status']}{link}\n")
        output.append("\n")

    if unknown_parts:
        output.append(f"### ❓ Unknown/Not Found ({len(unknown_parts)})\n\n")
        for r in unknown_parts:
            output.append(f"- **{r['part_number']}** - {r.get('error', 'Status unknown')}\n")
        output.append("\n")

    # Add recommendations
    if nrnd_parts:
        output.append(f"**⚠️ Recommendation:** Replace NRND parts with ACTIVE alternatives to avoid future supply issues.\n")

    return "".join(output)


def create_competitor_kill_sheet_tool(
    competitor_part: str,
    competitor_specs: Dict[str, Any] = None,
    use_case: str = None
) -> str:
    """
    Create a competitive analysis showing TI advantages over competitor parts.

    Args:
        competitor_part: Competitor part number (e.g., "STM32L476")
        competitor_specs: Optional specs dict (architecture, freq_mhz, flash_kb, ram_kb, price)
        use_case: Optional application description

    Returns:
        Detailed competitive kill sheet with TI recommendations
    """
    tools = SearchTools()

    output = []
    output.append(f"# Competitive Analysis: {competitor_part} vs TI Solutions\n\n")

    if use_case:
        output.append(f"**Application:** {use_case}\n\n")

    # Competitor info
    output.append(f"## Competitor: {competitor_part}\n\n")

    if competitor_specs:
        output.append(f"**Specifications:**\n")
        if 'architecture' in competitor_specs:
            output.append(f"- Architecture: {competitor_specs['architecture']}\n")
        if 'freq_mhz' in competitor_specs:
            output.append(f"- Frequency: {competitor_specs['freq_mhz']}MHz\n")
        if 'flash_kb' in competitor_specs:
            output.append(f"- Flash: {competitor_specs['flash_kb']}KB\n")
        if 'ram_kb' in competitor_specs:
            output.append(f"- RAM: {competitor_specs['ram_kb']}KB\n")
        if 'price' in competitor_specs:
            output.append(f"- Price: ${competitor_specs['price']}\n")
        output.append("\n")

    # Find matching TI parts
    search_query = f"microcontroller"
    if competitor_specs:
        if 'architecture' in competitor_specs:
            search_query += f" {competitor_specs['architecture']}"
        search_query += " low power"

    results = tools.collection.query(
        query_texts=[search_query],
        n_results=10
    )

    if not results['ids'] or not results['ids'][0]:
        return output + ["No TI alternatives found in database.\n"]

    # Find best TI match
    ti_candidates = []
    comp_flash = competitor_specs.get('flash_kb', 0) if competitor_specs else 0
    comp_ram = competitor_specs.get('ram_kb', 0) if competitor_specs else 0
    comp_freq = competitor_specs.get('freq_mhz', 0) if competitor_specs else 0

    for i in range(len(results['ids'][0])):
        meta = results['metadatas'][0][i]

        flash = meta.get('flash_kb_max', 0)
        ram = meta.get('ram_kb_max', 0)
        freq = meta.get('core_freq_mhz', 0)
        price_str = meta.get('price_usd', '')

        try:
            price = float(price_str) if price_str else None
        except:
            price = None

        # Score based on how well it matches
        score = 0
        if flash >= comp_flash * 0.8:  # At least 80% of flash
            score += 1
        if ram >= comp_ram * 0.8:
            score += 1
        if freq >= comp_freq * 0.8:
            score += 1

        if score >= 2:  # Must match at least 2 criteria
            ti_candidates.append({
                'part_number': meta.get('part_numbers', ''),
                'flash': flash,
                'ram': ram,
                'freq': freq,
                'price': price,
                'architecture': meta.get('architecture', ''),
                'metadata': meta,
                'score': score
            })

    # Sort by score and price
    ti_candidates.sort(key=lambda x: (-x['score'], x['price'] if x['price'] else 999))

    if not ti_candidates:
        output.append("## No comparable TI alternatives found\n\n")
        return "".join(output)

    # Show top TI recommendation
    best_ti = ti_candidates[0]

    output.append(f"## ✅ Recommended TI Alternative: {best_ti['part_number']}\n\n")

    datasheet_link = (
        best_ti['metadata'].get('pdf_datasheet_url') or
        best_ti['metadata'].get('html_datasheet_url') or
        best_ti['metadata'].get('datasheet_link', '')
    )

    if datasheet_link:
        output.append(f"📄 [Datasheet]({datasheet_link})\n\n")

    # Comparison table
    output.append(f"## Specification Comparison\n\n")
    output.append(f"| Feature | {competitor_part} | {best_ti['part_number']} | Advantage |\n")
    output.append(f"|---------|-------------------|----------------------|----------|\n")

    # Architecture
    comp_arch = competitor_specs.get('architecture', 'N/A') if competitor_specs else 'N/A'
    ti_arch = best_ti['architecture']
    output.append(f"| Architecture | {comp_arch} | {ti_arch} | - |\n")

    # Flash
    if competitor_specs and 'flash_kb' in competitor_specs:
        comp_flash_kb = competitor_specs['flash_kb']
        ti_flash_kb = best_ti['flash']
        adv = "✅ TI" if ti_flash_kb >= comp_flash_kb else "Competitor"
        output.append(f"| Flash Memory | {comp_flash_kb}KB | {ti_flash_kb}KB | {adv} |\n")

    # RAM
    if competitor_specs and 'ram_kb' in competitor_specs:
        comp_ram_kb = competitor_specs['ram_kb']
        ti_ram_kb = best_ti['ram']
        adv = "✅ TI" if ti_ram_kb >= comp_ram_kb else "Competitor"
        output.append(f"| SRAM | {comp_ram_kb}KB | {ti_ram_kb}KB | {adv} |\n")

    # Frequency
    if competitor_specs and 'freq_mhz' in competitor_specs:
        comp_freq_mhz = competitor_specs['freq_mhz']
        ti_freq_mhz = best_ti['freq']
        adv = "✅ TI" if ti_freq_mhz >= comp_freq_mhz else "Competitor"
        output.append(f"| Frequency | {comp_freq_mhz}MHz | {ti_freq_mhz}MHz | {adv} |\n")

    # Price
    if competitor_specs and 'price' in competitor_specs and best_ti['price']:
        comp_price = competitor_specs['price']
        ti_price = best_ti['price']
        adv = "✅ TI" if ti_price <= comp_price else "Competitor"
        savings = comp_price - ti_price
        output.append(f"| Price (1K qty) | ${comp_price:.2f} | ${ti_price:.2f} | {adv} |\n")

    output.append("\n")

    # TI Advantages
    output.append(f"## 🎯 Why Choose TI?\n\n")
    output.append(f"### Development Advantages\n")
    output.append(f"1. **Unified Toolchain** - Code Composer Studio (CCS) supports entire MSPM0 portfolio\n")
    output.append(f"2. **DriverLib & SDK** - Comprehensive software libraries accelerate development\n")
    output.append(f"3. **Single Vendor Support** - One FAE, one support portal, faster issue resolution\n")
    output.append(f"4. **Reference Designs** - Extensive application notes and reference designs\n")
    output.append(f"5. **Long-Term Support** - TI's commitment to 10+ year product lifecycles\n\n")

    output.append(f"### Cost Advantages\n")
    if best_ti['price'] and competitor_specs and 'price' in competitor_specs:
        if best_ti['price'] < competitor_specs['price']:
            savings = competitor_specs['price'] - best_ti['price']
            output.append(f"1. **Lower Unit Cost** - Save ${savings:.2f} per unit\n")
            output.append(f"   - 10K units: ${savings * 10000:,.2f} savings\n")
            output.append(f"   - 100K units: ${savings * 100000:,.2f} savings\n")
        else:
            output.append(f"1. **Competitive Pricing** - Similar cost with better ecosystem\n")

    output.append(f"2. **Reduced NRE** - Faster time to market with better tools\n")
    output.append(f"3. **Lower Support Costs** - Single vendor, unified documentation\n")
    output.append(f"4. **Future-Proofing** - Easy migration within TI portfolio\n\n")

    # Show other TI options
    if len(ti_candidates) > 1:
        output.append(f"## Alternative TI Options\n\n")
        for i, alt in enumerate(ti_candidates[1:4], 1):  # Show up to 3 more
            alt_meta = alt['metadata']
            alt_link = (
                alt_meta.get('pdf_datasheet_url') or
                alt_meta.get('html_datasheet_url') or
                ''
            )

            link_text = f" - 📄 [Datasheet]({alt_link})" if alt_link else ""

            output.append(f"{i}. **{alt['part_number']}**{link_text}\n")
            output.append(f"   - {alt['flash']}KB flash, {alt['ram']}KB RAM, {alt['freq']}MHz\n")
            if alt['price']:
                output.append(f"   - Price: ${alt['price']:.2f}\n")
            output.append("\n")

    return "".join(output)


def synthesize_use_case_solution_tool(
    use_case: str,
    constraints: Dict[str, Any] = None
) -> str:
    """
    Generate a comprehensive solution narrative for a use case.

    Args:
        use_case: Description of the application
        constraints: Optional constraints (budget_usd, battery_life_years, etc.)

    Returns:
        Complete solution narrative with architecture, reasoning, benefits
    """
    tools = SearchTools()

    output = []
    output.append(f"# Solution Architecture: {use_case}\n\n")

    # Show constraints if provided
    if constraints:
        output.append(f"## Requirements\n\n")
        if 'budget_usd' in constraints:
            output.append(f"- Budget: ${constraints['budget_usd']}\n")
        if 'battery_life_years' in constraints:
            output.append(f"- Battery Life: {constraints['battery_life_years']} years\n")
        if 'must_have_features' in constraints:
            output.append(f"- Must-Have Features: {', '.join(constraints['must_have_features'])}\n")
        if 'temperature_range' in constraints:
            output.append(f"- Temperature Range: {constraints['temperature_range']}\n")
        if 'size_constraint' in constraints:
            output.append(f"- Size: {constraints['size_constraint']}\n")
        output.append("\n")

    # Search for relevant components
    results = tools.collection.query(
        query_texts=[use_case],
        n_results=15
    )

    if not results['ids'] or not results['ids'][0]:
        return "No relevant TI components found for this use case.\n"

    # Categorize components
    mcus = []
    wireless = []
    power = []
    interface = []

    for i in range(len(results['ids'][0])):
        meta = results['metadatas'][0][i]
        device_type = meta.get('device_type', '')

        if 'Microcontroller' in device_type:
            mcus.append(meta)
        elif 'Wireless' in device_type:
            wireless.append(meta)
        elif 'Power' in device_type:
            power.append(meta)
        elif 'Interface' in device_type:
            interface.append(meta)

    # Build solution narrative
    output.append(f"## Executive Summary\n\n")
    output.append(f"This solution addresses the core requirements of {use_case.lower()} ")
    output.append(f"using Texas Instruments' portfolio of ultra-low-power components. ")
    output.append(f"The recommended architecture balances power efficiency, cost, and development complexity.\n\n")

    # Recommended components
    output.append(f"## Recommended Components\n\n")

    if mcus:
        best_mcu = mcus[0]
        mcu_part = best_mcu.get('part_numbers', '')
        mcu_link = best_mcu.get('pdf_datasheet_url') or best_mcu.get('html_datasheet_url', '')

        link_text = f" - 📄 [Datasheet]({mcu_link})" if mcu_link else ""

        output.append(f"### Microcontroller: {mcu_part}{link_text}\n\n")
        output.append(f"**Why this MCU:**\n")
        output.append(f"- Ultra-low-power consumption ideal for battery operation\n")
        output.append(f"- {best_mcu.get('flash_kb_max', 'N/A')}KB flash, {best_mcu.get('ram_kb_max', 'N/A')}KB RAM\n")
        output.append(f"- {best_mcu.get('architecture', 'N/A')} architecture for efficient code execution\n")

        price = best_mcu.get('price_usd', '')
        if price:
            output.append(f"- Cost-effective at ${price} (1K qty)\n")

        output.append("\n")

    if wireless:
        best_wireless = wireless[0]
        wireless_part = best_wireless.get('part_numbers', '')
        wireless_link = best_wireless.get('pdf_datasheet_url') or best_wireless.get('html_datasheet_url', '')

        link_text = f" - 📄 [Datasheet]({wireless_link})" if wireless_link else ""

        output.append(f"### Wireless: {wireless_part}{link_text}\n\n")
        output.append(f"**Connectivity:**\n")
        features = best_wireless.get('key_features', '')
        if features:
            for feat in features.split(',')[:3]:
                output.append(f"- {feat.strip()}\n")
        output.append("\n")

    if power:
        best_power = power[0]
        power_part = best_power.get('part_numbers', '')
        power_link = best_power.get('pdf_datasheet_url') or best_power.get('html_datasheet_url', '')

        link_text = f" - 📄 [Datasheet]({power_link})" if power_link else ""

        output.append(f"### Power Management: {power_part}{link_text}\n\n")
        output.append(f"**Power Features:**\n")
        output.append(f"- Efficient power conversion for extended battery life\n")
        output.append(f"- Low quiescent current in sleep modes\n\n")

    # Benefits
    output.append(f"## Key Benefits\n\n")
    output.append(f"### 1. Single-Vendor Advantage\n")
    output.append(f"- **Unified Development Tools** - Code Composer Studio supports entire solution\n")
    output.append(f"- **Integrated Software Stack** - Pre-tested DriverLib and SDK components\n")
    output.append(f"- **Single Point of Support** - One FAE team, faster issue resolution\n\n")

    output.append(f"### 2. Cost Efficiency\n")
    output.append(f"- **Reduced BOM** - Fewer external components needed\n")
    output.append(f"- **Lower NRE Costs** - Faster development with better tools\n")
    output.append(f"- **Competitive Pricing** - TI's scale advantages\n\n")

    output.append(f"### 3. Power Optimization\n")
    output.append(f"- **Industry-Leading Standby Current** - Measured in nanoamps\n")
    output.append(f"- **Efficient Active Modes** - Ultra-low mA/MHz performance\n")
    output.append(f"- **Flexible Power Modes** - Multiple sleep states for optimization\n\n")

    # Implementation guidance
    output.append(f"## Implementation Guide\n\n")
    output.append(f"### Step 1: Hardware Design\n")
    output.append(f"- Review TI reference designs for similar applications\n")
    output.append(f"- Utilize TI's WEBENCH Power Designer for power supply design\n")
    output.append(f"- Follow layout guidelines from datasheets\n\n")

    output.append(f"### Step 2: Software Development\n")
    output.append(f"- Install Code Composer Studio (CCS) IDE\n")
    output.append(f"- Download SDK and DriverLib for your components\n")
    output.append(f"- Start with example projects and customize\n\n")

    output.append(f"### Step 3: Testing & Optimization\n")
    output.append(f"- Use EnergyTrace for power consumption analysis\n")
    output.append(f"- Optimize sleep modes and clock frequencies\n")
    output.append(f"- Validate battery life estimates\n\n")

    # Alternatives
    if len(mcus) > 1:
        output.append(f"## Alternative Considerations\n\n")
        for i, alt_mcu in enumerate(mcus[1:3], 1):
            alt_part = alt_mcu.get('part_numbers', '')
            alt_link = alt_mcu.get('pdf_datasheet_url') or alt_mcu.get('html_datasheet_url', '')

            link_text = f" - 📄 [Datasheet]({alt_link})" if alt_link else ""

            output.append(f"**Alternative {i}: {alt_part}**{link_text}\n")

            # Why consider this alternative
            if alt_mcu.get('flash_kb_max', 0) > mcus[0].get('flash_kb_max', 0):
                output.append(f"- Consider if you need more flash ({alt_mcu.get('flash_kb_max')}KB)\n")
            elif alt_mcu.get('flash_kb_max', 0) < mcus[0].get('flash_kb_max', 0):
                output.append(f"- Consider for lower cost with less flash ({alt_mcu.get('flash_kb_max')}KB)\n")

            output.append("\n")

    return "".join(output)


def system_cost_analysis_tool(
    system_description: str,
    production_volume: int = 10000,
    competitor_solution: Dict[str, Any] = None
) -> str:
    """
    Comprehensive system-level cost analysis for all-TI solution.

    Analyzes complete TI solution (MCU + wireless + power + interface) and justifies
    cost and development advantages over mixed-vendor or competitor solutions.

    Args:
        system_description: Description of the system/application
        production_volume: Expected annual production volume
        competitor_solution: Optional dict with competitor parts and pricing
            Example: {
                "mcu": {"part": "STM32L476", "price": 3.50},
                "wireless": {"part": "nRF52840", "price": 2.80},
                "power": {"part": "TPS62xxx", "price": 0.65}
            }

    Returns:
        Comprehensive cost analysis with BOM breakdown, TCO, and justification
    """
    tools = SearchTools()

    output = []
    output.append(f"# System-Level Cost Analysis: All-TI Solution\n\n")
    output.append(f"**Application:** {system_description}\n")
    output.append(f"**Production Volume:** {production_volume:,} units/year\n\n")

    # Search for complete TI solution
    results = tools.collection.query(
        query_texts=[system_description],
        n_results=20
    )

    if not results['ids'] or not results['ids'][0]:
        return "No TI components found for this application.\n"

    # Categorize and select best components
    ti_solution = {
        'mcu': None,
        'wireless': None,
        'power': None,
        'interface': None
    }

    for i in range(len(results['ids'][0])):
        meta = results['metadatas'][0][i]
        device_type = meta.get('device_type', '')
        part = meta.get('part_numbers', '')

        price_str = meta.get('price_usd', '')
        try:
            price = float(price_str) if price_str else 0
        except:
            price = 0

        component_info = {
            'part': part,
            'price': price,
            'metadata': meta
        }

        if 'Microcontroller' in device_type and not ti_solution['mcu']:
            ti_solution['mcu'] = component_info
        elif 'Wireless' in device_type and not ti_solution['wireless']:
            ti_solution['wireless'] = component_info
        elif 'Power' in device_type and not ti_solution['power']:
            ti_solution['power'] = component_info
        elif 'Interface' in device_type and not ti_solution['interface']:
            ti_solution['interface'] = component_info

    # Calculate TI BOM cost
    output.append(f"## ✅ Recommended All-TI Solution\n\n")
    output.append(f"| Component | Part Number | Unit Price | Datasheet |\n")
    output.append(f"|-----------|-------------|------------|----------|\n")

    ti_total_bom = 0
    component_count = 0

    for comp_type, comp_info in ti_solution.items():
        if comp_info:
            component_count += 1
            part = comp_info['part']
            price = comp_info['price']
            ti_total_bom += price

            datasheet = (
                comp_info['metadata'].get('pdf_datasheet_url') or
                comp_info['metadata'].get('html_datasheet_url') or
                ''
            )

            link = f"[📄]({datasheet})" if datasheet else "-"

            output.append(f"| {comp_type.title()} | {part} | ${price:.2f} | {link} |\n")

    output.append(f"| **Total BOM** | **{component_count} components** | **${ti_total_bom:.2f}** | |\n\n")

    # Compare with competitor if provided
    if competitor_solution:
        output.append(f"## 🔄 Competitor/Mixed-Vendor Solution\n\n")
        output.append(f"| Component | Part Number | Unit Price |\n")
        output.append(f"|-----------|-------------|------------|\n")

        comp_total_bom = 0
        comp_component_count = 0

        for comp_type, comp_info in competitor_solution.items():
            if comp_info and 'part' in comp_info and 'price' in comp_info:
                comp_component_count += 1
                part = comp_info['part']
                price = comp_info['price']
                comp_total_bom += price

                output.append(f"| {comp_type.title()} | {part} | ${price:.2f} |\n")

        output.append(f"| **Total BOM** | **{comp_component_count} components** | **${comp_total_bom:.2f}** | |\n\n")

        # BOM Cost Comparison
        bom_diff = ti_total_bom - comp_total_bom
        bom_diff_pct = (bom_diff / comp_total_bom * 100) if comp_total_bom > 0 else 0

        output.append(f"### 💰 BOM Cost Comparison\n\n")

        if bom_diff < 0:
            output.append(f"**TI Solution is ${abs(bom_diff):.2f} ({abs(bom_diff_pct):.1f}%) cheaper per unit!**\n\n")
        elif bom_diff > 0:
            output.append(f"*Note: TI BOM is ${bom_diff:.2f} higher, BUT read TCO analysis below...*\n\n")
        else:
            output.append(f"**BOM costs are equivalent**\n\n")

        # Volume pricing impact
        output.append(f"**At {production_volume:,} units/year:**\n")
        output.append(f"- TI Solution: ${ti_total_bom * production_volume:,.2f}\n")
        output.append(f"- Competitor: ${comp_total_bom * production_volume:,.2f}\n")

        if bom_diff < 0:
            annual_savings = abs(bom_diff) * production_volume
            output.append(f"- **Annual BOM Savings: ${annual_savings:,.2f}** ✅\n\n")
        else:
            annual_diff = bom_diff * production_volume
            output.append(f"- BOM Difference: +${annual_diff:,.2f}\n\n")

    # Total Cost of Ownership (TCO) Analysis
    output.append(f"## 📊 Total Cost of Ownership (TCO) Analysis\n\n")
    output.append(f"### Development Costs (NRE)\n\n")

    # Estimate NRE costs
    output.append(f"| Cost Category | Mixed-Vendor | All-TI | TI Advantage |\n")
    output.append(f"|--------------|--------------|--------|---------------|\n")

    # Development time (in engineer-months)
    mixed_dev_time = 6  # months
    ti_dev_time = 4     # months (faster with unified tools)
    engineer_cost = 15000  # $/month

    mixed_dev_cost = mixed_dev_time * engineer_cost
    ti_dev_cost = ti_dev_time * engineer_cost
    dev_savings = mixed_dev_cost - ti_dev_cost

    output.append(f"| Development Time | {mixed_dev_time} months (${mixed_dev_cost:,}) | {ti_dev_time} months (${ti_dev_cost:,}) | **-${dev_savings:,}** |\n")

    # Support costs (annual)
    mixed_support = 20000  # Multiple vendors, complex support
    ti_support = 10000     # Single vendor
    support_savings = mixed_support - ti_support

    output.append(f"| Annual Support | ${mixed_support:,} | ${ti_support:,} | **-${support_savings:,}/year** |\n")

    # Testing & validation
    mixed_testing = 30000
    ti_testing = 20000
    testing_savings = mixed_testing - ti_testing

    output.append(f"| Testing & Validation | ${mixed_testing:,} | ${ti_testing:,} | **-${testing_savings:,}** |\n")

    # Training costs
    mixed_training = 15000  # Multiple tool chains
    ti_training = 8000      # Single CCS environment
    training_savings = mixed_training - ti_training

    output.append(f"| Training Costs | ${mixed_training:,} | ${ti_training:,} | **-${training_savings:,}** |\n\n")

    # Total NRE
    total_mixed_nre = mixed_dev_cost + mixed_testing + mixed_training
    total_ti_nre = ti_dev_cost + ti_testing + ti_training
    total_nre_savings = total_mixed_nre - total_ti_nre

    output.append(f"**Total NRE Savings with TI: ${total_nre_savings:,}**\n\n")

    # 3-Year TCO
    output.append(f"### 🎯 3-Year Total Cost of Ownership\n\n")

    years = 3

    # Mixed vendor TCO
    mixed_bom_3yr = (comp_total_bom * production_volume * years) if competitor_solution else 0
    mixed_support_3yr = mixed_support * years
    mixed_tco = total_mixed_nre + mixed_bom_3yr + mixed_support_3yr

    # TI TCO
    ti_bom_3yr = ti_total_bom * production_volume * years
    ti_support_3yr = ti_support * years
    ti_tco = total_ti_nre + ti_bom_3yr + ti_support_3yr

    tco_savings = mixed_tco - ti_tco
    tco_savings_pct = (tco_savings / mixed_tco * 100) if mixed_tco > 0 else 0

    if competitor_solution:
        output.append(f"| Cost Component | Mixed-Vendor | All-TI |\n")
        output.append(f"|----------------|--------------|--------|\n")
        output.append(f"| NRE (one-time) | ${total_mixed_nre:,} | ${total_ti_nre:,} |\n")
        output.append(f"| BOM (3 years @ {production_volume:,}/yr) | ${mixed_bom_3yr:,} | ${ti_bom_3yr:,} |\n")
        output.append(f"| Support (3 years) | ${mixed_support_3yr:,} | ${ti_support_3yr:,} |\n")
        output.append(f"| **Total 3-Year TCO** | **${mixed_tco:,}** | **${ti_tco:,}** |\n\n")

        if tco_savings > 0:
            output.append(f"### 🎉 **Total Savings with All-TI Solution: ${tco_savings:,} ({tco_savings_pct:.1f}%)**\n\n")
        else:
            output.append(f"### 3-Year TCO Difference: ${abs(tco_savings):,}\n\n")

    # Qualitative Benefits
    output.append(f"## 🌟 Strategic Advantages of All-TI Solution\n\n")

    output.append(f"### 1. Development Velocity\n")
    output.append(f"- **Unified IDE**: Code Composer Studio for all components\n")
    output.append(f"- **Integrated Stack**: Pre-tested drivers and middleware\n")
    output.append(f"- **Single Debug Environment**: One debugger, one workflow\n")
    output.append(f"- **Faster Time-to-Market**: 30-40% reduction in development time\n\n")

    output.append(f"### 2. Support & Reliability\n")
    output.append(f"- **Single Point of Contact**: One FAE team for all components\n")
    output.append(f"- **Faster Issue Resolution**: No finger-pointing between vendors\n")
    output.append(f"- **Unified Documentation**: Consistent format and quality\n")
    output.append(f"- **E2E Testing**: TI validates component interactions\n\n")

    output.append(f"### 3. Supply Chain Benefits\n")
    output.append(f"- **Simplified Procurement**: Single vendor relationship\n")
    output.append(f"- **Better Pricing Power**: Volume leverage across components\n")
    output.append(f"- **Reduced Risk**: No multi-vendor allocation games\n")
    output.append(f"- **Long-Term Availability**: TI's 10+ year commitment\n\n")

    output.append(f"### 4. Future-Proofing\n")
    output.append(f"- **Easy Upgrades**: Pin-compatible within families\n")
    output.append(f"- **Code Reuse**: Same tools/APIs across portfolio\n")
    output.append(f"- **Ecosystem Growth**: TI continuously expands portfolio\n")
    output.append(f"- **Long-Term Roadmap**: Clear migration path\n\n")

    # Risk Analysis
    output.append(f"## ⚠️ Risk Mitigation\n\n")
    output.append(f"### Risks of Mixed-Vendor Approach:\n")
    output.append(f"1. **Integration Complexity**: Components not designed to work together\n")
    output.append(f"2. **Support Gaps**: Each vendor only supports their part\n")
    output.append(f"3. **Tool Chain Hell**: Learning and maintaining multiple IDEs\n")
    output.append(f"4. **Supply Chain Fragility**: Multiple points of failure\n")
    output.append(f"5. **Hidden Costs**: Integration bugs, delayed launches, field issues\n\n")

    output.append(f"### All-TI Solution Mitigates:\n")
    output.append(f"- ✅ Pre-validated component combinations\n")
    output.append(f"- ✅ Unified support and warranty\n")
    output.append(f"- ✅ Single toolchain reduces training\n")
    output.append(f"- ✅ Simplified supply chain\n")
    output.append(f"- ✅ Faster issue resolution\n\n")

    # Summary recommendation
    output.append(f"## 📋 Executive Summary\n\n")

    if competitor_solution and tco_savings > 0:
        output.append(f"**Recommendation: All-TI Solution**\n\n")
        output.append(f"The all-TI solution delivers:\n")
        output.append(f"- **${tco_savings:,} savings** over 3 years ({tco_savings_pct:.1f}%)\n")
        output.append(f"- **{mixed_dev_time - ti_dev_time} months faster** time-to-market\n")
        output.append(f"- **Lower risk** with single-vendor support\n")
        output.append(f"- **Better long-term** positioning for product evolution\n\n")
    else:
        output.append(f"**Recommendation: All-TI Solution**\n\n")
        output.append(f"Even with comparable BOM costs, the all-TI solution offers:\n")
        output.append(f"- **${total_nre_savings:,} NRE savings**\n")
        output.append(f"- **Faster development** with unified tools\n")
        output.append(f"- **Lower operational costs** (support, training, maintenance)\n")
        output.append(f"- **Reduced risk** and better long-term support\n\n")

    output.append(f"*For detailed component specifications and reference designs, ")
    output.append(f"consult individual datasheets and TI's design resources.*\n")

    return "".join(output)
