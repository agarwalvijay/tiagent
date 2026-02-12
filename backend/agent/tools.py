"""Tools for the LangGraph agent."""
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
import chromadb
from chromadb.config import Settings
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
from backend.config import settings


class SemanticSearchInput(BaseModel):
    """Input for semantic search tool."""
    query: str = Field(description="Natural language query to search for")
    top_k: int = Field(default=5, description="Number of results to return")
    chunk_types: Optional[List[str]] = Field(
        default=None,
        description="Filter by chunk types: overview, features, specs, pins, description"
    )


class FilteredSearchInput(BaseModel):
    """Input for filtered search tool."""
    filters: Dict[str, Any] = Field(
        description="Metadata filters, e.g., {'device_type': 'Microcontroller', 'voltage_min_v': {'$gte': 1.0}}"
    )
    top_k: int = Field(default=10, description="Number of results to return")


class ComparisonInput(BaseModel):
    """Input for comparison tool."""
    part_numbers: List[str] = Field(
        description="List of 2-3 part numbers to compare",
        min_items=2,
        max_items=3
    )


class SearchTools:
    """Tools for searching the datasheet knowledge base."""

    def __init__(self):
        """Initialize search tools."""
        # Initialize ChromaDB with persistent storage
        self.client = chromadb.PersistentClient(
            path=settings.chroma_persist_dir
        )

        self.collection = self.client.get_collection(
            name=settings.chroma_collection_name
        )

        # Initialize embedding model
        self.use_openai_embeddings = settings.embedding_model.startswith("openai/")

        if self.use_openai_embeddings:
            from openai import OpenAI
            self.openai_client = OpenAI(api_key=settings.openai_api_key)
            self.embedding_model_name = settings.embedding_model.replace("openai/", "")
        else:
            from sentence_transformers import SentenceTransformer
            self.embedding_model = SentenceTransformer(settings.embedding_model)

    def semantic_search(self, query: str, top_k: int = 5, chunk_types: Optional[List[str]] = None) -> List[Dict]:
        """
        Perform semantic search over datasheets.

        Args:
            query: Natural language query
            top_k: Number of results to return
            chunk_types: Optional filter by chunk types

        Returns:
            List of relevant chunks with metadata
        """
        # Generate query embedding
        query_embedding = self._generate_embedding(query)

        # Build where filter
        where_filter = None
        if chunk_types:
            where_filter = {"chunk_type": {"$in": chunk_types}}

        # Query ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where_filter,
            include=["documents", "metadatas", "distances"]
        )

        # Format results
        formatted_results = []
        for i in range(len(results["ids"][0])):
            formatted_results.append({
                "chunk_id": results["ids"][0][i],
                "content": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "similarity": 1 - results["distances"][0][i],  # Convert distance to similarity
            })

        return formatted_results

    def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text using configured model."""
        if self.use_openai_embeddings:
            response = self.openai_client.embeddings.create(
                model=self.embedding_model_name,
                input=[text]
            )
            return response.data[0].embedding
        else:
            return self.embedding_model.encode([text], convert_to_numpy=True)[0].tolist()

    def filtered_search(self, filters: Dict[str, Any], top_k: int = 10) -> List[Dict]:
        """
        Search using exact metadata filters.

        Args:
            filters: Metadata filters (ChromaDB where clause)
            top_k: Number of results to return

        Returns:
            List of matching chunks
        """
        # Get matching chunks
        results = self.collection.get(
            where=filters,
            limit=top_k,
            include=["documents", "metadatas"]
        )

        # Format results
        formatted_results = []
        for i in range(len(results["ids"])):
            formatted_results.append({
                "chunk_id": results["ids"][i],
                "content": results["documents"][i],
                "metadata": results["metadatas"][i],
            })

        return formatted_results

    def hybrid_search(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 5,
        chunk_types: Optional[List[str]] = None
    ) -> List[Dict]:
        """
        Perform hybrid search combining semantic and metadata filtering.

        Args:
            query: Natural language query
            filters: Optional metadata filters
            top_k: Number of results
            chunk_types: Optional chunk type filter

        Returns:
            List of relevant chunks
        """
        # Generate query embedding
        query_embedding = self._generate_embedding(query)

        # Build where filter
        where_filter = {}
        if chunk_types:
            where_filter["chunk_type"] = {"$in": chunk_types}
        if filters:
            where_filter.update(filters)

        where_filter = where_filter if where_filter else None

        # Query ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where_filter,
            include=["documents", "metadatas", "distances"]
        )

        # Format results
        formatted_results = []
        for i in range(len(results["ids"][0])):
            formatted_results.append({
                "chunk_id": results["ids"][0][i],
                "content": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "similarity": 1 - results["distances"][0][i],
            })

        return formatted_results

    def get_by_part_number(self, part_number: str) -> List[Dict]:
        """
        Get all chunks for a specific part number.

        Args:
            part_number: Part number to search for

        Returns:
            List of chunks for that part
        """
        # ChromaDB 0.5.x doesn't support $contains, so we get all and filter manually
        all_results = self.collection.get(
            limit=10000,
            include=["metadatas", "documents"]
        )

        matching_ids = []
        for i, metadata in enumerate(all_results["metadatas"]):
            part_nums = metadata.get("part_numbers", "").upper()
            # Check if the part number is in the comma-separated list
            if part_number.upper() in part_nums:
                matching_ids.append(all_results["ids"][i])

        if matching_ids:
            results = self.collection.get(
                ids=matching_ids,
                include=["documents", "metadatas"]
            )
        else:
            results = {"ids": [], "documents": [], "metadatas": []}

        # Format results
        formatted_results = []
        for i in range(len(results["ids"])):
            formatted_results.append({
                "chunk_id": results["ids"][i],
                "content": results["documents"][i],
                "metadata": results["metadatas"][i],
            })

        return formatted_results

    def compare_parts(self, part_numbers: List[str]) -> Dict[str, Any]:
        """
        Compare multiple parts side-by-side.

        Args:
            part_numbers: List of 2-3 part numbers

        Returns:
            Comparison data
        """
        comparison = {
            "parts": {},
            "common_features": [],
            "differences": {}
        }

        for part_num in part_numbers:
            # Get overview chunk for each part
            chunks = self.get_by_part_number(part_num)

            if not chunks:
                comparison["parts"][part_num] = {"error": "Part not found"}
                continue

            # Get the overview chunk
            overview = next(
                (c for c in chunks if c["metadata"].get("chunk_type") == "overview"),
                chunks[0] if chunks else None
            )

            if overview:
                metadata = overview["metadata"]

                # Format flash KB (show single value if min == max)
                flash_min = metadata.get('flash_kb_min', 0)
                flash_max = metadata.get('flash_kb_max', 0)
                if flash_min == flash_max and flash_min > 0:
                    flash_kb = f"{flash_max} KB"
                elif flash_max > 0:
                    flash_kb = f"{flash_min}-{flash_max} KB"
                else:
                    flash_kb = "N/A"

                # Format RAM KB (show single value if min == max)
                ram_min = metadata.get('ram_kb_min', 0)
                ram_max = metadata.get('ram_kb_max', 0)
                if ram_min == ram_max and ram_min > 0:
                    ram_kb = f"{ram_max} KB"
                elif ram_max > 0:
                    ram_kb = f"{ram_min}-{ram_max} KB"
                else:
                    ram_kb = "N/A"

                # Format voltage range (only show if available)
                voltage_min = metadata.get('voltage_min_v', 0)
                voltage_max = metadata.get('voltage_max_v', 0)
                if voltage_min > 0 or voltage_max > 0:
                    voltage_range = f"{voltage_min}-{voltage_max} V"
                else:
                    voltage_range = "N/A"

                # Format temperature range
                temp_min = metadata.get('operating_temp_min_c', -999)
                temp_max = metadata.get('operating_temp_max_c', 999)
                if temp_min != -999 and temp_max != 999:
                    temp_range = f"{temp_min} to {temp_max} °C"
                else:
                    temp_range = "N/A"

                # Get datasheet link (priority: PDF > HTML > Product Page)
                datasheet_link = (
                    metadata.get("pdf_datasheet_url") or
                    metadata.get("html_datasheet_url") or
                    metadata.get("product_page_url") or
                    metadata.get("datasheet_link", "")
                )

                comparison["parts"][part_num] = {
                    "document_id": metadata.get("document_id"),
                    "device_type": metadata.get("device_type"),
                    "architecture": metadata.get("architecture"),
                    "core_freq_mhz": metadata.get("core_freq_mhz"),
                    "flash_kb": flash_kb,
                    "ram_kb": ram_kb,
                    "voltage_range": voltage_range,
                    "temp_range": temp_range,
                    "peripherals": metadata.get("peripherals", "").split(","),
                    "key_features": metadata.get("key_features", "").split(","),
                    "content": overview["content"][:500],  # First 500 chars
                    "datasheet_link": datasheet_link,
                }

        return comparison

    def get_device_recommendations(
        self,
        use_case: str,
        requirements: Optional[Dict[str, Any]] = None
    ) -> List[Dict]:
        """
        Recommend devices for a specific use case.

        Args:
            use_case: Description of what to build
            requirements: Optional specific requirements

        Returns:
            List of recommended devices
        """
        # First, do semantic search on the use case
        semantic_results = self.semantic_search(
            query=use_case,
            top_k=10,
            chunk_types=["overview", "features"]
        )

        # If requirements provided, filter further
        if requirements:
            filtered_results = []
            for result in semantic_results:
                metadata = result["metadata"]
                matches = True

                # Check each requirement
                for key, value in requirements.items():
                    if key == "min_freq_mhz":
                        if metadata.get("core_freq_mhz", 0) < value:
                            matches = False
                    elif key == "min_flash_kb":
                        if metadata.get("flash_kb_max", 0) < value:
                            matches = False
                    elif key == "max_voltage_v":
                        if metadata.get("voltage_min_v", 999) > value:
                            matches = False
                    # Add more requirement checks as needed

                if matches:
                    filtered_results.append(result)

            semantic_results = filtered_results

        return semantic_results[:5]  # Top 5 recommendations


# Helper function to score and rank parts
def score_part(chunks: List[Dict], query_specs: List[str] = None) -> Dict:
    """
    Score a part based on available information quality.

    Scoring formula:
    + Spec match count × 3
    + Constraint satisfied × 5
    + Numeric spec present × 2
    + Spec section present × 2
    - Missing required spec × 4
    - Overview-only evidence × 2
    """
    score = 0
    has_specs = False
    has_overview_only = True
    numeric_specs = 0
    spec_sections = 0

    query_specs = query_specs or []
    matched_specs = 0

    for chunk in chunks:
        chunk_type = chunk['metadata'].get('chunk_type', '')
        content = chunk['content'].lower()

        # Check chunk types
        if chunk_type in ['specs', 'features', 'description']:
            has_specs = True
            has_overview_only = False
            spec_sections += 1

        if chunk_type == 'overview' and not has_specs:
            has_overview_only = True

        # Count numeric specs (e.g., "3.3V", "100MHz", "88nA")
        import re
        numeric_pattern = r'\d+\.?\d*\s*(?:V|MHz|GHz|mA|µA|nA|KB|MB|°C|mW)'
        numeric_specs += len(re.findall(numeric_pattern, content))

        # Check for query spec matches
        for spec in query_specs:
            if spec.lower() in content:
                matched_specs += 1

    # Calculate score
    score += matched_specs * 3  # Spec match
    score += numeric_specs * 2  # Numeric specs present
    score += spec_sections * 2  # Spec sections present

    # Penalties
    if has_overview_only:
        score -= 2  # Overview-only penalty

    # Check for missing critical specs (if query mentioned them but not found)
    if query_specs:
        missing_specs = len(query_specs) - matched_specs
        score -= missing_specs * 4

    return {
        'score': score,
        'matched_specs': matched_specs,
        'numeric_specs': numeric_specs,
        'spec_sections': spec_sections,
        'has_overview_only': has_overview_only
    }


def group_and_rank_results(results: List[Dict], query_specs: List[str] = None, top_k: int = 5) -> List[Dict]:
    """
    Group chunks by part number and rank by quality score.

    Returns top-K parts with their aggregated chunks and scores.
    """
    # Group by part number
    parts = {}
    for result in results:
        part_nums = result['metadata'].get('part_numbers', 'Unknown')
        if part_nums not in parts:
            parts[part_nums] = []
        parts[part_nums].append(result)

    # Score each part
    ranked_parts = []
    for part_num, chunks in parts.items():
        if part_num == 'Unknown' or not part_num:
            continue

        score_info = score_part(chunks, query_specs)

        ranked_parts.append({
            'part_number': part_num,
            'chunks': chunks,
            'score': score_info['score'],
            'score_details': score_info,
            'metadata': chunks[0]['metadata']  # Use first chunk's metadata
        })

    # Sort by score (descending)
    ranked_parts.sort(key=lambda x: x['score'], reverse=True)

    return ranked_parts[:top_k]


# Helper function to filter results based on search hints
def filter_by_hints(results: List[Dict], negative_terms: List[str] = None) -> List[Dict]:
    """Filter results to remove chunks containing negative terms."""
    if not negative_terms:
        return results

    filtered = []
    for result in results:
        content_lower = result['content'].lower()
        # Check if any negative term appears in the content
        has_negative = any(term.lower() in content_lower for term in negative_terms)
        if not has_negative:
            filtered.append(result)

    return filtered


# Tool function wrappers for LangGraph
def semantic_search_tool(query: str, top_k: int = 5, chunk_types: Optional[List[str]] = None, query_specs: List[str] = None) -> str:
    """Search datasheets using natural language."""
    tools = SearchTools()

    # Get more results initially (we'll group and rank them)
    results = tools.semantic_search(query, top_k * 3, chunk_types)

    if not results:
        return "No results found."

    # Group by part and rank by quality score
    ranked_parts = group_and_rank_results(results, query_specs, top_k)

    if not ranked_parts:
        return "No parts found matching the criteria."

    # Format results for LLM
    output = []
    for i, part_data in enumerate(ranked_parts, 1):
        meta = part_data['metadata']
        score_details = part_data['score_details']

        # Aggregate content from all chunks for this part
        all_content = "\n\n".join([chunk['content'] for chunk in part_data['chunks'][:3]])  # Top 3 chunks

        output.append(
            f"Rank {i} (Score: {part_data['score']}, Matched specs: {score_details['matched_specs']}):\n"
            f"Part: {part_data['part_number']}\n"
            f"Type: {meta.get('device_type', 'N/A')}\n"
            f"Quality: {score_details['numeric_specs']} numeric specs, {score_details['spec_sections']} spec sections\n"
            f"Content:\n{all_content[:800]}...\n"
        )

    return "\n---\n".join(output)


def filtered_search_tool(filters: Dict[str, Any], top_k: int = 10) -> str:
    """Search using exact specifications."""
    tools = SearchTools()
    results = tools.filtered_search(filters, top_k)

    if not results:
        return "No devices match the specified filters."

    # Format results
    output = []
    unique_parts = set()

    for result in results:
        part_nums = result["metadata"].get("part_numbers", "")
        if part_nums and part_nums not in unique_parts:
            unique_parts.add(part_nums)

            # Get datasheet link
            meta = result["metadata"]
            datasheet_link = (
                meta.get('pdf_datasheet_url') or
                meta.get('html_datasheet_url') or
                meta.get('product_page_url') or
                meta.get('datasheet_link', '')
            )

            link_text = f"Datasheet: [View]({datasheet_link})\n" if datasheet_link else ""

            output.append(
                f"Part: {part_nums}\n"
                f"Type: {result['metadata'].get('device_type', 'N/A')}\n"
                f"Architecture: {result['metadata'].get('architecture', 'N/A')}\n"
                f"Frequency: {result['metadata'].get('core_freq_mhz', 'N/A')} MHz\n"
                f"Voltage: {result['metadata'].get('voltage_min_v', 'N/A')}-{result['metadata'].get('voltage_max_v', 'N/A')} V\n"
                f"{link_text}"
            )

    return "\n---\n".join(output[:10])  # Limit to 10 unique parts


def get_by_part_number_tool(part_number: str, query_hint: str = "") -> str:
    """Get detailed information about a specific part number."""
    tools = SearchTools()
    results = tools.get_by_part_number(part_number)

    if not results:
        return f"Part {part_number} not found in database."

    # If query_hint provided, filter results to most relevant chunks
    if query_hint:
        query_lower = query_hint.lower()
        # Prioritize chunks matching query keywords
        relevant_results = []
        for result in results:
            content_lower = result['content'].lower()
            # Check for power-related keywords
            if any(kw in query_lower for kw in ['power', 'current', 'sleep', 'standby', 'shutdown', 'consumption']):
                if any(kw in content_lower for kw in ['power', 'current', 'sleep', 'standby', 'shutdown', 'µa', 'ma', 'na']):
                    relevant_results.append(result)
            # Check for spec-related keywords
            elif any(kw in query_lower for kw in ['spec', 'characteristic', 'electrical']):
                if any(kw in content_lower for kw in ['specification', 'characteristic', 'electrical', 'rating']):
                    relevant_results.append(result)
            else:
                relevant_results.append(result)

        if relevant_results:
            results = relevant_results[:5]  # Top 5 relevant chunks
        else:
            results = results[:5]  # Fallback to first 5

    # Format output with datasheet link if available
    output = [f"Information for {part_number}:\n"]

    # Add datasheet link from first result's metadata
    if results:
        meta = results[0]['metadata']
        datasheet_link = (
            meta.get('pdf_datasheet_url') or
            meta.get('html_datasheet_url') or
            meta.get('product_page_url') or
            meta.get('datasheet_link', '')
        )
        if datasheet_link:
            output.append(f"📄 [View Datasheet]({datasheet_link})\n")

    for i, result in enumerate(results[:10], 1):  # Max 10 chunks
        meta = result['metadata']
        section = meta.get('section', 'N/A')

        # Improve section naming for N/A sections
        if section == 'N/A' or not section:
            chunk_type = meta.get('chunk_type', 'overview')
            if chunk_type == 'overview':
                section = 'Features (Overview)'
            else:
                section = 'Overview'

        content = result['content'][:1500]  # Limit content length

        output.append(
            f"\n--- Section: {section} ---\n"
            f"{content}\n"
        )

    return "".join(output)


def compare_parts_tool(part_numbers: List[str]) -> str:
    """Compare multiple parts side-by-side."""
    tools = SearchTools()
    comparison = tools.compare_parts(part_numbers)

    # Check for errors first
    errors = []
    valid_parts = {}
    for part, data in comparison["parts"].items():
        if "error" in data:
            errors.append(f"{part}: {data['error']}")
        else:
            valid_parts[part] = data

    if not valid_parts:
        return "No valid parts found for comparison.\n" + "\n".join(errors)

    # Build markdown table
    output = ["## Comparison\n\n"]

    if errors:
        output.append("**Note:** " + ", ".join(errors) + "\n\n")

    # Create table header
    header = "| Specification | " + " | ".join(valid_parts.keys()) + " |"
    separator = "|" + "|".join(["---"] * (len(valid_parts) + 1)) + "|"

    output.append(header + "\n")
    output.append(separator + "\n")

    # Add datasheet links row first
    output.append("| **Datasheet** |")
    for part_num, part_data in valid_parts.items():
        link = part_data.get('datasheet_link', '')
        if link:
            output.append(f" [View Datasheet]({link}) |")
        else:
            output.append(" — |")
    output.append("\n")

    # Add rows
    rows = [
        ("Type", "device_type"),
        ("Architecture", "architecture"),
        ("Frequency (MHz)", "core_freq_mhz"),
        ("Flash (KB)", "flash_kb"),
        ("RAM (KB)", "ram_kb"),
        ("Voltage Range", "voltage_range"),
        ("Temperature Range", "temp_range"),
    ]

    for row_label, key in rows:
        row = f"| **{row_label}** |"
        for part_data in valid_parts.values():
            value = part_data.get(key, "N/A")
            if value == "N/A" or value is None or value == "":
                value = "—"
            row += f" {value} |"
        output.append(row + "\n")

    # Add peripherals row (truncate if too long)
    output.append("| **Peripherals** |")
    for part_data in valid_parts.values():
        peripherals = part_data.get('peripherals', [])
        periph_str = ', '.join(peripherals[:5])
        if len(peripherals) > 5:
            periph_str += f" (+{len(peripherals) - 5} more)"
        output.append(f" {periph_str or '—'} |")
    output.append("\n")

    # Add features row
    output.append("| **Key Features** |")
    for part_data in valid_parts.values():
        features = part_data.get('key_features', [])
        features_str = ', '.join(features) if features else '—'
        output.append(f" {features_str} |")
    output.append("\n")

    return "".join(output)


def recommend_for_use_case_tool(use_case: str, requirements: Optional[Dict[str, Any]] = None) -> str:
    """Recommend devices for a specific use case."""
    tools = SearchTools()
    results = tools.get_device_recommendations(use_case, requirements)

    if not results:
        return "No suitable devices found for this use case."

    # Extract query specs from use case
    query_specs = []
    use_case_lower = use_case.lower()
    common_specs = ['ble', 'usb', 'adc', 'i2c', 'spi', 'low power', 'battery', 'wifi', 'uart', 'can']
    for spec in common_specs:
        if spec in use_case_lower:
            query_specs.append(spec)

    # Group and rank by quality
    ranked_parts = group_and_rank_results(results, query_specs, top_k=5)

    if not ranked_parts:
        return "No parts found matching the use case."

    output = [f"Recommendations for: {use_case}\n"]

    for i, part_data in enumerate(ranked_parts, 1):
        meta = part_data['metadata']
        score_details = part_data['score_details']

        # Get datasheet link
        datasheet_link = (
            meta.get('pdf_datasheet_url') or
            meta.get('html_datasheet_url') or
            meta.get('product_page_url') or
            meta.get('datasheet_link', '')
        )

        link_text = f"\n   📄 [View Datasheet]({datasheet_link})" if datasheet_link else ""

        output.append(
            f"\n{i}. {part_data['part_number']} (Score: {part_data['score']})\n"
            f"   Type: {meta.get('device_type', 'N/A')}\n"
            f"   Architecture: {meta.get('architecture', 'N/A')}\n"
            f"   Frequency: {meta.get('core_freq_mhz', 'N/A')} MHz\n"
            f"   Voltage: {meta.get('voltage_min_v', 'N/A')}-{meta.get('voltage_max_v', 'N/A')} V\n"
            f"   Key Features: {meta.get('key_features', 'N/A')}\n"
            f"   Match Quality: {score_details['matched_specs']} specs matched, {score_details['numeric_specs']} numeric specs{link_text}\n"
        )

    return "".join(output)
