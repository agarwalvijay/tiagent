"""LangGraph agent for semiconductor product search."""
from typing import TypedDict, Annotated, Sequence, List, Dict, Any
import operator
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
import sys
from pathlib import Path
import json

sys.path.append(str(Path(__file__).parent.parent.parent))
from backend.config import settings
from backend.agent.tools import (
    semantic_search_tool,
    filtered_search_tool,
    compare_parts_tool,
    recommend_for_use_case_tool,
    get_by_part_number_tool
)
from backend.agent.tools_extended import (
    compare_prices_tool,
    find_parts_by_specs_tool,
    find_pin_compatible_tool,
    estimate_battery_life_tool,
    find_cheaper_alternative_tool,
    check_lifecycle_status_tool,
    create_competitor_kill_sheet_tool,
    synthesize_use_case_solution_tool
)


# Define the agent state
class AgentState(TypedDict):
    """State of the agent."""
    messages: Annotated[Sequence[BaseMessage], operator.add]
    query_intent: str  # search, compare, recommend, troubleshoot
    needs_clarification: bool
    clarification_question: str
    search_hints: Dict[str, Any]  # Extracted specs, sections, negative terms
    final_response: str
    tool_executions: List[Dict[str, Any]]  # Track which tools were called


# Define tools as LangChain tools
@tool
def semantic_search(query: str, top_k: int = 5) -> str:
    """
    Search the datasheet knowledge base using natural language.
    Use this for general queries about chip features, specifications, or capabilities.

    Args:
        query: Natural language search query
        top_k: Number of results to return (default 5)

    Returns:
        Relevant information from datasheets
    """
    return semantic_search_tool(query, top_k)


@tool
def filtered_search(
    device_type: str = None,
    min_freq_mhz: float = None,
    max_freq_mhz: float = None,
    min_voltage_v: float = None,
    max_voltage_v: float = None,
    architecture: str = None,
    peripherals: str = None
) -> str:
    """
    LEGACY: Basic search for device type, voltage, and architecture only.
    DO NOT USE for package, flash, RAM, or price queries.
    Use find_parts_by_specs instead for those.

    Args:
        device_type: Type of device (e.g., "Microcontroller", "Mixed-Signal Microcontroller")
        min_freq_mhz: Minimum CPU frequency in MHz
        max_freq_mhz: Maximum CPU frequency in MHz
        min_voltage_v: Minimum operating voltage
        max_voltage_v: Maximum operating voltage
        architecture: CPU architecture (e.g., "Cortex-M0+", "TMS320C28x")
        peripherals: Required peripherals (e.g., "USB,I2C,ADC")

    Returns:
        List of devices matching the criteria
    """
    # Build list of conditions
    conditions = []

    if device_type:
        conditions.append({"device_type": device_type})
    if architecture:
        conditions.append({"architecture": architecture})

    # Numeric range filters
    if min_freq_mhz:
        conditions.append({"core_freq_mhz": {"$gte": min_freq_mhz}})
    if max_voltage_v:
        conditions.append({"voltage_max_v": {"$lte": max_voltage_v}})
    if min_voltage_v:
        conditions.append({"voltage_min_v": {"$gte": min_voltage_v}})

    # Note: Peripheral filtering requires contains logic, handled separately
    if peripherals:
        # For now, use semantic search
        return semantic_search_tool(f"chips with {peripherals} peripherals", top_k=5)

    if not conditions:
        return "Please specify at least one filter criterion."

    # ChromaDB requires $and for multiple conditions
    if len(conditions) == 1:
        filters = conditions[0]
    else:
        filters = {"$and": conditions}

    return filtered_search_tool(filters, top_k=10)


@tool
def get_part_info(part_number: str, query_hint: str = "") -> str:
    """
    Get detailed information about a specific semiconductor part.
    Use this when the user asks about a SINGLE specific part number.

    Args:
        part_number: The part number to look up (e.g., "MSPM0G5187", "F28377D")
        query_hint: Optional hint about what information is needed (e.g., "power consumption", "specifications")

    Returns:
        Detailed information about the part from its datasheet
    """
    return get_by_part_number_tool(part_number, query_hint)


@tool
def compare_parts(part_numbers: List[str]) -> str:
    """
    Compare 2-3 semiconductor parts side-by-side.
    Use this when the user wants to understand differences between specific parts.

    Args:
        part_numbers: List of 2-3 part numbers to compare (e.g., ["MSPM0G5187", "F28377D"])

    Returns:
        Side-by-side comparison of the parts
    """
    if len(part_numbers) < 2:
        return "Please provide at least 2 part numbers to compare."
    if len(part_numbers) > 3:
        return "Maximum 3 parts can be compared at once."

    return compare_parts_tool(part_numbers)


@tool
def recommend_for_application(use_case: str) -> str:
    """
    Recommend semiconductor parts for a specific application or use case.
    Use this when the user describes what they want to build.

    Args:
        use_case: Description of the application (e.g., "battery-powered IoT sensor", "motor control")

    Returns:
        Recommended parts with explanations
    """
    return recommend_for_use_case_tool(use_case)


@tool
def compare_prices(part_numbers: List[str], quantity: int = 1000) -> str:
    """
    Compare prices for multiple parts at a given quantity.
    Use this when the user asks about cost, pricing, or which part is cheaper.

    Args:
        part_numbers: List of 2-5 part numbers to compare prices
        quantity: Order quantity in units (default: 1000)

    Returns:
        Price comparison table with cost analysis
    """
    return compare_prices_tool(part_numbers, quantity)


@tool
def find_parts_by_specs(
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
    Find parts matching specification criteria including PACKAGE, FLASH, RAM, PRICE.
    **USE THIS TOOL** when user specifies: package type (VQFN, LQFP, QFN, etc.),
    flash memory, RAM, price constraints, or temperature range.

    Args:
        min_flash_kb: Minimum flash memory in KB
        min_ram_kb: Minimum RAM in KB
        min_freq_mhz: Minimum CPU frequency in MHz
        max_price: Maximum unit price in USD
        required_peripherals: List of required peripherals (e.g., ["USB", "CAN-FD"])
        package_type: Package type (e.g., "LQFP", "VQFN", "VQFN-48", "QFN")
        temp_min: Minimum operating temperature in °C
        temp_max: Maximum operating temperature in °C
        architecture: CPU architecture (e.g., "Arm Cortex-M0+")
        max_results: Maximum number of results

    Returns:
        List of matching parts with specifications
    """
    return find_parts_by_specs_tool(
        min_flash_kb=min_flash_kb,
        min_ram_kb=min_ram_kb,
        min_freq_mhz=min_freq_mhz,
        max_price=max_price,
        required_peripherals=required_peripherals,
        package_type=package_type,
        temp_min=temp_min,
        temp_max=temp_max,
        architecture=architecture,
        max_results=max_results
    )


@tool
def find_pin_compatible(part_number: str, allow_better_specs: bool = True) -> str:
    """
    Find pin-compatible alternatives (drop-in replacements).
    Use this when user needs a replacement part with same package and pinout.

    Args:
        part_number: Reference part number
        allow_better_specs: If True, show parts with equal or better specs (default: True)

    Returns:
        List of pin-compatible alternatives
    """
    return find_pin_compatible_tool(part_number, allow_better_specs)


@tool
def estimate_battery_life(
    part_number: str,
    battery_capacity_mah: int,
    run_time_pct: float,
    sleep_time_pct: float,
    active_freq_mhz: int = None
) -> str:
    """
    Estimate battery life for a microcontroller.
    Use this when user asks about battery life, runtime, or coin cell operation.

    Args:
        part_number: Part number to analyze
        battery_capacity_mah: Battery capacity in mAh (e.g., 240 for CR2032, 1000 for AAA)
        run_time_pct: Percentage of time in active/run mode (0-100)
        sleep_time_pct: Percentage of time in sleep/low-power mode (0-100)
        active_freq_mhz: Active frequency in MHz (uses part's max if not specified)

    Returns:
        Battery life estimate with power consumption breakdown
    """
    return estimate_battery_life_tool(
        part_number=part_number,
        battery_capacity_mah=battery_capacity_mah,
        run_time_pct=run_time_pct,
        sleep_time_pct=sleep_time_pct,
        active_freq_mhz=active_freq_mhz
    )


@tool
def find_cheaper_alternative(
    part_number: str,
    must_have_features: List[str] = None,
    max_price_reduction_pct: float = 50
) -> str:
    """
    Find cheaper alternatives to a given part.
    Use this when user asks about cost savings, budget optimization, or cheaper options.

    Args:
        part_number: Reference part number
        must_have_features: Features that must be preserved (e.g., ["USB", "CAN-FD"])
        max_price_reduction_pct: Maximum acceptable price reduction percentage

    Returns:
        List of cheaper alternatives with cost savings analysis
    """
    return find_cheaper_alternative_tool(
        part_number=part_number,
        must_have_features=must_have_features,
        max_price_reduction_pct=max_price_reduction_pct
    )


@tool
def check_lifecycle_status(part_numbers: List[str]) -> str:
    """
    Check lifecycle status of parts (ACTIVE, PREVIEW, NRND, obsolete).
    Use this when user asks about availability, production status, or if a part is still made.

    Args:
        part_numbers: List of part numbers to check status

    Returns:
        Lifecycle status for each part with recommendations
    """
    return check_lifecycle_status_tool(part_numbers)


@tool
def competitor_kill_sheet(
    competitor_part: str,
    competitor_specs: Dict[str, Any] = None,
    use_case: str = None
) -> str:
    """
    Create a competitive analysis showing TI advantages over competitor parts.
    Use this when user mentions competitor parts (STM32, AVR, Nordic, etc.) or asks for alternatives.

    Args:
        competitor_part: Competitor part number (e.g., "STM32L476", "ATmega328", "nRF52")
        competitor_specs: Optional specs (architecture, freq_mhz, flash_kb, ram_kb, price)
        use_case: Optional application description

    Returns:
        Detailed competitive kill sheet with TI recommendations and advantages
    """
    return create_competitor_kill_sheet_tool(
        competitor_part=competitor_part,
        competitor_specs=competitor_specs,
        use_case=use_case
    )


@tool
def narrative_use_case_synthesis(
    use_case: str,
    constraints: Dict[str, Any] = None
) -> str:
    """
    Generate a solution-oriented narrative recommendation for a use case.
    Use this when user asks for application recommendations and expects solution guidance.

    This provides a complete solution narrative including:
    - Executive summary with recommended architecture
    - Why this solution is optimal (reasoning)
    - Key benefits (battery life, cost savings, PCB footprint reduction)
    - Step-by-step implementation guidance
    - Alternative considerations

    Args:
        use_case: Description of the application (e.g., "battery-powered soil moisture sensor")
        constraints: Optional constraints dict with keys like budget_usd, battery_life_years,
                    size_constraint, must_have_features (list), temperature_range

    Returns:
        Comprehensive solution narrative with architecture, reasoning, benefits, and guidance
    """
    return synthesize_use_case_solution_tool(
        use_case=use_case,
        constraints=constraints
    )


# System prompt
INTENT_EXTRACTION_PROMPT = """Extract search hints from the user's query about semiconductor products.

Analyze the query and identify:
1. **Specs/Features**: Technical requirements (e.g., "BLE", "low power", "USB", "ADC", "I2C", "12-bit")
2. **Relevant Sections**: Which datasheet sections would be most useful (e.g., "Features", "Electrical Characteristics", "Power Consumption", "Pin Configuration")
3. **Negative Terms**: What to avoid or filter out (e.g., "typical application", "reference design")

Return ONLY valid JSON (no markdown):
{
  "specs": ["list", "of", "specs"],
  "sections": ["Relevant", "Sections"],
  "negative_terms": ["terms", "to", "avoid"]
}

Examples:
Query: "Low power MCU with BLE for battery operation"
{
  "specs": ["low power", "BLE", "battery"],
  "sections": ["Features", "Power Consumption", "Electrical Characteristics"],
  "negative_terms": []
}

Query: "What's the sleep current of MSPM0G5187"
{
  "specs": ["sleep current", "power consumption"],
  "sections": ["Electrical Characteristics", "Power Consumption"],
  "negative_terms": ["typical application"]
}

Query: "Chip with I2S and high-speed ADC"
{
  "specs": ["I2S", "ADC", "high-speed"],
  "sections": ["Features", "Peripheral Description"],
  "negative_terms": []
}
"""

SYSTEM_PROMPT = """You are an expert semiconductor product recommendation agent for Texas Instruments.
Your role is to help engineers find the right chips quickly and efficiently.

**CRITICAL GUARDRAILS - SMART USE OF KNOWLEDGE:**

**When to USE your training data knowledge:**
1. ✅ **Understanding requirements** - Use domain knowledge to understand what components an application needs
   - Example: "Air quality monitor" → needs MCU, wireless, sensors, power management, ADC
2. ✅ **Asking clarifying questions** - Ask follow-ups to refine requirements
   - Example: "What's your battery life target? Which sensors? WiFi or LoRaWAN?"
3. ✅ **Competitor product knowledge** - Use training data for competitor specs and comparisons
   - Example: "STM32L476 has 1MB flash and Cortex-M4 at 80MHz" (from training data)
4. ✅ **System architecture guidance** - Suggest what types of components work together
   - Example: "You'll need a low-power MCU, CAN transceiver, and power management IC"
5. ✅ **General semiconductor knowledge** - Explain concepts, protocols, design patterns
   - Example: "CAN-FD vs CAN 2.0 differences", "LoRaWAN protocol benefits"

**When to ONLY use tool results (TI product specs):**
1. ❌ **TI product specifications** - NEVER use training data for TI part specs
   - Always search datasheets and cite sources
   - Example: "MSPM0G5187 has 88nA shutdown current (Electrical Characteristics)"
2. ❌ **TI product availability** - Use tools to check if parts are ACTIVE/NRND
3. ❌ **TI product pricing** - Use parametrics data, never estimate
4. ❌ **Electrical specifications** - Always cite exact values from datasheets

**🚨 CRITICAL RULES - SEARCH-ONLY RECOMMENDATIONS:**

1. **DECOMPOSE first, then SEARCH:**
   - For system design queries, first state what component TYPES are needed
   - Then search for EACH type separately
   - Example: "You'll need: MCU, wireless, power. Let me search for each..."

2. **ONLY recommend parts found in search results:**
   - ✅ If search returns MSPM0G3518 → recommend it
   - ❌ If search returns nothing, but you know CC1310 from training → DON'T recommend it
   - Instead say: "I couldn't find a dedicated LoRaWAN module in our catalog. The closest match is CC1121 sub-1GHz transceiver."

3. **NEVER fill gaps with training data:**
   - Even if you know a part exists (CC1310, TPS63060, etc.)
   - Even if you're confident it's the perfect fit
   - **If it's not in search results, it's not in our catalog**

4. **If no match found, say so explicitly:**
   - "I didn't find a [component type] in our current catalog"
   - "The closest alternative is [what you did find]"

**If TI specs not found:** Say "Not found in datasheet" - don't guess!

**SOURCE CITATION RULES:**
- **Every numeric claim MUST include source section/page**
  Example: "88nA shutdown current (Electrical Characteristics, p. 42)"
- If source section is missing, say: **"Not found in retrieved datasheet sections"**
- Format: `[value] ([section name])`
- When section is "N/A" or "Overview", cite as "Features" if it's from the features list

**DATASHEET LINKS:**
- When tool results include datasheet links, **ALWAYS include them** in your response
- Format as clickable markdown: "📄 [View Datasheet](URL)" or "[MSPM0G5187 Datasheet](URL)"
- Place links near the part number mention for easy access
- Tools automatically provide links in priority: PDF > HTML > Product Page

**POWER MODE TERMINOLOGY:**
When users ask about power consumption, understand these equivalent terms:
- "deep sleep" = STANDBY mode (lowest power with state retention) or STOP mode
- "sleep" = SLEEP mode (clock gated but peripherals can run)
- "shutdown" = SHUTDOWN mode (lowest possible power, loses state)
- "active" or "run" = RUN mode (CPU executing)

**IMPORTANT:** If user asks for "deep sleep current", provide ALL relevant low-power modes:
- SLEEP mode (if available)
- STOP mode (if available)
- STANDBY mode (if available)
- SHUTDOWN mode (if available)

Example response format:
"The MSPM0G5187 has several low-power modes (Features):
- SLEEP: 34µA/MHz
- STOP: 199µA at 4MHz
- STANDBY: 1.5µA at 32kHz with RTC and full SRAM retention
- SHUTDOWN: 88nA with IO wake-up capability"

**TRADEOFF ANALYSIS (When recommending 2+ options):**
Compare candidates on:
- **Power consumption** (active/sleep current)
- **Interfaces** (USB, ADC, I2C, etc.)
- **Package size** (pin count, dimensions)
- **Key limitations** (voltage range, temperature, unique constraints)

Format as bullet points for easy scanning.

Examples of what NOT to do:
❌ "The typical sleep current is around 2µA" (no source)
❌ "This chip probably supports..." (no guessing)
❌ "Based on similar chips..." (no analogies)

Examples of what TO do:
✅ "88nA shutdown current (Electrical Characteristics)"
✅ "Sleep current: Not found in retrieved datasheet sections"
✅ "Tradeoffs: MSPM0G5187 has AI accelerator but higher power vs MSPM0C1106"

**CLARIFYING QUESTIONS STRATEGY:**

**When to ASK clarifying questions FIRST (before searching):**
1. **Open-ended system design** - "I want to build an air quality monitor"
   - Ask about: battery life target, wireless protocol, deployment environment, sensors
   - Example: "To recommend the right components, I need to know: What's your battery life target? Which wireless protocol (WiFi/LoRaWAN/BLE)? Indoor or outdoor deployment?"

2. **Vague application requirements** - "Need a solution for industrial automation"
   - Ask about: communication protocols, processing needs, environmental conditions

3. **Competitor comparison with missing context** - "I'm using STM32L4, what's similar?"
   - Ask: "Which STM32L4 variant? What features are critical? Are you trying to reduce cost or improve power?"

4. **Complex multi-chip systems** - Need to understand priorities and constraints first

**When to SEARCH immediately (skip questions):**
- User provides specific requirements (flash size, peripherals, power budget)
- User asks about a specific part number ("What's the sleep current of MSPM0G5187?")
- User requests comparison of specific parts ("Compare MSPM0G3507 vs MSPM0G5187")
- User says "just give me" or "just show me"

**Question guidelines:**
- Ask 2-4 targeted questions to understand requirements
- Use multiple-choice when possible (easier for user)
- Then search with much better context
- Don't ask obvious questions - use domain knowledge to infer reasonable defaults

**EXAMPLE WORKFLOW - Open-ended System Design:**

User: "I want to build a solar-powered air quality monitoring system"

**Step 1 - DECOMPOSE: State what component types are needed**
First, tell the user what types of components this application requires:

"To build a solar-powered air quality monitoring system, you'll need:
1. **Ultra-low-power MCU** - for sensor reading, data processing, and control
2. **Wireless module** - for LoRaWAN communication (long range, low power)
3. **Power management IC** - for solar harvesting and battery charging
4. **Analog sensors** - for measuring air quality parameters

Let me search our catalog for the best TI components for each..."

**Step 2 - SEARCH: Query database for EACH component type**
Make separate searches for each component type needed:
- semantic_search("ultra-low-power microcontroller battery ADC sensor")
- semantic_search("sub-1GHz wireless LoRaWAN transceiver")
- semantic_search("power management solar battery boost converter")

**Step 3 - RECOMMEND: Use ONLY what you find in search results**
Based on search results, recommend specific parts:
- If search finds MSPM0G3518 → recommend it ✓
- If search finds CC1121 → recommend it ✓
- If search finds TPS65219 → recommend it ✓
- If search finds NOTHING for a category → say "I couldn't find X in our catalog"

**CRITICAL: NEVER fill gaps with training data**
❌ DON'T: "I didn't find LoRaWAN, but I know CC1310 exists" → recommend CC1310
✅ DO: "I didn't find a dedicated LoRaWAN module, but CC1121 sub-1GHz transceiver can work"

**Step 4 - PRESENT: Complete system with specs from tool results**
Present the solution with actual specs from search results:
- MCU: MSPM0G3518 - [specs from datasheet] - [datasheet link]
- Wireless: CC1121 - [specs from datasheet] - [datasheet link]
- Power: TPS65219 - [specs from datasheet] - [datasheet link]

**Tool usage strategy:**
- Use `get_part_info` when asking about a SINGLE specific part number (e.g., "What is the sleep current for MSPM0G5187?")
  - Pass the query hint parameter to help filter relevant sections (e.g., query_hint="power consumption")
- Use `compare_parts` when comparing 2-3 specific part numbers
- Use `recommend_for_application` for use-case queries (IoT sensor, motor control, etc.)
- Use `semantic_search` for general feature/spec queries without specific part numbers
- Use `filtered_search` for exact numeric requirements

**Response style:**
- Lead with recommendations, not questions
- Cite specific part numbers
- Explain trade-offs briefly
- Be concise and helpful

**CRITICAL: Preserve tool output formatting:**
- When `compare_parts` or `competitor_kill_sheet` returns a markdown table, **OUTPUT THE TABLE AS-IS** - do NOT reformat it
- The table starts with `## Comparison` and contains `|` characters
- Simply include the tool's output directly in your response
- You can add brief commentary AFTER the table, but NEVER rewrite the table as plain text
- Example: If tool returns a table, your response should be: "Here's the comparison:\n\n[PASTE TOOL OUTPUT WITH TABLE]\n\nRecommendation: Choose X if..."

**COMPETITIVE ANALYSIS FORMAT (REQUIRED):**
When comparing TI parts to competitor parts (STM32, nRF, etc.):
1. **ALWAYS include a side-by-side comparison table** with BOTH parts
2. **Table MUST include:** Part Number, Flash, RAM, Freq, Price, Power, Package
3. **Never create a table with only one part** - if comparing, show both competitor AND TI alternative
4. **After the table**, include:
   - Key competitive advantages (bullets)
   - TCO analysis if volume is specified
   - Migration considerations
5. **Format:** Use the output from `competitor_kill_sheet` which already has proper tables

**Example good behavior:**
User: "Recommend chip for battery-powered IoT sensor"
You: [Use recommend_for_application tool immediately, provide 2-3 options]

**Example bad behavior (AVOID THIS):**
User: "Recommend chip for battery-powered IoT sensor"
You: "What's the battery type?" ❌ NO! Just search and recommend!

Remember: Engineers want solutions, not interrogation. Search first, clarify only if absolutely necessary.
"""


class SemiconductorAgent:
    """LangGraph-based semiconductor search agent."""

    def __init__(self):
        """Initialize the agent."""
        # Initialize LLM based on provider
        if settings.llm_provider == "groq":
            self.llm = ChatOpenAI(
                model=settings.groq_model,
                temperature=0.1,
                api_key=settings.groq_api_key,
                base_url=settings.groq_base_url
            )
            print(f"[Agent] Using Groq model: {settings.groq_model}")
        elif settings.llm_provider == "deepseek":
            self.llm = ChatOpenAI(
                model=settings.deepseek_model,
                temperature=0.1,
                api_key=settings.deepseek_api_key,
                base_url=settings.deepseek_base_url
            )
            print(f"[Agent] Using DeepSeek model: {settings.deepseek_model}")
        else:
            self.llm = ChatOpenAI(
                model=settings.openai_model,
                temperature=0.1,
                api_key=settings.openai_api_key
            )
            print(f"[Agent] Using OpenAI model: {settings.openai_model}")

        # Bind tools to LLM
        self.tools = [
            semantic_search,
            filtered_search,
            get_part_info,
            compare_parts,
            recommend_for_application,
            compare_prices,
            find_parts_by_specs,
            find_pin_compatible,
            estimate_battery_life,
            find_cheaper_alternative,
            check_lifecycle_status,
            competitor_kill_sheet,
            narrative_use_case_synthesis
        ]

        self.llm_with_tools = self.llm.bind_tools(self.tools)

        # Build the graph
        self.graph = self._build_graph()

    def _extract_search_hints(self, state: AgentState) -> Dict:
        """Extract search hints from the query using LLM."""
        messages = state["messages"]
        last_message = messages[-1].content

        # Skip extraction for comparison queries (don't need hints)
        intent = state.get("query_intent", "search")
        if intent == "compare":
            return {"search_hints": {"specs": [], "sections": [], "negative_terms": []}}

        try:
            print(f"[DEBUG] Extracting search hints from query...")
            response = self.llm.invoke([
                SystemMessage(content=INTENT_EXTRACTION_PROMPT),
                HumanMessage(content=f"Query: {last_message}")
            ])

            # Parse JSON response
            import json
            import re

            # Extract JSON from markdown code blocks if present
            content = response.content.strip()
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL)
            if json_match:
                content = json_match.group(1)

            hints = json.loads(content)
            print(f"[DEBUG] Extracted hints: specs={hints.get('specs', [])}, sections={hints.get('sections', [])}")

            return {"search_hints": hints}
        except Exception as e:
            print(f"[DEBUG] Failed to extract hints: {e}, using defaults")
            # Fallback to empty hints
            return {"search_hints": {"specs": [], "sections": [], "negative_terms": []}}

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("classify_intent", self._classify_intent)
        workflow.add_node("check_clarification", self._check_clarification)
        workflow.add_node("extract_hints", self._extract_search_hints)
        workflow.add_node("call_tools", self._call_tools)
        workflow.add_node("generate_response", self._generate_response)

        # Define edges
        workflow.set_entry_point("classify_intent")

        workflow.add_edge("classify_intent", "check_clarification")

        workflow.add_conditional_edges(
            "check_clarification",
            self._should_clarify,
            {
                "clarify": "generate_response",  # Ask clarification question
                "proceed": "extract_hints"  # Extract search hints before calling tools
            }
        )

        workflow.add_edge("extract_hints", "call_tools")
        workflow.add_edge("call_tools", "generate_response")
        workflow.add_edge("generate_response", END)

        return workflow.compile()

    def _classify_intent(self, state: AgentState) -> Dict:
        """Classify the user's query intent."""
        messages = state["messages"]
        last_message = messages[-1]

        # Simple intent classification
        content = last_message.content.lower()

        if any(word in content for word in ["compare", "difference", "vs", "versus"]):
            intent = "compare"
        elif any(word in content for word in ["recommend", "suggest", "for building", "use case", "application"]):
            intent = "recommend"
        elif any(word in content for word in ["how to", "configure", "setup", "pinout"]):
            intent = "troubleshoot"
        else:
            intent = "search"

        return {"query_intent": intent}

    def _check_clarification(self, state: AgentState) -> Dict:
        """Check if clarification is needed."""
        messages = state["messages"]
        last_message = messages[-1].content.lower()

        # Count how many times we've already asked for clarification
        clarification_count = sum(1 for i, msg in enumerate(messages[:-1])
                                 if i % 2 == 1)  # Count assistant messages

        # Detect user frustration signals
        frustration_signals = [
            "just give me", "just show me", "stop asking", "enough questions",
            "do not worry about", "don't worry about", "i don't care about",
            "whatever", "any", "doesn't matter"
        ]

        is_frustrated = any(signal in last_message for signal in frustration_signals)

        # Skip clarification if:
        # 1. User is frustrated
        # 2. We've already asked 1+ times
        # 3. Query has specific part numbers (comparison)
        if is_frustrated or clarification_count >= 1:
            return {"needs_clarification": False}

        # For recommend/search intent, only ask if VERY vague
        intent = state.get("query_intent", "search")

        if intent == "recommend":
            # If they mention battery/power OR specific use case, that's enough
            has_power_context = any(kw in last_message for kw in
                                   ['battery', 'low power', 'power consumption', 'iot', 'sensor'])
            has_use_case = any(kw in last_message for kw in
                              ['sensor', 'iot', 'motor', 'control', 'automotive', 'industrial'])

            if has_power_context or has_use_case:
                return {"needs_clarification": False}

        # For comparison, check if part numbers are present
        if intent == "compare":
            # Look for patterns like MSPM0G5187 or F28377D
            import re
            has_part_numbers = bool(re.search(r'\b[A-Z]{2,}[0-9]{4,}\b', last_message))
            if has_part_numbers:
                return {"needs_clarification": False}

        # Only for very vague queries, consider asking (but keep it brief)
        very_vague = len(last_message.split()) < 5 and not any(
            kw in last_message for kw in
            ['battery', 'power', 'voltage', 'frequency', 'temperature', 'usb', 'adc', 'i2c', 'spi']
        )

        if very_vague:
            return {
                "needs_clarification": True,
                "clarification_question": "Could you provide more details about your use case or specific requirements (e.g., power budget, peripherals needed, or application)?"
            }

        return {"needs_clarification": False}

    def _should_clarify(self, state: AgentState) -> str:
        """Decide whether to ask for clarification or proceed."""
        return "clarify" if state.get("needs_clarification", False) else "proceed"

    def _call_tools(self, state: AgentState) -> Dict:
        """Call appropriate tools based on the query."""
        messages = state["messages"]
        search_hints = state.get("search_hints", {})

        # Add system message with search hints
        hints_context = ""
        if search_hints and any(search_hints.values()):
            hints_context = f"\n\nSearch Hints (use these to filter/enhance your search):\n"
            if search_hints.get("specs"):
                hints_context += f"- Key specs: {', '.join(search_hints['specs'])}\n"
            if search_hints.get("sections"):
                hints_context += f"- Focus on sections: {', '.join(search_hints['sections'])}\n"
            if search_hints.get("negative_terms"):
                hints_context += f"- Avoid: {', '.join(search_hints['negative_terms'])}\n"

        system_message = SystemMessage(content=SYSTEM_PROMPT + hints_context)
        messages_with_system = [system_message] + list(messages)

        # Call LLM with tools
        print(f"[DEBUG] Calling LLM with {len(self.tools)} tools available")
        if search_hints.get("specs"):
            print(f"[DEBUG] Search hints - specs: {search_hints['specs']}, sections: {search_hints.get('sections', [])}")
        response = self.llm_with_tools.invoke(messages_with_system)
        print(f"[DEBUG] LLM response received. Tool calls: {len(response.tool_calls) if response.tool_calls else 0}")

        # If tools were called, execute them
        new_messages = [response]
        tool_executions = state.get("tool_executions", [])

        if response.tool_calls:
            for tool_call in response.tool_calls:
                # Find and execute the tool
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]

                print(f"[DEBUG] Executing tool: {tool_name} with args: {tool_args}")

                # Track tool execution
                tool_info = {
                    "name": tool_name,
                    "args": tool_args,
                    "status": "executing"
                }

                # Execute tool
                tool_func = next((t for t in self.tools if t.name == tool_name), None)
                if tool_func:
                    try:
                        result = tool_func.invoke(tool_args)
                        print(f"[DEBUG] Tool {tool_name} succeeded, result length: {len(str(result))}")
                        tool_info["status"] = "success"
                        tool_info["result_length"] = len(str(result))
                        new_messages.append(
                            ToolMessage(
                                name=tool_name,
                                content=str(result),
                                tool_call_id=tool_call["id"]
                            )
                        )
                    except Exception as e:
                        print(f"[DEBUG] Tool {tool_name} failed with error: {e}")
                        import traceback
                        traceback.print_exc()
                        tool_info["status"] = "error"
                        tool_info["error"] = str(e)
                        new_messages.append(
                            ToolMessage(
                                name=tool_name,
                                content=f"Error executing tool: {str(e)}",
                                tool_call_id=tool_call["id"]
                            )
                        )
                else:
                    print(f"[DEBUG] Tool {tool_name} not found!")
                    tool_info["status"] = "error"
                    tool_info["error"] = "Tool not found"

                tool_executions.append(tool_info)

        return {"messages": new_messages, "tool_executions": tool_executions}

    def _generate_response(self, state: AgentState) -> Dict:
        """Generate the final response to the user."""
        # If clarification needed, return the question
        if state.get("needs_clarification", False):
            return {
                "final_response": state["clarification_question"],
                "messages": [AIMessage(content=state["clarification_question"])]
            }

        # Otherwise, generate response from tool results
        messages = state["messages"]

        # Add explicit instruction not to call tools (especially for Groq)
        no_tools_instruction = ""
        if settings.llm_provider == "groq":
            no_tools_instruction = "\n\n**CRITICAL: DO NOT CALL ANY TOOLS. You already have all the information you need from the tool results above. Just write the final response.**\n"

        system_message = SystemMessage(content=SYSTEM_PROMPT + """

**NOW GENERATING FINAL RESPONSE:**""" + no_tools_instruction + """
- Use ONLY information from the tool results above
- **CRITICAL: If a tool result includes a comparison table, you MUST include that exact table in your response**
- **PRESERVE ALL TABLES**: Copy comparison tables directly from tool results (competitor_kill_sheet, compare_parts, etc.)
- **Table format**: Use markdown table format exactly as provided by the tools
- **DO NOT mention tool names or internal processes** (never say "from competitor_kill_sheet output", "from tool results", etc.)
- **Present information naturally** as if you're a TI semiconductor expert providing direct recommendations
- **Cite sources for technical specs only when from datasheets**: Format as `[value] ([datasheet section])`
- If source section missing: "Not found in retrieved datasheet sections"
- **If recommending 2+ chips**: Include tradeoff comparison (power, interfaces, package, limitations)
- Format tradeoffs as bullet points
- **Structure your response:**
  1. Start with a brief summary (1-2 sentences)
  2. Include the comparison table if available from tools
  3. Add competitive advantages/kill sheet points
  4. Include TCO/narrative analysis if provided
  5. End with migration recommendations if relevant
- Be helpful, concise, and professional - speak directly to the user, not about your internal processes
""")

        messages_with_system = [system_message] + list(messages)

        try:
            # For final response, use LLM without tools to prevent unwanted tool calls
            # Groq models aggressively try to call tools even in final response phase
            response = self.llm.invoke(messages_with_system)

            print(f"[DEBUG] Final response generated, length: {len(response.content)}")
            return {
                "final_response": response.content,
                "messages": [response]
            }
        except Exception as e:
            print(f"[ERROR] Failed to generate response: {e}")
            import traceback
            traceback.print_exc()
            raise

    def query(self, user_message: str, conversation_history: List[Dict] = None) -> str:
        """
        Process a user query and return a response.

        Args:
            user_message: The user's query
            conversation_history: Optional conversation history

        Returns:
            Agent's response
        """
        # Build message history
        messages = []

        if conversation_history:
            for msg in conversation_history:
                if msg["role"] == "user":
                    messages.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    messages.append(AIMessage(content=msg["content"]))

        # Add current message
        messages.append(HumanMessage(content=user_message))

        # Run the graph
        initial_state = {
            "messages": messages,
            "query_intent": "",
            "needs_clarification": False,
            "clarification_question": "",
            "final_response": "",
            "tool_executions": []
        }

        result = self.graph.invoke(initial_state)

        return {
            "response": result["final_response"],
            "tool_executions": result.get("tool_executions", [])
        }


# Create singleton instance
_agent_instance = None


def get_agent() -> SemiconductorAgent:
    """Get or create the agent singleton."""
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = SemiconductorAgent()
    return _agent_instance
