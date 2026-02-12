# TI Semiconductor Product Finder Agent

An intelligent agent for finding and recommending semiconductor products (chips, SoCs, analog chips, dev boards) based on specifications and use cases.

## 🎯 Features

- **🔍 Semantic Search**: Find products based on natural language descriptions
- **🎯 Specification Filtering**: Exact matching on voltage, frequency, temperature, peripherals
- **🤖 LangGraph Agent**: Intelligent orchestration with proactive clarification questions
- **💬 Multi-turn Conversations**: Maintains context across queries
- **📊 Comparison Mode**: Side-by-side comparison of multiple chips
- **🔄 Alternative Finding**: Find substitutes and alternatives
- **🏗️ Application Recommendations**: Get complete solutions for specific use cases
- **📚 28 TI Datasheets**: Pre-loaded with real Texas Instruments datasheets

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────┐
│                   User Interface                     │
│              React + TypeScript Chat UI              │
└────────────────────┬────────────────────────────────┘
                     │ REST API
                     ▼
┌─────────────────────────────────────────────────────┐
│                 FastAPI Backend                      │
│  ┌──────────────────────────────────────────────┐   │
│  │          LangGraph Agent (GPT-4o)            │   │
│  │  ┌────────────────────────────────────────┐  │   │
│  │  │  Tools:                                │  │   │
│  │  │  - Semantic Search                     │  │   │
│  │  │  - Filtered Search                     │  │   │
│  │  │  - Compare Parts                       │  │   │
│  │  │  - Recommend for Use Case              │  │   │
│  │  └────────────────────────────────────────┘  │   │
│  └──────────────────────────────────────────────┘   │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
         ┌───────────────────────┐
         │  ChromaDB (Vector DB)  │
         │  - 500+ chunks         │
         │  - Metadata filtering  │
         │  - Semantic embeddings │
         └───────────────────────┘
```

## 📦 Tech Stack

**Backend:**
- Python 3.11
- FastAPI (REST API)
- LangGraph (Agent orchestration)
- OpenAI GPT-4o (LLM)
- ChromaDB (Vector database)
- sentence-transformers (Embeddings)
- PyMuPDF + pdfplumber (PDF parsing)

**Frontend:**
- React 18 + TypeScript
- React Markdown (Message rendering)
- Axios (API client)

## 🚀 Quick Start

See **[QUICKSTART.md](QUICKSTART.md)** for detailed setup instructions.

### TL;DR

```bash
# 1. Add OpenAI API key to .env
echo "OPENAI_API_KEY=sk-your-key" > .env

# 2. Run setup (10-15 minutes)
./setup.sh

# 3. Start backend (terminal 1)
./run_backend.sh

# 4. Start frontend (terminal 2)
./run_frontend.sh

# 5. Open http://localhost:3000
```

## 📖 Example Queries

**Find by Specifications:**
```
"Find a 32-bit MCU with USB and ADC under 3.3V"
"I need a low-power chip with I2C and SPI that works at -40°C"
"Which chips have AI accelerators?"
```

**Compare Parts:**
```
"Compare MSPM0G5187 with MSPM0C1106"
"What's the difference between F28377D-SEP and MSPM0G5187?"
```

**Use Case Recommendations:**
```
"What chips would work for a battery-powered IoT sensor?"
"Recommend an MCU for motor control in automotive applications"
"Best chip for industrial automation at high temperatures"
```

**Technical Details:**
```
"What are the features of MSPM0G5187?"
"How do I configure I2C on the MSPM0G5187?"
"What pins support SPI on the F28377D?"
```

## 📁 Project Structure

```
TI/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI application
│   │   └── models.py            # Pydantic models
│   ├── parsers/
│   │   └── pdf_parser.py        # PDF extraction & metadata
│   ├── ingestion/
│   │   └── ingest_datasheets.py # ChromaDB ingestion pipeline
│   ├── agent/
│   │   ├── graph.py             # LangGraph agent orchestration
│   │   └── tools.py             # Search tools
│   └── config.py                # Configuration
├── frontend/
│   ├── src/
│   │   ├── App.tsx              # Main application
│   │   ├── components/
│   │   │   └── ChatMessage.tsx  # Message component
│   │   └── services/
│   │       └── api.ts           # API client
│   └── package.json
├── Datasheets/                  # PDF datasheets (28 files)
├── chroma_db/                   # Vector DB (generated)
├── requirements.txt             # Python dependencies
├── setup.sh                     # Setup script
├── run_backend.sh              # Start backend
├── run_frontend.sh             # Start frontend
├── .env                        # Environment variables
└── README.md
```

## 🔧 Advanced Usage

### Adding New Datasheets

Place PDF files in the `Datasheets/` folder and run:

```bash
python3 -m backend.ingestion.ingest_datasheets --datasheet-dir Datasheets
```

### View Database Statistics

```bash
python3 -m backend.ingestion.ingest_datasheets --stats
```

### Health Check

```bash
curl http://localhost:8000/api/health
```

## 🧠 How It Works

### 1. PDF Parsing
- Extracts structured metadata (part numbers, specs, architecture)
- Identifies sections (Features, Specifications, Pin Config, etc.)
- Creates semantic chunks optimized for retrieval

### 2. Vector Storage
- Stores chunks in ChromaDB with metadata
- Generates embeddings using sentence-transformers
- Enables hybrid search (semantic + metadata filtering)

### 3. LangGraph Agent
- Classifies user intent (search, compare, recommend, troubleshoot)
- Decides when to ask clarifying questions
- Calls appropriate tools (semantic search, filtered search, comparison)
- Synthesizes responses with citations

### 4. Response Generation
- GPT-4o generates natural language responses
- Cites specific part numbers and specifications
- Explains trade-offs between options

## 🎨 Customization

### Change LLM Model

Edit `.env`:
```bash
OPENAI_MODEL=gpt-4o-mini  # Faster, cheaper
# or
OPENAI_MODEL=gpt-4o       # Default, most capable
```

### Adjust Search Parameters

Edit `backend/agent/tools.py`:
```python
# Increase number of search results
def semantic_search(query: str, top_k: int = 10):  # Was 5
```

### Add New Tools

1. Define tool in `backend/agent/tools.py`
2. Add to `self.tools` list in `backend/agent/graph.py`
3. Update system prompt with tool description

## 🐛 Troubleshooting

**Issue: "No results found"**
- Run `python3 -m backend.ingestion.ingest_datasheets --stats` to verify data
- Check if ChromaDB is populated: `ls chroma_db/`

**Issue: Slow responses**
- First query is slower (model loading)
- Consider using `gpt-4o-mini` for faster responses
- Check OpenAI API rate limits

**Issue: Backend won't start**
- Verify `.env` has valid `OPENAI_API_KEY`
- Install dependencies: `pip3 install -r requirements.txt`
- Check port 8000 isn't already in use

**Issue: Frontend can't reach backend**
- Ensure backend is running on port 8000
- Check `frontend/src/services/api.ts` has correct URL

## 🚀 Deployment (GCP VM)

For production deployment on Google Cloud Platform:

1. **Provision VM**
   - Machine: e2-standard-2 (2 vCPU, 8 GB)
   - OS: Ubuntu 22.04 LTS
   - Firewall: Allow HTTP/HTTPS

2. **Install Dependencies**
   ```bash
   sudo apt update
   sudo apt install python3.11 python3-pip nodejs npm
   ```

3. **Clone and Setup**
   ```bash
   git clone <your-repo>
   cd TI
   ./setup.sh
   ```

4. **Run with systemd**
   Create service files for backend and frontend
   (examples in `deployment/` folder if needed)

5. **Setup Nginx**
   Use Nginx as reverse proxy for production

## 📄 License

MIT

## 🤝 Contributing

This is a demonstration project. For production use:
- Add authentication
- Implement Redis for session storage
- Add rate limiting
- Set up monitoring (Prometheus, Grafana)
- Add comprehensive tests

## 📞 Support

For issues or questions:
- Check [QUICKSTART.md](QUICKSTART.md) for setup help
- Review error logs in backend console
- Verify ChromaDB statistics with `--stats` flag
