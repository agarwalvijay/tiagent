"""Ingestion pipeline for processing datasheets into ChromaDB."""
import sys
from pathlib import Path
from typing import List
import chromadb
from chromadb.config import Settings
from tqdm import tqdm
import argparse

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from backend.parsers.pdf_parser import TIDatasheetParser, DatasheetChunk
from backend.parsers.parametrics_parser import ParametricsParser
from backend.config import settings


class DatasheetIngestionPipeline:
    """Pipeline for ingesting datasheets into ChromaDB."""

    def __init__(self, persist_directory: str = None, collection_name: str = None, parametrics_csv_path: str = None):
        """
        Initialize the ingestion pipeline.

        Args:
            persist_directory: Path to ChromaDB storage
            collection_name: Name of the collection
            parametrics_csv_path: Optional path to TI parametrics CSV for authoritative metadata
        """
        self.persist_directory = persist_directory or settings.chroma_persist_dir
        self.collection_name = collection_name or settings.chroma_collection_name

        # Initialize parametrics parser if CSV provided
        self.parametrics_parser = None
        if parametrics_csv_path:
            try:
                print(f"Loading parametrics CSV: {parametrics_csv_path}")
                self.parametrics_parser = ParametricsParser(parametrics_csv_path)
                stats = self.parametrics_parser.get_stats()
                print(f"  ✓ Loaded {stats['mspm0_products']} MSPM0 products from {stats['total_products']} total")
            except Exception as e:
                print(f"  ⚠️  Failed to load parametrics CSV: {e}")
                print(f"  → Will use PDF-extracted metadata as fallback")

        # Initialize ChromaDB with persistent storage
        self.client = chromadb.PersistentClient(
            path=self.persist_directory
        )

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "TI Semiconductor Datasheets"}
        )

        # Initialize embedding model
        print(f"Loading embedding model: {settings.embedding_model}")
        self.use_openai_embeddings = settings.embedding_model.startswith("openai/")

        if self.use_openai_embeddings:
            from openai import OpenAI
            self.openai_client = OpenAI(api_key=settings.openai_api_key)
            self.embedding_model_name = settings.embedding_model.replace("openai/", "")
            print(f"  Using OpenAI embeddings: {self.embedding_model_name}")
        else:
            from sentence_transformers import SentenceTransformer
            self.embedding_model = SentenceTransformer(settings.embedding_model)
            print(f"  Using local sentence-transformers: {settings.embedding_model}")

    def ingest_datasheet(self, pdf_path: str) -> bool:
        """Ingest a single datasheet."""
        try:
            print(f"\nProcessing: {Path(pdf_path).name}")

            # Parse the datasheet (with optional parametrics enrichment)
            parser = TIDatasheetParser(pdf_path, parametrics_parser=self.parametrics_parser)
            metadata, chunks = parser.parse()
            parser.close()

            print(f"  Document ID: {metadata.document_id}")
            print(f"  Part Numbers: {', '.join(metadata.part_numbers[:3])}")
            print(f"  Device Type: {metadata.device_type}")
            print(f"  Extracted {len(chunks)} chunks")

            # Check if already ingested (by document_id)
            existing = self.collection.get(
                where={"document_id": metadata.document_id}
            )

            if existing['ids']:
                print(f"  ⚠️  Already ingested. Updating...")
                # Delete existing chunks
                self.collection.delete(ids=existing['ids'])

            # Prepare data for ChromaDB
            chunk_ids = []
            chunk_texts = []
            chunk_metadatas = []

            for chunk in tqdm(chunks, desc="  Preparing chunks"):
                chunk_ids.append(chunk.chunk_id)

                # Truncate if too long for OpenAI embeddings (8192 tokens ≈ 30000 chars max)
                content = chunk.content
                if len(content) > 30000:
                    content = content[:30000]

                chunk_texts.append(content)

                # Prepare metadata (ChromaDB requires flat dict)
                chunk_metadata = {
                    **chunk.metadata,
                    "chunk_type": chunk.chunk_type,
                    "page_numbers": ",".join(map(str, chunk.page_numbers)),
                    "pdf_filename": Path(pdf_path).name,
                }

                # Convert all values to strings or numbers (ChromaDB requirement)
                for key, value in chunk_metadata.items():
                    if value is None:
                        chunk_metadata[key] = ""
                    elif isinstance(value, (list, tuple)):
                        chunk_metadata[key] = ",".join(map(str, value))
                    elif not isinstance(value, (str, int, float, bool)):
                        chunk_metadata[key] = str(value)

                chunk_metadatas.append(chunk_metadata)

            # Generate embeddings
            print("  Generating embeddings...")
            embeddings = self._generate_embeddings(chunk_texts)

            # Add to ChromaDB
            print("  Adding to ChromaDB...")
            self.collection.add(
                ids=chunk_ids,
                documents=chunk_texts,
                embeddings=embeddings,
                metadatas=chunk_metadatas
            )

            print(f"  ✓ Successfully ingested {len(chunks)} chunks")
            return True

        except Exception as e:
            print(f"  ✗ Error processing {pdf_path}: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

    def _generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using configured model (OpenAI or sentence-transformers)."""
        if self.use_openai_embeddings:
            # Use OpenAI embeddings
            embeddings = []

            # Process in batches of 100 (OpenAI limit is 2048, but we'll be conservative)
            batch_size = 100
            for i in tqdm(range(0, len(texts), batch_size), desc="  OpenAI API calls"):
                batch = texts[i:i + batch_size]

                try:
                    response = self.openai_client.embeddings.create(
                        model=self.embedding_model_name,
                        input=batch
                    )

                    batch_embeddings = [e.embedding for e in response.data]
                    embeddings.extend(batch_embeddings)

                except Exception as e:
                    print(f"\n  ⚠️  Error generating embeddings for batch {i//batch_size}: {e}")
                    # Fallback: try one at a time
                    for text in batch:
                        try:
                            # Truncate if too long (8192 tokens ≈ 32000 chars)
                            if len(text) > 32000:
                                text = text[:32000]
                                print(f"  ⚠️  Truncated oversized text to 32000 chars")

                            response = self.openai_client.embeddings.create(
                                model=self.embedding_model_name,
                                input=[text]
                            )
                            embeddings.append(response.data[0].embedding)
                        except Exception as e2:
                            print(f"  ⚠️  Failed even after truncation, using zero vector: {e2}")
                            # Use zero vector as fallback (will have low similarity)
                            embeddings.append([0.0] * 1536)  # text-embedding-3-small dimension

            return embeddings
        else:
            # Use sentence-transformers (local)
            embeddings = self.embedding_model.encode(
                texts,
                show_progress_bar=True,
                convert_to_numpy=True
            ).tolist()
            return embeddings

    def ingest_directory(self, directory: str, pattern: str = "*.pdf") -> dict:
        """Ingest all PDFs in a directory."""
        pdf_dir = Path(directory)
        pdf_files = list(pdf_dir.glob(pattern))

        if not pdf_files:
            print(f"No PDF files found in {directory}")
            return {"success": 0, "failed": 0}

        print(f"\nFound {len(pdf_files)} PDF files to process\n")
        print("=" * 70)

        results = {"success": 0, "failed": 0}

        for pdf_path in pdf_files:
            if self.ingest_datasheet(str(pdf_path)):
                results["success"] += 1
            else:
                results["failed"] += 1

        print("\n" + "=" * 70)
        print(f"\nIngestion complete!")
        print(f"  ✓ Success: {results['success']}")
        print(f"  ✗ Failed: {results['failed']}")
        print(f"\nTotal documents in collection: {self.collection.count()}")

        return results

    def get_stats(self) -> dict:
        """Get statistics about the ingested data."""
        count = self.collection.count()

        # Get sample to analyze
        sample = self.collection.get(limit=min(100, count), include=["metadatas"])

        # Aggregate stats
        device_types = {}
        architectures = {}

        for metadata in sample["metadatas"]:
            device_type = metadata.get("device_type", "Unknown")
            device_types[device_type] = device_types.get(device_type, 0) + 1

            architecture = metadata.get("architecture", "Unknown")
            architectures[architecture] = architectures.get(architecture, 0) + 1

        return {
            "total_chunks": count,
            "device_types": device_types,
            "architectures": architectures,
        }


def main():
    """Main entry point for the ingestion script."""
    parser = argparse.ArgumentParser(description="Ingest TI datasheets into ChromaDB")
    parser.add_argument(
        "--datasheet-dir",
        type=str,
        default="Datasheets",
        help="Directory containing PDF datasheets"
    )
    parser.add_argument(
        "--parametrics-csv",
        type=str,
        default=None,
        help="Path to TI parametrics CSV file for authoritative metadata"
    )
    parser.add_argument(
        "--incremental",
        action="store_true",
        help="Only process new datasheets"
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show statistics about ingested data"
    )

    args = parser.parse_args()

    # Initialize pipeline (with optional parametrics CSV)
    pipeline = DatasheetIngestionPipeline(parametrics_csv_path=args.parametrics_csv)

    if args.stats:
        # Show stats
        stats = pipeline.get_stats()
        print("\n" + "=" * 70)
        print("CHROMADB STATISTICS")
        print("=" * 70)
        print(f"\nTotal chunks: {stats['total_chunks']}")
        print(f"\nDevice types:")
        for dtype, count in stats['device_types'].items():
            print(f"  - {dtype}: {count}")
        print(f"\nArchitectures:")
        for arch, count in stats['architectures'].items():
            print(f"  - {arch}: {count}")
        return

    # Ingest datasheets
    pipeline.ingest_directory(args.datasheet_dir)

    # Show final stats
    print("\n")
    stats = pipeline.get_stats()
    print(f"Final collection size: {stats['total_chunks']} chunks")


if __name__ == "__main__":
    main()
