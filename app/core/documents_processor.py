import tempfile 
from pathlib import Path
from typing import BinaryIO

from langchain_community.document_loaders import (
    CSVLoader,
    PyPDFLoader,
    TextLoader
)

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import get_settings
from app.utils.logger import get_logger 

logger = get_logger(__name__)

class DocumentProcessor:
    """Process Documents For Rag Pipeline"""

    SUPPORTED_EXTENSIONS ={".pdf",".txt",".csv"}

    def __init__(self,chunk_size:int | None =None,
                 chunk_overlap:int  | None =None,):
        """Initialize document processor 
        Args:
             chunk_size:Size of tet chunks(default from settings)
             chunk_overlap :overlap between chunks (default from setting)
        """
        settings = get_settings()
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap

        self.text_splitter =RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n","\n","."," ",""],
            length_function = len

        )
        logger.info(
            f"Document Processor initialized with chunk_size={self.chunk_size}",
            f"chunk_overlap={self.chunk_overlap}"
            
        )
    def load_pdf (self,file_path:str|Path)-> list[Document]:

        file_path=Path(file_path)
        logger.info(f"loading PDF:{file_path.name}")

        loader=PyPDFLoader(str(file_path))
        documents=loader.load
        logger.info(f"loaded{len(documents)} pages from {file_path.name}")
        return documents
        
        def load_text(self, file_path: str | Path) -> list[Document]:
            """Load a text file.

        Args:
            file_path: Path to text file

        Returns:
            List of Document objects
        """
        file_path = Path(file_path)
        logger.info(f"Loading text file: {file_path.name}")

        loader = TextLoader(str(file_path), encoding="utf-8")
        documents = loader.load()

        logger.info(f"Loaded text file: {file_path.name}")
        return documents

    def load_csv(self, file_path: str | Path) -> list[Document]:
        """Load a CSV file.

        Args:
            file_path: Path to CSV file

        Returns:
            List of Document objects (one per row)
        """
        file_path = Path(file_path)
        logger.info(f"Loading CSV: {file_path.name}")

        loader = CSVLoader(str(file_path), encoding="utf-8")
        documents = loader.load()

        logger.info(f"Loaded {len(documents)} rows from {file_path.name}")
        return documents
    
    