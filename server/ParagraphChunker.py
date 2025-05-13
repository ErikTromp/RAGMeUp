from typing import List, Optional
import re

class ParagraphChunker:
    """
    A text splitter that respects paragraph boundaries and combines paragraphs until 
    reaching a maximum size without breaking any paragraph.
    """
    
    def __init__(self, 
                 max_chunk_size: int = 512,
                 paragraph_separator: str = "\n\s*\n"
                 ):
        """
        Initialize the ParagraphChunker.
        
        Args:
            max_chunk_size: The target maximum size for each chunk (can be exceeded for large paragraphs)
            paragraph_separator: Regex pattern to identify paragraph breaks (default: two or more newlines)
        """
        self.max_chunk_size = max_chunk_size
        self.paragraph_separator = paragraph_separator
        
    def split_text(self, text: str) -> List[str]:
        """
        Split text into chunks, respecting paragraph boundaries.
        
        Args:
            text: The text to split
            
        Returns:
            A list of text chunks
        """
        # Split the text into paragraphs
        paragraphs = re.split(self.paragraph_separator, text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        chunks = []
        current_chunk = []
        current_size = 0
        
        for paragraph in paragraphs:
            paragraph_size = len(paragraph)
            
            # If adding this paragraph would exceed the max size and we already have content,
            # finalize the current chunk
            if current_size > 0 and current_size + paragraph_size > self.max_chunk_size:
                chunks.append("\n\n".join(current_chunk))
                current_chunk = []
                current_size = 0
            
            # Add the paragraph to the current chunk
            current_chunk.append(paragraph)
            current_size += paragraph_size
        
        # Add the final chunk if it has content
        if current_chunk:
            chunks.append("\n\n".join(current_chunk))
        
        return chunks