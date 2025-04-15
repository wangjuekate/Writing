
"""
Enhanced BibTeX Parser using scholarly and pybtex
"""
from pybtex.database.input import bibtex
import pandas as pd
import numpy as np
from scholarly import scholarly
import re
from tqdm import tqdm
import logging
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def clean_text(text):
    """Clean text by removing newlines and extra spaces"""
    if not text:
        return ""
    return re.sub(r'\s+', ' ', text.replace('\n', ' ')).strip()

def extract_doi(text):
    """Extract DOI from text using improved pattern matching"""
    if not text:
        return None
    
    # DOI patterns
    patterns = [
        r'10\.\d{4,}/[-._;()/:\w]+',  # Basic DOI pattern
        r'DOI[^\n]*?(10\.\d{4,}/[-._;()/:\w]+)',  # DOI with prefix
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            doi = match.group(0) if 'DOI' not in match.group(0).upper() else match.group(1)
            return clean_text(doi)
    return None

def parse_authors(entry):
    """Parse author information from a BibTeX entry"""
    if 'author' not in entry.persons:
        return []
    
    authors = []
    for person in entry.persons['author']:
        author = {
            'first': ' '.join(person.first()),
            'last': ' '.join(person.last()),
            'middle': ' '.join(person.middle()),
        }
        authors.append(author)
    return authors

def extract_year(text):
    """Extract year from text"""
    if not text:
        return None
    match = re.search(r'\b(19|20)\d{2}\b', text)
    return int(match.group(0)) if match else None

def parse_bibtex_file(file_path):
    """Parse BibTeX file into structured DataFrames"""
    logging.info(f"Reading BibTeX file: {file_path}")
    
    # Initialize parser
    parser = bibtex.Parser()
    bib_data = parser.parse_file(file_path)
    
    # Initialize data containers
    papers = []
    authors = []
    citations = []
    
    # Process each entry
    logging.info("Processing entries...")
    for entry_key, entry in tqdm(bib_data.entries.items()):
        # Extract basic paper information
        fields = entry.fields
        
        # Process paper
        paper = {
            'entry_key': entry_key,
            'title': clean_text(str(fields.get('title', ''))),
            'year': extract_year(str(fields.get('year', ''))),
            'journal': clean_text(str(fields.get('journal', ''))),
            'doi': extract_doi(str(fields.get('doi', ''))),
            'abstract': clean_text(str(fields.get('abstract', ''))),
            'volume': str(fields.get('volume', '')),
            'number': str(fields.get('number', '')),
            'pages': str(fields.get('pages', '')),
            'publisher': clean_text(str(fields.get('publisher', ''))),
            'times_cited': int(fields.get('times-cited', 0)),
            'usage_count_180': int(fields.get('usage-count-last-180-days', 0)),
            'usage_count_2013': int(fields.get('usage-count-since-2013', 0))
        }
        papers.append(paper)
        
        # Process authors
        entry_authors = parse_authors(entry)
        for author in entry_authors:
            author_data = {
                'entry_key': entry_key,
                'first_name': author['first'],
                'middle_name': author['middle'],
                'last_name': author['last'],
                'full_name': f"{author['first']} {author['middle']} {author['last']}".strip()
            }
            authors.append(author_data)
        
        # Process citations
        if 'cited-references' in fields:
            refs = str(fields['cited-references']).split('\n')
            for ref in refs:
                ref = clean_text(ref)
                if not ref:
                    continue
                
                citation = {
                    'citing_key': entry_key,
                    'citing_doi': paper['doi'],
                    'cited_doi': extract_doi(ref),
                    'cited_year': extract_year(ref),
                    'raw_reference': ref
                }
                citations.append(citation)
    
    # Create DataFrames
    papers_df = pd.DataFrame(papers)
    authors_df = pd.DataFrame(authors)
    citations_df = pd.DataFrame(citations)
    
    # Save to CSV
    logging.info("Saving results to CSV files...")
    papers_df.to_csv('papers_enhanced.csv', index=False)
    authors_df.to_csv('authors_enhanced.csv', index=False)
    citations_df.to_csv('citations_enhanced.csv', index=False)
    
    # Generate summary statistics
    print("\nDataset Summary:")
    print("-" * 50)
    print(f"Total papers: {len(papers_df)}")
    print(f"Total authors: {len(authors_df['full_name'].unique())}")
    print(f"Total citations: {len(citations_df)}")
    print(f"\nYear range: {papers_df['year'].min()} - {papers_df['year'].max()}")
    print(f"Total citation count: {papers_df['times_cited'].sum():,}")
    
    print("\nTop 5 Most Cited Papers:")
    print(papers_df.nlargest(5, 'times_cited')[['title', 'times_cited']])
    
    print("\nTop 5 Authors by Paper Count:")
    author_counts = authors_df['full_name'].value_counts().head()
    print(author_counts)
    
    return papers_df, authors_df, citations_df

if __name__ == "__main__":
    bib_file = "../CitationResponsibleAI.bib"
    papers_df, authors_df, citations_df = parse_bibtex_file(bib_file)
