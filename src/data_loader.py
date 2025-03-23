import os
import pandas as pd
import re
from tqdm import tqdm
import glob
from config import Config

class ArticleParser:
    """Parser for ProQuest news articles."""
    
    def __init__(self):
        self.article_pattern = re.compile(r'____________________________________________________________\n\n(.*?)(?=____________________________________________________________|\Z)', 
                                         re.DOTALL)
        self.title_pattern = re.compile(r'(.*?)\n\nAuthor:', re.DOTALL)
        self.author_pattern = re.compile(r'Author: (.*?)\n', re.DOTALL)
        self.date_pattern = re.compile(r'date=(\d{4})', re.DOTALL)
        self.fulltext_pattern = re.compile(r'Full text: (.*?)(?=\Z)', re.DOTALL)
    
    def extract_articles(self, file_content):
        """Extract individual articles from a ProQuest file."""
        # Skip the search strategy at the beginning
        matches = self.article_pattern.finditer(file_content)
        articles = []
        
        for match in matches:
            article_text = match.group(1).strip()
            if not article_text or article_text.startswith("Search Strategy"):
                continue
                
            # Extract article components
            title_match = self.title_pattern.search(article_text)
            title = title_match.group(1).strip() if title_match else "Unknown Title"
            
            author_match = self.author_pattern.search(article_text)
            author = author_match.group(1).strip() if author_match else "Unknown Author"
            
            date_match = self.date_pattern.search(article_text)
            date = date_match.group(1) if date_match else "Unknown Date"
            
            fulltext_match = self.fulltext_pattern.search(article_text)
            fulltext = fulltext_match.group(1).strip() if fulltext_match else ""
            
            # Skip articles with no full text
            if not fulltext:
                continue
                
            articles.append({
                'title': title,
                'author': author,
                'date': date,
                'full_text': fulltext
            })
            
        return articles

class DataLoader:
    """Load and process the news articles dataset."""
    
    def __init__(self, years=None):
        self.config = Config()
        self.article_parser = ArticleParser()
        self.years = years or self.config.YEARS_TO_ANALYZE
        self.countries_df = self._load_countries()
    
    def _load_countries(self):
        """Load the country data from the state system file."""
        return pd.read_csv(self.config.STATE_SYSTEM_FILE)
    
    def get_country_names(self):
        """Get a list of country names from the state system."""
        return self.countries_df['statenme'].unique().tolist()
    
    def _get_article_files(self):
        """Get all article files for the specified years."""
        all_files = []
        for year in self.years:
            year_dir = os.path.join(self.config.ARTICLES_DIR, str(year))
            if os.path.exists(year_dir):
                files = glob.glob(os.path.join(year_dir, "*.txt"))
                all_files.extend(files)
        return all_files
    
    def load_articles(self):
        """Load all articles from the specified years."""
        article_files = self._get_article_files()
        all_articles = []
        
        for file_path in tqdm(article_files, desc="Loading articles"):
            try:
                with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                    file_content = f.read()
                
                articles = self.article_parser.extract_articles(file_content)
                
                # Add source file information
                for article in articles:
                    article['source_file'] = os.path.basename(file_path)
                    article['year'] = os.path.basename(os.path.dirname(file_path))
                
                all_articles.extend(articles)
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")
        
        return pd.DataFrame(all_articles)
    
    def prefilter_articles(self, articles_df):
        """Apply initial filtering to reduce the number of articles to process."""
        # Check for military-related keywords in text
        military_pattern = '|'.join(self.config.MILITARY_KEYWORDS)
        articles_df['has_military_terms'] = articles_df['full_text'].str.contains(
            military_pattern, case=False, regex=True)
        
        # Check for death-related keywords in text
        death_pattern = '|'.join(self.config.DEATH_KEYWORDS)
        articles_df['has_death_terms'] = articles_df['full_text'].str.contains(
            death_pattern, case=False, regex=True)
        
        # Check for country mentions
        countries = self.get_country_names()
        country_pattern = '|'.join([fr'\b{re.escape(country)}\b' for country in countries])
        articles_df['has_country_mentions'] = articles_df['full_text'].str.contains(
            country_pattern, case=False, regex=True)
        
        # Filter articles that have military terms, death terms, and country mentions
        filtered_df = articles_df[
            articles_df['has_military_terms'] & 
            articles_df['has_death_terms'] & 
            articles_df['has_country_mentions']
        ]
        
        return filtered_df 