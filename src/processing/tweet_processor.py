from openai import OpenAI
from openai import AsyncOpenAI
import psycopg
import os
from typing import List, Optional, Union, Tuple
from datetime import datetime
import json
import logging
import sys
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import asyncio
import time
from collections import deque

load_dotenv()

logging.basicConfig(format="%(asctime)s %(levelname)s %(process)d: \
                    %(filename)s:%(lineno)d %(message)s", level=logging.INFO,
                    stream=sys.stdout)
logger = logging.getLogger("tweet_processor")

class CryptoProjectExtraction(BaseModel):
    """
    Pydantic model for structured extraction of cryptocurrency projects and
    tokens from text.
    
    Attributes:
        projects: List of standardized cryptocurrency project names
        tokens: List of cryptocurrency token names (without $ symbol)
        raw_projects: Original text mentions of cryptocurrency projects
        raw_tokens: Original text mentions of cryptocurrency tokens
        uncertainty: Confidence score for extraction accuracy (0-1)
    """
    projects: Optional[List[str]] = Field(None, description="A list of \
                                          cryptocurrency project names in \
                                          the text")
    tokens: Optional[List[str]] = Field(None, description="A list of \
                                        cryptocurrency token names in the \
                                        text. Only include the letters not \
                                        the dollar sign if it is present")
    raw_projects: Optional[List[str]] = Field(None, description="A list of \
                                              the original text of the \
                                              cryptocurrency project mentions \
                                              (for checking accuracy)")
    raw_tokens: Optional[List[str]] = Field(None, description="A list of the \
                                            original text of the \
                                            cryptocurrency token mentions \
                                            (for checking accuracy)")
    uncertainty: Optional[float] = Field(None, description="An indicator of \
                                         how sure you are about the accuracy \
                                         of the extraction, on a scale of 0 \
                                         to 1")
    
class ProjectValidatorProject(BaseModel):
    project_name: str = Field(description="The name of the project")
    project_description: str = Field(description="The description of the project")
    project_aliases: Optional[list[str]] = Field(description="The aliases of the project, including symbols. ONLY INCLUDE verified aliases. DO NOT INCLUDE possible aliases.")
    related_tokens: Optional[list[Union[str, Tuple[str, str]]]] = Field(description="The tokens related to the project. Can be either just the token name or a tuple of (token_name, token_symbol).")
    project_exists: bool = Field(description="Whether the project exists")

    @classmethod
    def is_valid(cls, data):
        try:
            valid = cls.model_validate(data)
            if valid:
                return True
            else:
                return False
        except Exception as e:
            logger.error("Validation error: %s", e)
            return False

class ProjectValidatorToken(BaseModel):
    token_name: str = Field(description="The name of the token. Return the name provided by the user if applicable.")
    token_symbol: Optional[str] = Field(description="The symbol of the token. Return the symbol provided by the user if applicable.")
    token_description: str = Field(description="The description of the token")
    related_project: str = Field(description="The project related to the token. ONLY INCLUDE a verified project. DO NOT INCLUDE a possible project.")
    token_aliases: Optional[list[str]] = Field(description="The aliases of the token. ONLY INCLUDE verified aliases. DO NOT INCLUDE possible aliases.")
    token_exists: bool = Field(description="Whether the token exists")

    @classmethod
    def is_valid(cls, data):
        try:
            valid = cls.model_validate(data)
            if valid:
                return True
            else:
                return False
        except Exception as e:
            logger.error(f"Validation error: {e}")
            return False

class RateLimiter:
    """
    Rate limiter for API calls to respect rate limits.
    Uses a token bucket algorithm to control request rates.
    
    Args:
        max_calls: Maximum number of calls allowed in the time window
        time_window_seconds: The time window in seconds (e.g., 60 for per minute, 10 for per 10 seconds)
    """
    def __init__(self, max_calls, time_window_seconds):
        self.max_calls = max_calls
        self.time_window_seconds = time_window_seconds
        self.tokens = max_calls  # Start with full bucket
        self.last_check = time.time()
        self.lock = asyncio.Lock()
        
    async def acquire(self):
        """Wait until a token is available to make an API call."""
        async with self.lock:
            current_time = time.time()
            time_passed = current_time - self.last_check
            self.last_check = current_time
            
            # Add tokens based on time passed, up to the maximum
            token_rate = self.max_calls / self.time_window_seconds  # tokens per second
            self.tokens = min(self.max_calls, self.tokens + time_passed * token_rate)
            
            if self.tokens < 1:
                # Not enough tokens, calculate wait time
                wait_time = (1 - self.tokens) / token_rate
                logger.debug(f"Rate limit reached, waiting {wait_time:.2f} seconds")
                await asyncio.sleep(wait_time)
                self.tokens = 0
            else:
                self.tokens -= 1

class TweetProcessor:
    """
    Processes tweets to extract cryptocurrency project and token mentions.
    
    This class handles database connections, tweet retrieval, and uses OpenAI's
    language models to extract cryptocurrency-related information from tweet
    text.
    """
    def __init__(self):
        """
        Initialize the TweetProcessor with OpenAI client and database
        connection.
        
        Sets up connections to the OpenAI API and PostgreSQL database using
        environment variables for configuration.
        """
        try:
            openai_api_key = os.getenv("OPENAI_API_KEY")
            perplexity_api_key = os.getenv("PERPLEXITY_API_KEY")
            openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
            self.openai_client = OpenAI(api_key=openai_api_key)
            self.perplexity_client = OpenAI(base_url="https://api.perplexity.ai", api_key=perplexity_api_key)
            self.openrouter_client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=openrouter_api_key)
            
            self.async_openai_client = AsyncOpenAI(api_key=openai_api_key)
            self.async_perplexity_client = AsyncOpenAI(base_url="https://api.perplexity.ai", api_key=perplexity_api_key)
            self.async_openrouter_client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=openrouter_api_key)
            
            self.max_concurrent_tweets = 200
            self.max_concurrent_validations = 70
            
            # self.tweet_db_conn = psycopg.connect(
            #     dbname=os.getenv("DATABASE_NAME"),
            #     user=os.getenv("DATABASE_USERNAME"),
            #     password=os.getenv("DATABASE_PASSWORD"),
            #     host=os.getenv("DATABASE_HOST"),
            #     port=os.getenv("DATABASE_PORT")
            # )
            self.tweet_db_conn = psycopg.connect(
                dbname=os.getenv("DATABASE_NAME"),
                user=os.getenv("DATABASE_USERNAME"),
                password=os.getenv("DATABASE_PASSWORD"),
                host=os.getenv("DATABASE_HOST"),
                port=os.getenv("DATABASE_PORT")
            )
            self.mindshare_db_conn = psycopg.connect(
                os.getenv("MINDSHARE_DB_URL")
            )
            self.checkpoint_file = "last_processed_tweet_timestamp.txt"
            
            self.openai_rate_limiter = RateLimiter(
                max_calls=4900,
                time_window_seconds=60
            )
            self.openrouter_rate_limiter = RateLimiter(
                max_calls=65,
                time_window_seconds=10
            )
            self.perplexity_rate_limiter = RateLimiter(
                max_calls=60,
                time_window_seconds=60
            )
        except psycopg.Error as e:
            logger.critical(f"Failed to connect to database: {e}")
            raise
        except Exception as e:
            logger.critical(f"Initialization error: {e}")
            raise
    def test_connection(self):
        '''
        Test the database connection by executing a simple query.
        
        Logs the result of a test query to verify database connectivity.
        '''
        try:
            with self.tweet_db_conn.cursor() as cur:
                cur.execute("SELECT text FROM tweets LIMIT 1")
                result = cur.fetchone()
                logger.info(f"Connection successful! Test query result: {result}")
        except psycopg.Error as e:
            logger.error(f"Database connection test failed: {e}")
            raise
    
    def get_last_processed_timestamp(self):
        """
        Get the timestamp of the last processed tweet.
        
        Returns:
            datetime: The timestamp of the last processed tweet or a default
                      date if no previous processing has occurred.
        """
        try:
            with open(self.checkpoint_file, 'r') as f:
                lines = f.readlines()
                if lines:
                    timestamp_str = lines[-1].strip()
                    return datetime.fromisoformat(timestamp_str)
                else:
                    # Use environment variable if available, otherwise use default
                    default_date_str = os.getenv('DEFAULT_START_DATE', '2025-02-24T11:00:00')
                    return datetime.fromisoformat(default_date_str)
        except (FileNotFoundError, ValueError):
            default_date_str = os.getenv('DEFAULT_START_DATE', '2025-02-24T11:00:00')
            return datetime.fromisoformat(default_date_str)
    
    def save_last_processed_timestamp(self, timestamp):
        """
        Save the timestamp of the last processed tweet by appending to file.
        
        Args:
            timestamp (datetime): The timestamp to save.
        """
        with open(self.checkpoint_file, 'a') as f:
            f.write(str(timestamp) + '\n')
        
    def validate_crypto(self, project_name=None, project_symbol=None, project_token=None):
        """
        Validate a project name using Perplexity API.
        
        Args:
            project_name (str): The name of the project to validate.
            project_symbol (str): The symbol of the project to validate.
            project_token (str): The token to validate.
        """
        if project_name:
            response_format = ProjectValidatorProject
            prompt = f"Does the project named '{project_name}' exist? If the name is not a project but a token, return the correct project name."
            prompt_output = "project_name, project_symbol, project_description, project_aliases, related_tokens, project_exists."
        elif project_symbol:
            response_format = ProjectValidatorToken
            prompt = f"Does the project symbol '{project_symbol}' exist? If the symbol is not a project but a token, return the correct project name."
            prompt_output = "token_name, token_symbol, token_description, related_project, token_aliases, token_exists."
        elif project_token:
            response_format = ProjectValidatorToken
            prompt = f"Does the crypto token '{project_token}' exist? "
            prompt_output = "token_name, token_symbol, token_description, related_project, token_aliases, token_exists."
        else:
            raise ValueError("No project name, symbol, or token provided")
        try:
            response = self.openrouter_client.chat.completions.create(
                model="perplexity/sonar-pro",
                messages=[
                    {"role": "system", "content": "Be precise and concise. Return only the JSON object."},
                    {"role": "user", "content": (
                        prompt +
                        "Please output a JSON object containing the following fields: " +
                        prompt_output
                    )},
                ],
                response_format={
                    "type": "json_schema",
                    "schema": response_format.model_json_schema()
                },
                temperature=0
            )
            content = response.choices[0].message.content
            
            try:
                parsed_data = json.loads(content)
                validated_model = response_format(**parsed_data)
                return validated_model.model_dump()
            except json.JSONDecodeError:
                json_start = content.find('{')
                if json_start == -1:
                    logger.error(f"No JSON object found in response: {content}")
                    return None
                    
                parsed_data, _ = json.JSONDecoder().raw_decode(content[json_start:])
                validated_model = response_format(**parsed_data)
                return validated_model.model_dump()
            else:
                logger.error("Parsed response is None or invalid")
                return None
           
        except Exception as e:
            logger.error(f"Error validating project/token {project_name or project_symbol or project_token}: {e}")
            return None
        
    def check_project_obj(self, project=None, token=None):
        """
        Check if a project or token exists in the database.
        
        Args:
            project: Project name to check
            token: Token name to check
            
        Returns:
            int: Project ID if found, None otherwise
        """
        try:
            with self.mindshare_db_conn.cursor() as cur:
                if project:
                    cur.execute("SELECT project_id FROM projects WHERE project_name = %s", (project,))
                    result = cur.fetchone()
                    if result:
                        return result[0]
                    else:
                        cur.execute("SELECT project_id FROM project_aliases WHERE alias = %s", (project,))
                        result = cur.fetchone()
                        if result:
                            return result[0]
                        else:
                            return None
                elif token:
                    cur.execute("SELECT project_id FROM tokens WHERE token_name = %s", (token,))
                    result = cur.fetchone()
                    if result:
                        return result[0]
                    else:
                        return None
                else:
                    raise ValueError("No project or token provided")
        except psycopg.Error as e:
            logger.error(f"Database error while checking project or token: {e}")
            return None
        except Exception as e:
            logger.error(f"Error while checking project or token: {e}")
            return None
    
    def process_tweet(self, tweet):
        """
        Process a single tweet to extract crypto project/token mentions and
        save the results to the database.
        
        Args:
            tweet (tuple): A tuple containing (id, text, user_id, username, 
                            timestamp) of the tweet to process.
                          
        Returns:
            datetime: The timestamp of the processed tweet.
        """
        try:
            id = tweet[0]
            text = tweet[1]
            timestamp = tweet[4]
            
            projects, tokens = self.extract_crypto_mentions(text)
            
            projects = projects or []
            tokens = tokens or []
            
            if projects or tokens:
                logger.info(f"Found mentions of projects: {projects}, "
                            f"tokens: {tokens} in tweet {id}, text: {text}")
                for project in projects:
                    logger.debug(f"Checking if project {project} exists with check_project_obj")
                    project_exists = self.check_project_obj(project)
                    logger.debug(f"Successfully checked if project {project} exists with check_project_obj")
                    if project_exists:
                        logger.info(f"Project {project} already exists in database")
                        self.save_extracted_info(tweet, project_id=project_exists)
                    else:
                        project_validation = self.validate_crypto(project_name=project)
                        if project_validation and project_validation.get("project_exists"):
                            logger.info(f"Project {project} exists")
                            project_id = self.save_crypto_info(project_validation, project=project)
                            if project_id is not None:
                                self.save_extracted_info(tweet, project_id=project_id)
                        else:
                            logger.info(f"Project {project} does not exist")
                for token in tokens:
                    token_exists = self.check_project_obj(token)
                    if token_exists:
                        logger.info(f"Token {token} already exists in database")
                        self.save_extracted_info(tweet, project_id=token_exists)
                    else:
                        token_validation = self.validate_crypto(project_token=token)
                        if token_validation and token_validation.get("token_exists"):
                            logger.info(f"Token {token} exists")
                            project_id = self.save_crypto_info(token_validation, token=token)
                            if project_id is not None:
                                self.save_extracted_info(tweet, project_id=project_id)
                        else:
                            logger.info(f"Token {token} does not exist")
            return timestamp
        except Exception as e:
            logger.error(f"Error processing tweet {tweet[0] if len(tweet) > 0 \
                                                   else 'unknown'}: {e}")
            return tweet[4] if len(tweet) > 4 else None
    
    def extract_crypto_mentions(self, text):
        """
        Use LLM to extract crypto project mentions from tweet text.
        
        Args:
            text (str): The text content of the tweet to analyze.
            
        Returns:
            tuple: A tuple containing (projects, tokens) where each is a list
                  of extracted cryptocurrency projects and tokens, or (None,
                  None) on error.
        """
        try:
            response = self.openai_client.beta.chat.completions.parse(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content":'''You are an AI assistant tasked with extracting crypto projects and tokens from tweets. This task requires careful attention to detail and the ability to recognize various forms and mentions of cryptocurrencies. Here are your instructions:
                     1. Definitions:
                        - Crypto project: A blockchain-based initiative (e.g., Ethereum, Cardano, Polkadot, Qubetics, MegaETH, aixbt, etc.)
                            - doesn't have to be the underlying blockchain, but a project that uses blockchain technology as a core technology
                        - Token: A cryptocurrency or digital asset associated with a specific project (e.g., ETH, ADA, DOT, $HBAR, AIXBT)
                     2. Your task is to identify and extract all mentions of crypto projects and tokens from the given tweet, considering the following scenarios:
                        a) Misspellings: Identify projects/tokens even if they are slightly misspelled (e.g., "Etherium" instead of "Ethereum")
                        b) Similar tokens: Differentiate between tokens with similar names or symbols (e.g., LUNA for Terra vs. LUNC for Luna Classic)
                        c) Symbol usage: Recognize when token symbols are used in place of full names (e.g., "BTC" for Bitcoin)
                        d) Abbreviations: Identify common abbreviations or shorthand for projects/tokens (e.g., "ETH" for Ethereum)
                        e) Hashtags: Extract projects/tokens mentioned in hashtags (e.g., #Bitcoin, #ETH)
                        f) Camel case: Recognize projects/tokens written in camel case (e.g., BitCoin, CardAno)
                        g) Slang or nicknames: Identify common slang terms or nicknames for projects/tokens (e.g., "Eth" for Ethereum, "Doge" for Dogecoin)
                        h) @: if someone mentions a project or token by name, it should be extracted

                    3. For each identified project or token, you will extract the following information:
                    - The extracted mention as it appears in the tweet
                    - The standardized name of the project or token
                    - Whether it's primarily a project, a token, or both

                    4. If you're unsure about a potential crypto mention, include it in your output and note your uncertainty.

                    5. If no crypto projects or tokens are found in the tweet, return an empty list.
                    
                    Please provide your analysis of the crypto projects and tokens mentioned in this tweet. Present your findings in a clear, structured format within the provided format.'''},
                    {"role": "user", "content": "Here is the tweet to analyze: " + text}
                ],
                response_format=CryptoProjectExtraction,
                temperature=0
            )
            
            crypto_project_extraction = response.choices[0].message
            data = json.loads(crypto_project_extraction.content)
            if crypto_project_extraction.refusal:
                logger.info(f"Failed to extract crypto projects or tokens, \
                            LLM refusal: {crypto_project_extraction.refusal}")
                return None, None
            else:
                projects = data.get("projects")
                tokens = data.get("tokens")
                return projects, tokens
                
        except Exception as e:
            logger.error(f"Error extracting crypto projects: {e}, \
                         text: {text}")
            return None, None
        
    def save_crypto_info(self, validation, project=None, token=None):
        """
        Save new project or token information to the database.
        """
        try:
            if project:
                with self.mindshare_db_conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO projects (project_name, description, is_active)
                        VALUES (%s, %s, %s)
                        RETURNING project_id
                    """, (project, validation["project_description"], validation["project_exists"]))
                    project_id = cur.fetchone()[0]
                    self.mindshare_db_conn.commit()
                    if validation["project_aliases"] and len(validation["project_aliases"]) > 0:
                        for alias in validation["project_aliases"]:
                            cur.execute("""
                                INSERT INTO project_aliases (project_id, alias)
                                VALUES (%s, %s)
                                """, (project_id, alias))
                            self.mindshare_db_conn.commit()
                    if validation["related_tokens"] and len(validation["related_tokens"]) > 0:
                        for token in validation["related_tokens"]:
                            if isinstance(token, (list, tuple)) and len(token) >= 2:
                                token_name = token[0]
                                token_symbol = token[1]
                            elif isinstance(token, str):
                                token_name = token
                                token_symbol = None
                            else:
                                logger.warning(f"Skipping invalid token format: {token}")
                                continue
                                
                            cur.execute("""
                                INSERT INTO tokens (token_name, project_id, token_symbol)
                                VALUES (%s, %s, %s)
                                """, (token_name, project_id, token_symbol))
                            self.mindshare_db_conn.commit()
                    return project_id
            elif token:
                with self.mindshare_db_conn.cursor() as cur:
                    project_name = validation["related_project"]
                    check_project = self.check_project_obj(project=project_name)
                    if check_project:
                        project_id = check_project
                    else:
                        cur.execute("""
                            INSERT INTO projects (project_name, is_active)
                            VALUES (%s, %s)
                            RETURNING project_id
                        """, (project_name, True))
                        project_id = cur.fetchone()[0]
                    cur.execute("""
                        INSERT INTO tokens (token_name, project_id, token_symbol)
                        VALUES (%s, %s, %s)
                        ON CONFLICT (project_id, token_name) DO NOTHING
                    """, (validation["token_name"], project_id, validation["token_symbol"]))
                    self.mindshare_db_conn.commit()
                    return project_id
            else:
                raise ValueError("No project or token provided")
        except psycopg.Error as e:
            self.mindshare_db_conn.rollback()
            logger.error(f"Database error while saving extracted info for "
                         f"tweet {id}: {e}")
            return None
        except Exception as e:
            self.mindshare_db_conn.rollback()
            logger.error(f"Error saving extracted info for tweet {id}: {e}")
            return None
    
    def save_extracted_info(self, tweet, project_id):
        """
        Save extracted cryptocurrency information to the database.
        
        Args:
            tweet (tuple): The tweet data.
            project_id (int): The ID of the project.
        """
        try:
            metrics = tweet[5]
            with self.mindshare_db_conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO mentions (tweet_timestamp, project_id, like_count, quote_count, reply_count, 
                                         retweet_count, bookmark_count, impression_count, tweet_id, user_id)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (tweet_timestamp, project_id, tweet_id) DO NOTHING
                """, (tweet[4], project_id, metrics["like_count"], metrics["quote_count"], 
                     metrics["reply_count"], metrics["retweet_count"], metrics["bookmark_count"], 
                     metrics["impression_count"], tweet[0], tweet[2]))
                self.mindshare_db_conn.commit()
                
                # Log whether it was a new insertion or not
                if cur.rowcount == 0:
                    logger.debug(f"Mention for tweet {tweet[0]} and project {project_id} already exists, skipped")
                else:
                    logger.debug(f"Saved mention for tweet {tweet[0]} and project {project_id}")
                
        except psycopg.Error as e:
            self.mindshare_db_conn.rollback()
            logger.error(f"Database error while saving extracted info for tweet {tweet[0]}: {e}")
        except Exception as e:
            self.mindshare_db_conn.rollback()
            logger.error(f"Error saving extracted info for tweet {tweet[0]}: {e}")

    def process_batch_tweets(self, batch_size=100):
        """
        Process historical tweets in batches.
        
        Retrieves tweets from the database that haven't been processed yet,
        extracts cryptocurrency information, and saves processing progress.
        
        Args:
            batch_size (int): The number of tweets to process in one batch.
        """
        try:
            last_timestamp = self.get_last_processed_timestamp()
            
            logger.info(f"Starting processing from timestamp: "
                        f"{last_timestamp}")
            
            with self.tweet_db_conn.cursor() as cur:
                cur.execute("""
                    SELECT id, text, user_id, username, timestamp, public_metrics
                    FROM tweets_full 
                    WHERE timestamp > %s
                    ORDER BY timestamp ASC
                    LIMIT %s
                """, (last_timestamp, batch_size))
                
                tweets = cur.fetchall()
                
                if not tweets:
                    logger.info("No new tweets to process")
                    return
                
                logger.info(f"Processing {len(tweets)} tweets")
                
                last_successful_timestamp = last_timestamp
                processed_count = 0
                
                for tweet in tweets:
                    try:
                        tweet_timestamp = self.process_tweet(tweet)
                        if tweet_timestamp:
                            last_successful_timestamp = tweet_timestamp
                            processed_count += 1
                    except Exception as e:
                        logger.error(f"Failed to process tweet "
                                     f"{tweet[0] if len(tweet) > 0 \
                                        else 'unknown'}: {e}")
                        continue
                
                self.save_last_processed_timestamp(last_successful_timestamp)
                logger.info(f"Successfully processed "
                            f"{processed_count}/{len(tweets)} tweets. "
                            f"Last timestamp: {last_successful_timestamp}")
        except psycopg.Error as e:
            logger.error(f"Database error during batch processing: {e}")
        except Exception as e:
            logger.error(f"Error during batch processing: {e}")

    async def async_validate_crypto(self, project_name=None, project_symbol=None, project_token=None):
        """
        Async version of validate_crypto using the async OpenAI client.
        
        Args:
            project_name (str): The name of the project to validate.
            project_symbol (str): The symbol of the project to validate.
            project_token (str): The token to validate.
        """
        await self.openrouter_rate_limiter.acquire()
        
        if project_name:
            response_format = ProjectValidatorProject
            prompt = f"Does the project named '{project_name}' exist? If the name is not a project but a token, return the correct project name."
            prompt_output = "project_name, project_symbol, project_description, project_aliases, related_tokens, project_exists."
        elif project_symbol:
            response_format = ProjectValidatorToken
            prompt = f"Does the project symbol '{project_symbol}' exist? If the symbol is not a project but a token, return the correct project name."
            prompt_output = "token_name, token_symbol, token_description, related_project, token_aliases, token_exists."
        elif project_token:
            response_format = ProjectValidatorToken
            prompt = f"Does the crypto token '{project_token}' exist? "
            prompt_output = "token_name, token_symbol, token_description, related_project, token_aliases, token_exists."
        else:
            raise ValueError("No project name, symbol, or token provided")
        try:
            response = await self.async_openrouter_client.chat.completions.create(
                model="perplexity/sonar-pro",
                messages=[
                    {"role": "system", "content": "Be precise and concise. Return only the JSON object."},
                    {"role": "user", "content": (
                        prompt +
                        "Please output a JSON object containing the following fields: " +
                        prompt_output
                    )},
                ],
                response_format={
                    "type": "json_schema",
                    "schema": response_format.model_json_schema()
                },
                temperature=0
            )
            content = response.choices[0].message.content
            
            try:
                parsed_data = json.loads(content)
                validated_model = response_format(**parsed_data)
                return validated_model.model_dump()
            except json.JSONDecodeError:
                json_start = content.find('{')
                if json_start == -1:
                    logger.error(f"No JSON object found in response: {content}")
                    return None
                    
                parsed_data, _ = json.JSONDecoder().raw_decode(content[json_start:])
                validated_model = response_format(**parsed_data)
                return validated_model.model_dump()
            except Exception as e:
                logger.error(f"Error parsing response: {e}")
                return None
           
        except Exception as e:
            logger.error(f"Error validating project/token {project_name or project_symbol or project_token}: {e}")
            return None
    
    async def async_extract_crypto_mentions(self, text):
        """
        Async version of extract_crypto_mentions using the async OpenAI client.
        
        Args:
            text (str): The text content of the tweet to analyze.
            
        Returns:
            tuple: A tuple containing (projects, tokens) where each is a list
                  of extracted cryptocurrency projects and tokens, or (None,
                  None) on error.
        """
        await self.openai_rate_limiter.acquire()
        
        try:
            response = await self.async_openai_client.beta.chat.completions.parse(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content":'''You are an AI assistant tasked with extracting crypto projects and tokens from tweets. This task requires careful attention to detail and the ability to recognize various forms and mentions of cryptocurrencies. Here are your instructions:
                     1. Definitions:
                        - Crypto project: A blockchain-based initiative (e.g., Ethereum, Cardano, Polkadot, Qubetics, MegaETH, aixbt, etc.)
                            - doesn't have to be the underlying blockchain, but a project that uses blockchain technology as a core technology
                        - Token: A cryptocurrency or digital asset associated with a specific project (e.g., ETH, ADA, DOT, $HBAR, AIXBT)
                     2. Your task is to identify and extract all mentions of crypto projects and tokens from the given tweet, considering the following scenarios:
                        a) Misspellings: Identify projects/tokens even if they are slightly misspelled (e.g., "Etherium" instead of "Ethereum")
                        b) Similar tokens: Differentiate between tokens with similar names or symbols (e.g., LUNA for Terra vs. LUNC for Luna Classic)
                        c) Symbol usage: Recognize when token symbols are used in place of full names (e.g., "BTC" for Bitcoin)
                        d) Abbreviations: Identify common abbreviations or shorthand for projects/tokens (e.g., "ETH" for Ethereum)
                        e) Hashtags: Extract projects/tokens mentioned in hashtags (e.g., #Bitcoin, #ETH)
                        f) Camel case: Recognize projects/tokens written in camel case (e.g., BitCoin, CardAno)
                        g) Slang or nicknames: Identify common slang terms or nicknames for projects/tokens (e.g., "Eth" for Ethereum, "Doge" for Dogecoin)
                        h) @: if someone mentions a project or token by name, it should be extracted

                    3. For each identified project or token, you will extract the following information:
                    - The extracted mention as it appears in the tweet
                    - The standardized name of the project or token
                    - Whether it's primarily a project, a token, or both

                    4. If you're unsure about a potential crypto mention, include it in your output and note your uncertainty.

                    5. If no crypto projects or tokens are found in the tweet, return an empty list.
                    
                    Please provide your analysis of the crypto projects and tokens mentioned in this tweet. Present your findings in a clear, structured format within the provided format.'''},
                    {"role": "user", "content": "Here is the tweet to analyze: " + text}
                ],
                response_format=CryptoProjectExtraction,
                temperature=0
            )
            
            crypto_project_extraction = response.choices[0].message
            data = json.loads(crypto_project_extraction.content)
            if crypto_project_extraction.refusal:
                logger.info(f"Failed to extract crypto projects or tokens, \
                            LLM refusal: {crypto_project_extraction.refusal}")
                return None, None
            else:
                projects = data.get("projects")
                tokens = data.get("tokens")
                return projects, tokens
                
        except Exception as e:
            logger.error(f"Error extracting crypto projects: {e}, \
                         text: {text}")
            return None, None
    
    async def async_process_tweet(self, tweet):
        """
        Async version of process_tweet that processes a single tweet with async LLM calls.
        
        Args:
            tweet (tuple): A tuple containing (id, text, user_id, username, 
                            timestamp) of the tweet to process.
                          
        Returns:
            datetime: The timestamp of the processed tweet.
        """
        try:
            id = tweet[0]
            text = tweet[1]
            timestamp = tweet[4]
            
            projects, tokens = await self.async_extract_crypto_mentions(text)
            
            projects = projects or []
            tokens = tokens or []
            
            if projects or tokens:
                logger.info(f"Found mentions of projects: {projects}, "
                            f"tokens: {tokens} in tweet {id}, text: {text}")
                
                # Track projects and tokens that need validation
                validation_tasks = []
                project_map = {}
                token_map = {}
                
                # Check which projects need validation
                for project in projects:
                    logger.debug(f"Checking if project {project} exists with check_project_obj")
                    project_exists = self.check_project_obj(project)
                    logger.debug(f"Successfully checked if project {project} exists with check_project_obj")
                    if project_exists:
                        logger.info(f"Project {project} already exists in database")
                        self.save_extracted_info(tweet, project_id=project_exists)
                    else:
                        # Create validation task and track it
                        task = self.async_validate_crypto(project_name=project)
                        validation_tasks.append(task)
                        project_map[task] = project
                
                # Check which tokens need validation
                for token in tokens:
                    token_exists = self.check_project_obj(token=token)
                    if token_exists:
                        logger.info(f"Token {token} already exists in database")
                        self.save_extracted_info(tweet, project_id=token_exists)
                    else:
                        # Create validation task and track it
                        task = self.async_validate_crypto(project_token=token)
                        validation_tasks.append(task)
                        token_map[task] = token
                
                # Run all validation tasks concurrently
                if validation_tasks:
                    validation_results = await asyncio.gather(*validation_tasks, return_exceptions=True)
                    
                    # Process results
                    for task, result in zip(validation_tasks, validation_results):
                        if isinstance(result, Exception):
                            if task in project_map:
                                logger.error(f"Error validating project {project_map[task]}: {result}")
                            else:
                                logger.error(f"Error validating token {token_map[task]}: {result}")
                            continue
                        
                        if task in project_map:
                            project = project_map[task]
                            if result and result.get("project_exists"):
                                logger.info(f"Project {project} exists")
                                project_id = self.save_crypto_info(result, project=project)
                                if project_id is not None:
                                    self.save_extracted_info(tweet, project_id=project_id)
                            else:
                                logger.info(f"Project {project} does not exist")
                        else:
                            token = token_map[task]
                            if result and result.get("token_exists"):
                                logger.info(f"Token {token} exists")
                                project_id = self.save_crypto_info(result, token=token)
                                if project_id is not None:
                                    self.save_extracted_info(tweet, project_id=project_id)
                            else:
                                logger.info(f"Token {token} does not exist")
            
            return timestamp
        except Exception as e:
            logger.error(f"Error processing tweet {tweet[0] if len(tweet) > 0 \
                                                   else 'unknown'}: {e}")
            return tweet[4] if len(tweet) > 4 else None
    
    async def async_process_batch_tweets(self, batch_size=100):
        """
        Async version of process_batch_tweets that processes tweets concurrently.
        
        Retrieves tweets from the database that haven't been processed yet,
        extracts cryptocurrency information, and saves processing progress.
        
        Args:
            batch_size (int): The number of tweets to process in one batch.
        """
        try:
            last_timestamp = self.get_last_processed_timestamp()
            
            logger.info(f"Starting processing from timestamp: "
                        f"{last_timestamp}")
            
            with self.tweet_db_conn.cursor() as cur:
                cur.execute("""
                    SELECT id, text, user_id, username, timestamp, public_metrics
                    FROM tweets_full 
                    WHERE timestamp > %s
                    ORDER BY timestamp ASC
                    LIMIT %s
                """, (last_timestamp, batch_size))
                
                tweets = cur.fetchall()
                
                if not tweets:
                    logger.info("No new tweets to process")
                    return
                
                logger.info(f"Processing {len(tweets)} tweets")
                
                last_successful_timestamp = last_timestamp
                processed_count = 0
                
                semaphore = asyncio.Semaphore(self.max_concurrent_tweets)
                
                async def process_with_semaphore(tweet):
                    async with semaphore:
                        return await self.async_process_tweet(tweet)
                
                tasks = [process_with_semaphore(tweet) for tweet in tweets]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        logger.error(f"Failed to process tweet {tweets[i][0]}: {result}")
                    elif result:
                        last_successful_timestamp = max(last_successful_timestamp, result)
                        processed_count += 1
                
                self.save_last_processed_timestamp(last_successful_timestamp)
                logger.info(f"Successfully processed "
                            f"{processed_count}/{len(tweets)} tweets. "
                            f"Last timestamp: {last_successful_timestamp}")
        except psycopg.Error as e:
            logger.error(f"Database error during batch processing: {e}")
        except Exception as e:
            logger.error(f"Error during batch processing: {e}")

    async def throttled_validate_crypto(self, semaphore, *args, **kwargs):
        """
        A wrapper around async_validate_crypto that respects a semaphore to limit concurrency.
        
        Args:
            semaphore: The semaphore to use for throttling
            *args, **kwargs: Arguments to pass to async_validate_crypto
            
        Returns:
            The result from async_validate_crypto
        """
        async with semaphore:
            return await self.async_validate_crypto(*args, **kwargs)

def main():
    """
    Main entry point for the tweet processor application.
    
    Initializes the TweetProcessor, tests the database connection,
    and processes a batch of tweets.
    """
    try:
        processor = TweetProcessor()
        # processor.test_connection()
        # processor.process_batch_tweets(batch_size=50)
        
        asyncio.run(processor.async_process_batch_tweets(batch_size=200))
    except Exception as e:
        logger.critical(f"Fatal error in main function: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()