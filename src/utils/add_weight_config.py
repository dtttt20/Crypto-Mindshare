"""
Utility script to add or update weight configurations for the mindshare calculator.
"""

import argparse
import sys
import os
import json
from datetime import datetime
import logging
import psycopg
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Add or update weight configurations for mindshare calculation')
    
    input_group = parser.add_mutually_exclusive_group(required=True)
    
    input_group.add_argument('--name', help='Name of the weight configuration')
    parser.add_argument('--description', default='', help='Description of the weight configuration')
    parser.add_argument('--weights', help='JSON string or path to JSON file containing weights (like_weight, quote_weight, reply_weight, retweet_weight, bookmark_weight, impression_weight, time_decay_half_life_days)')
    parser.add_argument('--inactive', action='store_true', help='Set this configuration as inactive (active by default)')
    
    input_group.add_argument('--json', help='Path to JSON file containing all configuration parameters (name, description, weights, active)')
    
    return parser.parse_args()

def load_weights(weights_input):
    """Load weights from a JSON string or file."""
    if not weights_input:
        raise ValueError("Weights input cannot be None")
        
    try:
        return json.loads(weights_input)
    except json.JSONDecodeError:
        if os.path.isfile(weights_input):
            with open(weights_input, 'r') as f:
                return json.load(f)
        else:
            raise ValueError(f"Could not parse weights as JSON or find file: {weights_input}")

class WeightConfig:
    def __init__(self):
        try:
            self.mindshare_db_conn = psycopg.connect(os.getenv("MINDSHARE_DB_URL"))
        except psycopg.Error as e:
            logger.error(f"Error connecting to mindshare database: {e}")
            sys.exit(1)

    def add_weight_config(self, name, description, weights, active=False):
        """Add a new weight configuration to the database."""
        try:
            with self.mindshare_db_conn.cursor() as cursor:
                if active:
                    logger.info("Setting all existing configurations as inactive")
                    cursor.execute("UPDATE weight_config SET is_active = FALSE")
                
                cursor.execute("SELECT * FROM weight_config WHERE name = %s", (name,))
                existing_config = cursor.fetchone()
                
                if existing_config:
                    logger.info(f"Updating existing weight configuration: {name}")
                    cursor.execute(
                        """
                        UPDATE weight_config
                        SET description = %s, weights = %s, is_active = %s, updated_at = %s
                        WHERE name = %s
                        """,
                        (description, json.dumps(weights), active, datetime.now(), name)
                    )
                else:
                    logger.info(f"Creating new weight configuration: {name}")
                    cursor.execute(
                        """
                        INSERT INTO weight_config
                        (name, description, weights, is_active, created_at, updated_at)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        """,
                        (name, description, json.dumps(weights), active, datetime.now(), datetime.now())
                    )
                
                self.mindshare_db_conn.commit()
                logger.info(f"Successfully {'updated' if existing_config else 'added'} weight configuration: {name}")
                
                if active:
                    logger.info(f"Configuration '{name}' is now the active configuration")
                
                return True
        except Exception as e:
            self.mindshare_db_conn.rollback()
            logger.error(f"Error adding weight configuration: {str(e)}")
            raise

def main():
    """Main entry point for the script."""
    args = parse_arguments()
    
    try:
        if args.json:
            if not os.path.isfile(args.json):
                raise ValueError(f"Could not find JSON configuration file: {args.json}")
            
            with open(args.json, 'r') as f:
                config = json.load(f)
            
            name = config.get('name')
            description = config.get('description', '')
            weights = config.get('weights')
            active = config.get('active', True)
            
            if not name:
                raise ValueError("Missing required 'name' parameter in JSON configuration")
            if not weights:
                raise ValueError("Missing required 'weights' parameter in JSON configuration")
        else:
            if not args.name:
                raise ValueError("--name is required when not using --json")
            if not args.weights:
                raise ValueError("--weights is required when not using --json")
            
            name = args.name
            description = args.description
            weights = load_weights(args.weights)
            active = not args.inactive
        
        required_fields = ['like_weight', 'quote_weight', 'reply_weight', 'retweet_weight', 
                          'bookmark_weight', 'impression_weight', 'time_decay_half_life_days']
        missing_fields = [field for field in required_fields if field not in weights]
        if missing_fields:
            raise ValueError(f"Missing required weight fields: {', '.join(missing_fields)}")
            
        weight_config = WeightConfig()
        weight_config.add_weight_config(name, description, weights, active)
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
