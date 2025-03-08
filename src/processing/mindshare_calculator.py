import psycopg
from psycopg.rows import dict_row
import redis
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
import json
from typing import Dict, List, Tuple, Any, Optional
import os
import sys
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(format="%(asctime)s %(levelname)s %(process)d: \
                    %(filename)s:%(lineno)d %(message)s", level=logging.INFO,
                    stream=sys.stdout)
logger = logging.getLogger("mindshare_calculator")

class MindshareCalculator:
    """
    Service to calculate mindshare percentages for crypto projects based on 
    social media mentions.
    """
    
    def __init__(self, redis_config: Dict[str, Any]):
        """
        Initialize the calculator with database and Redis configurations.
        
        Args:
            db_config: Database connection parameters
            redis_config: Redis connection parameters
        """
        self.mindshare_db_conn = psycopg.connect(
                os.getenv("MINDSHARE_DB_URL")
        )
        # self.redis_client = redis.Redis(**redis_config)
    
    def get_active_weight_config(self) -> Dict[str, Any]:
        """
        Retrieve the currently active weight configuration from the database.
        
        Returns:
            Dictionary containing weight parameters
        """
        with self.mindshare_db_conn.cursor(row_factory=dict_row) as cursor:
            cursor.execute("""
                SELECT * FROM weight_config WHERE is_active = TRUE
            """)
            config = cursor.fetchone()
            
            if not config:
                logger.error("No active weight configuration found")
                raise ValueError("No active weight configuration")
                
            return dict(config)
    
    def get_recent_mentions(self, days: int = 30) -> Tuple[pd.DataFrame, datetime]:
        """
        Fetch recent tweet mentions for calculation.
        
        Args:
            days: Number of days of history to include
            
        Returns:
            Tuple containing the mentions DataFrame and current timestamp
        """
        timestamp = datetime.now()
        cutoff_date = timestamp - timedelta(days=days)
        
        query = """
            SELECT 
                tweet_timestamp, project_id, 
                like_count, quote_count, reply_count, retweet_count,
                bookmark_count, impression_count, tweet_id, user_id
            FROM mentions
            WHERE tweet_timestamp > %s
        """
        
        logger.info(f"Fetching mentions since {cutoff_date}")
        mentions_df = pd.read_sql(query, self.mindshare_db_conn, params=(cutoff_date,))
        logger.info(f"Fetched {len(mentions_df)} mentions")
        
        return mentions_df, timestamp
    
    def calculate_weighted_mentions(
        self, 
        mentions_df: pd.DataFrame, 
        weights: Dict[str, float], 
        timestamp: datetime
    ) -> pd.DataFrame:
        """
        Calculate weighted mentions for each project using the specified weights.
        
        Args:
            mentions_df: DataFrame containing tweet mentions
            weights: Weight configuration dictionary
            timestamp: Current timestamp for time decay calculation
            
        Returns:
            DataFrame with project_id and weighted_mentions columns
        """
        half_life_days = weights['time_decay_half_life_days']
        # follower_w = weights['follower_weight']
        like_w = weights['like_weight']
        quote_w = weights['quote_weight']
        reply_w = weights['reply_weight']
        retweet_w = weights['retweet_weight']
        bookmark_w = weights['bookmark_weight']
        impression_w = weights['impression_weight']
        
        # Calculate time decay (days, with 30-day half-life)
        mentions_df['days_ago'] = (timestamp - pd.to_datetime(mentions_df['tweet_timestamp'])).dt.total_seconds() / (24 * 3600)
        
        # Using half-life formula: decay_factor = 0.5^(days/half_life_days)
        mentions_df['time_decay_factor'] = np.power(0.5, mentions_df['days_ago'] / half_life_days)
        
        # Calculate total engagement weighted mentions
        mentions_df['weighted_mentions'] = (
            # (1 + follower_w * np.log1p(mentions_df['user_follower_count'])) *
            (1 + like_w * mentions_df['like_count']) *
            (1 + retweet_w * mentions_df['retweet_count']) *
            (1 + quote_w * mentions_df['quote_count']) *
            (1 + reply_w * mentions_df['reply_count']) *
            (1 + bookmark_w * mentions_df['bookmark_count']) *
            (1 + impression_w * mentions_df['impression_count'])
        )
        
        # Calculate final weight with time decay
        mentions_df['final_weighted_mentions'] = mentions_df['time_decay_factor'] * mentions_df['weighted_mentions']
        
        # Group by project and sum weights
        project_weights = mentions_df.groupby('project_id')['final_weighted_mentions'].sum().reset_index()
        project_weights.columns = ['project_id', 'weighted_mentions']
        
        return project_weights
    
    def calculate_mindshare(self) -> bool:
        """
        Main method to calculate mindshare percentages.
        
        Returns:
            Boolean indicating success or failure
        """
        try:
            logger.info("Starting mindshare calculation")
            
            weight_config = self.get_active_weight_config()
            weights = weight_config['weights']
            weight_id = weight_config['config_id']
            days_to_calculate = weights['time_decay_half_life_days']
            
            mentions_df, timestamp = self.get_recent_mentions(days=days_to_calculate)
            
            if len(mentions_df) == 0:
                logger.warning("No mentions found for calculation")
                return True
            
            project_weights = self.calculate_weighted_mentions(mentions_df, weights, timestamp)
            
            total_weighted_mentions = project_weights['weighted_mentions'].sum()
            if total_weighted_mentions > 0:
                project_weights['mindshare_percentage'] = (
                    project_weights['weighted_mentions'] / total_weighted_mentions * 100
                )
            else:
                project_weights['mindshare_percentage'] = 0
                logger.warning("Total weighted mentions is zero")
            
            project_weights = project_weights.sort_values('mindshare_percentage', ascending=False)
            project_weights['ranking'] = range(1, len(project_weights) + 1)
            
            self.store_mindshare_results(project_weights, weight_id, timestamp)
            #self.update_redis_cache(project_weights, weight_id, timestamp)
            #self.update_influential_tweets(mentions_df, weights, timestamp)
            logger.info("Mindshare calculation completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error calculating mindshare: {str(e)}", exc_info=True)
            return False
    
    def store_mindshare_results(
        self, 
        project_weights: pd.DataFrame, 
        weight_id: int, 
        timestamp: datetime
    ) -> None:
        """
        Store the calculated mindshare results in the database.
        
        Args:
            project_weights: DataFrame with project weights and mindshare percentages
            weight_id: ID of the weight configuration used
            timestamp: Timestamp for the snapshot
        """
        try:
            with self.mindshare_db_conn.cursor() as cursor:
                for _, row in project_weights.iterrows():
                    cursor.execute("""
                        INSERT INTO mindshare_snapshots 
                        (project_id, weight_config_id, mindshare_percentage, snapshot_timestamp, total_weighted_mentions, ranking)
                        VALUES (%s, %s, %s, %s, %s, %s)
                    """, (
                        int(row['project_id']), 
                        weight_id,
                        float(row['mindshare_percentage']),
                        timestamp,
                        float(row['weighted_mentions']),
                        int(row['ranking']),
                    ))
                
                self.mindshare_db_conn.commit()
        except Exception as e:
            self.mindshare_db_conn.rollback()
            logger.error(f"Error storing mindshare results: {e}")
    
    def update_redis_cache(
        self, 
        project_weights: pd.DataFrame, 
        weight_id: int, 
        timestamp: datetime
    ) -> None:
        """
        Update Redis cache with latest mindshare data.
        
        Args:
            project_weights: DataFrame with calculation results
            weight_id: ID of the weight configuration used
            timestamp: Current timestamp
        """
        timestamp_str = timestamp.strftime("%Y-%m-%d-%H-%M")
        logger.info("Updating Redis cache")
        
        ranking_key = f"mindshare_ranking:{weight_id}:{timestamp_str}"
        pipeline = self.redis_client.pipeline()
        
        pipeline.delete(ranking_key)
        
        top_100 = project_weights.head(100)
        for _, row in top_100.iterrows():
            pipeline.zadd(
                ranking_key, 
                {str(int(row['project_id'])): float(row['mindshare_percentage'])}
            )
        
        # 24hr expiry
        pipeline.expire(ranking_key, 86400)
        
        # Update current day's data for each project
        day_str = timestamp.strftime("%Y-%m-%d")
        for _, row in project_weights.iterrows():
            project_key = f"project_mindshare:{int(row['project_id'])}:{weight_id}"
            pipeline.hset(
                project_key,
                day_str,
                float(row['mindshare_percentage'])
            )
            pipeline.expire(project_key, 30 * 86400)  # 30 days expiry
        
        pipeline.execute()
        
        self.update_gainers_losers_cache(weight_id, timestamp)
    
    def update_gainers_losers_cache(self, weight_id: int, timestamp: datetime) -> None:
        """
        Calculate and update top gainers and losers in the Redis cache.
        
        Args:
            weight_id: ID of the weight configuration used
            timestamp: Current timestamp
        """
        # Time periods to calculate (in days)
        periods = [1, 7, 30]
        
        for period in periods:
            previous_time = timestamp - timedelta(days=period)
            
            with self.mindshare_db_conn.cursor(row_factory=dict_row) as cursor:
                cursor.execute("""
                    WITH current_data AS (
                        SELECT 
                            project_id, 
                            mindshare_percentage
                        FROM mindshare_snapshots
                        WHERE weight_config_id = %s AND snapshot_timestamp BETWEEN %s - INTERVAL '10 minutes' AND %s
                    ),
                    previous_data AS (
                        SELECT 
                            project_id, 
                            mindshare_percentage
                        FROM mindshare_snapshots
                        WHERE weight_config_id = %s AND snapshot_timestamp BETWEEN %s - INTERVAL '10 minutes' AND %s
                    )
                    SELECT 
                        c.project_id,
                        c.mindshare_percentage as current_percentage,
                        p.mindshare_percentage as previous_percentage,
                        (c.mindshare_percentage - p.mindshare_percentage) as change,
                        (c.mindshare_percentage - p.mindshare_percentage) / 
                            NULLIF(p.mindshare_percentage, 0) * 100 as percent_change
                    FROM current_data c
                    JOIN previous_data p ON c.project_id = p.project_id
                """, (
                    weight_id, timestamp - timedelta(minutes=10), timestamp,
                    weight_id, previous_time - timedelta(minutes=10), previous_time
                ))
                
                changes = cursor.fetchall()
            
            # Update Redis cache
            if changes:
                gainers_key = f"mindshare_gainers_{period}d:{weight_id}"
                losers_key = f"mindshare_losers_{period}d:{weight_id}"
                
                pipeline = self.redis_client.pipeline()
                
                # Clear previous data
                pipeline.delete(gainers_key)
                pipeline.delete(losers_key)
                
                # Add new data
                for row in changes:
                    project_id = row['project_id']
                    pct_change = row['percent_change'] or 0
                    
                    if pct_change > 0:
                        pipeline.zadd(gainers_key, {str(project_id): float(pct_change)})
                    else:
                        pipeline.zadd(losers_key, {str(project_id): float(abs(pct_change))})
                
                # Set expiry (keep for 24 hours)
                pipeline.expire(gainers_key, 86400)
                pipeline.expire(losers_key, 86400)
                
                pipeline.execute()
    
    def update_influential_tweets(
        self, 
        mentions_df: pd.DataFrame, 
        weights: Dict[str, float], 
        timestamp: datetime
    ) -> None:
        """
        Identify and store the most influential tweets for each project.
        
        Args:
            mentions_df: DataFrame containing tweet mentions
            weights: Weight configuration dictionary
            timestamp: Current timestamp
        """
        logger.info("Identifying influential tweets")
        
        # Extract weight parameters
        half_life_days = weights['time_decay_half_life_days']
        # follower_w = weights['follower_weight']
        like_w = weights['like_weight']
        retweet_w = weights['retweet_weight']
        quote_w = weights['quote_weight']
        reply_w = weights['reply_weight']
        bookmark_w = weights['bookmark_weight']
        impression_w = weights['impression_weight']
        
        # Filter recent tweets (last 24 hours)
        recent_cutoff = timestamp - timedelta(days=1)
        recent_mentions = mentions_df[pd.to_datetime(mentions_df['tweet_timestamp']) > recent_cutoff].copy()
        
        if len(recent_mentions) == 0:
            return
        
        # Calculate engagement score for each tweet
        recent_mentions['days_ago'] = (timestamp - pd.to_datetime(recent_mentions['tweet_timestamp'])).dt.total_seconds() / (24 * 3600)
        
        # Using half-life formula: decay_factor = 0.5^(days/half_life_days)
        recent_mentions['time_decay_factor'] = np.power(0.5, recent_mentions['days_ago'] / half_life_days)
        
        recent_mentions['weighted_score'] = (
            recent_mentions['time_decay_factor'] *
            # (1 + follower_w * np.log1p(recent_mentions['user_follower_count'])) *
            (1 + like_w * recent_mentions['like_count']) *
            (1 + retweet_w * recent_mentions['retweet_count']) *
            (1 + quote_w * recent_mentions['quote_count']) *
            (1 + reply_w * recent_mentions['reply_count']) *
            (1 + bookmark_w * recent_mentions['bookmark_count']) *
            (1 + impression_w * recent_mentions['impression_count'])
        )
        
        # Get top tweets for each project
        top_tweets = recent_mentions.sort_values('weighted_score', ascending=False)
        top_tweets_by_project = top_tweets.groupby('project_id').head(5)
        
        # Store in database
        try:
            with self.mindshare_db_conn.cursor() as cursor:
                for _, row in top_tweets_by_project.iterrows():
                    cursor.execute("""
                        INSERT INTO influential_tweets 
                        (project_id, tweet_id, weighted_score, created_at, recorded_at)
                        VALUES (%s, %s, %s, %s, %s)
                        ON CONFLICT (project_id, tweet_id) DO UPDATE 
                        SET weighted_score = EXCLUDED.weighted_score,
                            recorded_at = EXCLUDED.recorded_at
                    """, (
                        int(row['project_id']),
                        str(row['tweet_id']),
                        float(row['weighted_score']),
                        row['tweet_timestamp'],
                        timestamp
                    ))
                
                self.mindshare_db_conn.commit()
        except Exception as e:
            self.mindshare_db_conn.rollback()
            logger.error(f"Error updating influential tweets: {e}")
    
    def recalculate_historical_mindshare(
        self, 
        new_weight_id: int, 
        days_to_recalculate: int = 7
    ) -> bool:
        """
        Recalculate historical mindshare based on new weights.
        
        Args:
            new_weight_id: ID of the new weight configuration
            days_to_recalculate: Number of days of history to recalculate
            
        Returns:
            Boolean indicating success or failure
        """
        try:
            logger.info(f"Recalculating historical mindshare for past {days_to_recalculate} days")
            
            with self.mindshare_db_conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cursor:
                cursor.execute("""
                    SELECT * FROM engagement_weights WHERE weight_id = %s
                """, (new_weight_id,))
                
                new_weights = cursor.fetchone()
                if not new_weights:
                    raise ValueError(f"Weight configuration {new_weight_id} not found")
                
                new_weights = dict(new_weights)
            
            end_time = datetime.now()
            start_time = end_time - timedelta(days=days_to_recalculate)
            
            # For recent data (last 24 hours), recalculate every hour
            if days_to_recalculate <= 1:
                # Get hourly timestamps
                timestamps = [
                    end_time - timedelta(hours=i)
                    for i in range(0, 24)
                ]
            else:
                # For older data, recalculate daily
                timestamps = [
                    end_time - timedelta(days=i)
                    for i in range(0, days_to_recalculate)
                ]
            
            # Get all mentions for the period
            mentions_df, _ = self.get_recent_mentions(days=days_to_recalculate)
            
            if len(mentions_df) == 0:
                logger.warning("No mentions found for recalculation")
                return True
            
            # First, remove existing snapshots for this weight configuration
            try:
                with self.mindshare_db_conn.cursor() as cursor:
                    cursor.execute("""
                        DELETE FROM mindshare_snapshots
                        WHERE weight_config_id = %s AND snapshot_timestamp BETWEEN %s AND %s
                    """, (new_weight_id, start_time, end_time))
                    self.mindshare_db_conn.commit()
            except Exception as e:
                self.mindshare_db_conn.rollback()
                logger.error(f"Error deleting existing snapshots: {e}")
                return False
            
            # Recalculate for each timestamp
            for ts in timestamps:
                # Filter mentions relevant to this timestamp
                relevant_mentions = mentions_df[
                    pd.to_datetime(mentions_df['created_at']) <= ts
                ].copy()
                
                if len(relevant_mentions) == 0:
                    continue
                
                # Calculate weighted mentions
                project_weights = self.calculate_weighted_mentions(
                    relevant_mentions, new_weights, ts
                )
                
                # Calculate percentages
                total_weighted_mentions = project_weights['weighted_mentions'].sum()
                if total_weighted_mentions > 0:
                    project_weights['mindshare_percentage'] = (
                        project_weights['weighted_mentions'] / total_weighted_mentions * 100
                    )
                else:
                    project_weights['mindshare_percentage'] = 0
                
                # Add rankings
                project_weights = project_weights.sort_values('mindshare_percentage', ascending=False)
                project_weights['ranking'] = range(1, len(project_weights) + 1)
                
                # Store results in database
                self.store_mindshare_results(project_weights, new_weight_id, ts)
                
                # If this is the most recent timestamp, update cache
                if ts >= end_time - timedelta(hours=1):
                    self.update_redis_cache(project_weights, new_weight_id, ts)
            
            logger.info("Historical recalculation completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error recalculating historical mindshare: {str(e)}", exc_info=True)
            return False

    def cleanup_old_data(self, retention_days: Dict[str, int]) -> None:
        """
        Clean up old data according to retention policies.
        
        Args:
            retention_days: Dictionary mapping table names to retention days
        """
        try:
            logger.info("Running data cleanup")
            
            with self.mindshare_db_conn.cursor() as cursor:
                for table, days in retention_days.items():
                    if table == 'tweet_mentions':
                        cursor.execute("""
                            DELETE FROM tweet_mentions
                            WHERE created_at < NOW() - INTERVAL '%s days'
                        """, (days,))
                    elif table == 'mindshare_snapshots':
                        cursor.execute("""
                            DELETE FROM mindshare_snapshots
                            WHERE snapshot_time < NOW() - INTERVAL '%s days'
                        """, (days,))
                    elif table == 'influential_tweets':
                        cursor.execute("""
                            DELETE FROM influential_tweets
                            WHERE recorded_at < NOW() - INTERVAL '%s days'
                        """, (days,))
                
                self.mindshare_db_conn.commit()
                logger.info("Data cleanup completed")
        
        except Exception as e:
            self.mindshare_db_conn.rollback()
            logger.error(f"Error during data cleanup: {str(e)}", exc_info=True)

def main():
    db_config = os.getenv("MINDSHARE_DB_URL")
    redis_config = None
    calculator = MindshareCalculator(redis_config)
    calculator.calculate_mindshare()

if __name__ == "__main__":
    main()
    # import argparse
    # import configparser
    
    # Parse command line arguments
    # parser = argparse.ArgumentParser(description="Run mindshare calculations")
    # parser.add_argument("--config", default="config.ini", help="Path to config file")
    # parser.add_argument("--recalculate", type=int, help="Weight ID to use for recalculation")
    # parser.add_argument("--days", type=int, default=7, help="Days to recalculate")
    # parser.add_argument("--cleanup", action="store_true", help="Run data cleanup")
    # args = parser.parse_args()
    
    # Load configuration
    # config = configparser.ConfigParser()
    # config.read(args.config)
    
    # db_config = {
    #     "host": config.get("Database", "host"),
    #     "database": config.get("Database", "database"),
    #     "user": config.get("Database", "user"),
    #     "password": config.get("Database", "password")
    # }
    
    # redis_config = {
    #     "host": config.get("Redis", "host"),
    #     "port": config.getint("Redis", "port"),
    #     "db": config.getint("Redis", "db")
    # }
    
    # Initialize calculator
    # db_config = os.getenv("MINDSHARE_DB_URL")
    # redis_config = None
    # calculator = MindshareCalculator(db_config, redis_config)
    
    # Run the appropriate operation
    # if args.recalculate:
    #     success = calculator.recalculate_historical_mindshare(args.recalculate, args.days)
    #     exit(0 if success else 1)
    # elif args.cleanup:
    #     retention_policy = {
    #         "tweet_mentions": config.getint("Retention", "tweet_mentions", fallback=90),
    #         "mindshare_snapshots": config.getint("Retention", "mindshare_snapshots", fallback=30),
    #         "influential_tweets": config.getint("Retention", "influential_tweets", fallback=30)
    #     }
    #     calculator.cleanup_old_data(retention_policy)
    # else:
    #     success = calculator.calculate_mindshare()
    #     exit(0 if success else 1)