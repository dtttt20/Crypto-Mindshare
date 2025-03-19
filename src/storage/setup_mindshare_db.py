import psycopg
import os
import logging
import sys
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(format="%(asctime)s %(levelname)s %(process)d: \
                    %(filename)s:%(lineno)d %(message)s", level=logging.INFO,
                    stream=sys.stdout)
logger = logging.getLogger("setup_mindshare_db")

class MindshareDB:
    """Database interface for the mindshare database"""
    def __init__(self, db_name):
        self.mindshare_db_conn = psycopg.connect(
            os.getenv("MINDSHARE_DEFAULT_DB_URL"), autocommit=True
        )
        self.create_database(db_name)
        self.connect_to_database(db_name)

    def create_database(self, db_name):
        """Create a new database if it doesn't exist"""
        try:
            with self.mindshare_db_conn.cursor() as cur:
                cur.execute("SELECT 1 FROM pg_database WHERE datname = "
                            f"'{db_name}'")
                exists = cur.fetchone()

                if not exists:
                    logger.info(f"Database {db_name} does not exist, "
                                f"creating it")
                    cur.execute(f"CREATE DATABASE {db_name}")
                    logger.info(f"Database {db_name} successfully created")
                else:
                    logger.info(f"Database {db_name} exists")
                cur.close()
        except psycopg.Error as e:
            logger.error(f"Error creating database {db_name}: {e}")

    def connect_to_database(self, db_name):
        """Connect to the specified database and enable TimescaleDB"""
        try:
            self.mindshare_db_conn.close()

            db_url = os.getenv("MINDSHARE_DB_URL")

            self.mindshare_db_conn = psycopg.connect(db_url)
            logger.info(f"Connected to database {db_name}")

            with self.mindshare_db_conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS timescaledb;")
                self.mindshare_db_conn.commit()
                logger.info("TimescaleDB extension enabled")
        except psycopg.Error as e:
            if hasattr(self, 'mindshare_db_conn') and self.mindshare_db_conn:
                self.mindshare_db_conn.rollback()
            logger.error(f"Error connecting to database {db_name}: {e}")


    def create_tables(self):
        """Create all required tables for the database"""
        create_projects_table = """
            CREATE TABLE IF NOT EXISTS projects (
                project_id SERIAL PRIMARY KEY,
                project_name VARCHAR(100) NOT NULL,
                description TEXT,
                is_active BOOLEAN DEFAULT FALSE
            );

            CREATE UNIQUE INDEX IF NOT EXISTS idx_project_name ON projects (project_name);
            """
        
        create_project_aliases_table = """
            CREATE TABLE IF NOT EXISTS project_aliases (
                alias_id SERIAL PRIMARY KEY,
                project_id INTEGER NOT NULL REFERENCES projects(project_id),
                alias VARCHAR(100) NOT NULL,
                created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
            );

            CREATE UNIQUE INDEX idx_project_alias ON project_aliases(project_id, alias);
            CREATE INDEX idx_alias ON project_aliases(alias);
        """

        create_tokens_table = """
            CREATE TABLE IF NOT EXISTS tokens (
                token_id SERIAL PRIMARY KEY,
                token_name VARCHAR(100) NOT NULL,
                project_id INTEGER NOT NULL REFERENCES projects(project_id),
                token_symbol VARCHAR(30)
            );

            CREATE UNIQUE INDEX idx_token_name ON tokens(project_id, token_name);
            CREATE UNIQUE INDEX idx_token_symbol ON tokens(project_id, token_symbol);

        """

        create_mentions_table = """
            CREATE TABLE IF NOT EXISTS mentions (
                tweet_timestamp TIMESTAMPTZ NOT NULL,
                project_id INTEGER NOT NULL REFERENCES projects(project_id),
                like_count INTEGER,
                quote_count INTEGER,
                reply_count INTEGER,
                retweet_count INTEGER,
                bookmark_count INTEGER,
                impression_count BIGINT,
                tweet_id VARCHAR(30) NOT NULL,
                user_id VARCHAR(30) NOT NULL,
                tweet_text TEXT NOT NULL,
                CONSTRAINT mentions_pkey PRIMARY KEY (tweet_timestamp, project_id, tweet_id)
            );
        
            CREATE INDEX idx_tweet_mentions_project_id ON mentions(project_id);
            CREATE INDEX idx_tweet_mentions_tweet_id ON mentions(tweet_id);
            CREATE INDEX idx_tweet_mentions_user_id ON mentions(user_id);
        """
        create_mentions_hypertable = """
            SELECT create_hypertable(
                'mentions',
                'tweet_timestamp',
                if_not_exists => TRUE
            );
        """

        create_user_influence_table = """
            CREATE TABLE IF NOT EXISTS user_influence (
                id SERIAL PRIMARY KEY,
                user_id VARCHAR(30) NOT NULL,
                follower_count INTEGER,
                average_engagement BIGINT,
                total_posts INTEGER,
                last_updated TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                influence_score FLOAT
            );

            CREATE INDEX idx_user_influence_score ON user_influence(influence_score DESC);
            CREATE UNIQUE INDEX idx_user_influence_user_id ON user_influence(user_id);
        """

        create_weight_config_table = """
            CREATE TABLE IF NOT EXISTS weight_config (
                config_id SERIAL PRIMARY KEY,
                name VARCHAR(100) NOT NULL,
                weights JSONB NOT NULL,
                is_active BOOLEAN DEFAULT FALSE,
                description TEXT,
                created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
            );

            CREATE UNIQUE INDEX idx_active_weight ON weight_config(is_active) WHERE is_active = TRUE;
        """

        create_mindshare_snapshots_table = """
            CREATE TABLE IF NOT EXISTS mindshare_snapshots (
                snapshot_id SERIAL,
                project_id INTEGER NOT NULL REFERENCES projects(project_id),
                weight_config_id INTEGER NOT NULL REFERENCES weight_config(config_id),
                mindshare_percentage FLOAT NOT NULL,
                snapshot_timestamp TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                total_weighted_mentions BIGINT NOT NULL,
                ranking INTEGER
            );

            CREATE INDEX idx_mindshare_snapshots_project_id ON mindshare_snapshots(project_id);
            CREATE INDEX idx_mindshare_snapshots_ranking ON mindshare_snapshots(ranking);
        """

        create_mindshare_snapshots_hypertable = """
            SELECT create_hypertable(
                'mindshare_snapshots',
                'snapshot_timestamp',
                if_not_exists => TRUE
            );
        """

        create_influential_tweets_table = """
            CREATE TABLE IF NOT EXISTS influential_tweets (
                id SERIAL PRIMARY KEY,
                project_id INTEGER NOT NULL REFERENCES projects(project_id),
                tweet_id VARCHAR(30) NOT NULL,
                weighted_score FLOAT NOT NULL,
                tweet_text TEXT NOT NULL,
                created_at TIMESTAMPTZ NOT NULL,
                recorded_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                CONSTRAINT unique_project_tweet UNIQUE (project_id, tweet_id)
            );

            CREATE INDEX idx_influential_tweets_project ON influential_tweets(project_id);
            CREATE INDEX idx_influential_tweets_score ON influential_tweets(weighted_score DESC);
        """

        create_continuous_daily_mindshare_aggregates_table = """
            CREATE MATERIALIZED VIEW daily_mindshare
            WITH (timescaledb.continuous) AS
            SELECT
                project_id,
                weight_config_id,
                time_bucket('1 day', snapshot_timestamp) AS day,
                AVG(mindshare_percentage) AS avg_daily_mindshare,
                MAX(mindshare_percentage) AS max_daily_mindshare,
                MIN(mindshare_percentage) AS min_daily_mindshare

                FROM mindshare_snapshots
            GROUP BY project_id, weight_config_id, time_bucket('1 day', snapshot_timestamp);

        """

        create_daily_mindshare_aggregates_index = """
            CREATE INDEX idx_daily_mindshare ON daily_mindshare(day, project_id, weight_config_id);
        """

        create_daily_mindshare_aggregate_snapshot_table = """
            CREATE TABLE IF NOT EXISTS mindshare_daily_snapshots (
                snapshot_id SERIAL PRIMARY KEY,
                project_id INTEGER NOT NULL REFERENCES projects(project_id),
                weight_config_id INTEGER NOT NULL REFERENCES weight_config(config_id),
                avg_daily_mindshare FLOAT NOT NULL,
                max_daily_mindshare FLOAT,
                min_daily_mindshare FLOAT,
                total_mentions BIGINT NOT NULL,
                total_weighted_mentions BIGINT NOT NULL,
                engagement_summary TEXT,
                ranking INTEGER,
                snapshot_timestamp TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
            );

            CREATE INDEX idx_mindshare_snapshots_by_date ON mindshare_daily_snapshots(snapshot_timestamp, avg_daily_mindshare DESC);
            CREATE INDEX idx_mindshare_snapshots_by_project ON mindshare_daily_snapshots(project_id, snapshot_timestamp);
"""
        

        with self.mindshare_db_conn.cursor() as cur:
            cur.execute(create_projects_table)
            logger.info("Projects table created")
            cur.execute(create_project_aliases_table)
            logger.info("Project aliases table created")
            cur.execute(create_tokens_table)
            logger.info("Tokens table created")
            cur.execute(create_mentions_table)
            logger.info("Minute metrics table created")
            cur.execute(create_mentions_hypertable)
            logger.info("Mentions hypertable created")
            cur.execute(create_user_influence_table)
            logger.info("User influence table created")
            cur.execute(create_weight_config_table)
            logger.info("Weight config table created")
            cur.execute(create_mindshare_snapshots_table)
            logger.info("Mindshare snapshots table created")
            cur.execute(create_mindshare_snapshots_hypertable)
            logger.info("Mindshare snapshots hypertable created")
            cur.execute(create_influential_tweets_table)
            logger.info("Influential tweets table created")
            cur.execute(create_daily_mindshare_aggregate_snapshot_table)
            logger.info("Daily mindshare aggregate snapshot table created")
            
            self.mindshare_db_conn.commit()
            cur.close()
            self.mindshare_db_conn.close()
            
            with psycopg.connect(os.getenv("MINDSHARE_DB_URL"), autocommit=True) as autocommit_conn:
                with autocommit_conn.cursor() as cur:
                    cur.execute(create_continuous_daily_mindshare_aggregates_table)
                    logger.info("Continuous daily mindshare aggregates table created")
                    cur.execute(create_daily_mindshare_aggregates_index)
                    logger.info("Daily mindshare index created")
                cur.close()
            autocommit_conn.close()

if __name__ == "__main__":
    db = MindshareDB("mindshare")
    db.create_tables()