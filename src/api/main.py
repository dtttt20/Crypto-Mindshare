from fastapi import FastAPI, HTTPException, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
import redis
import psycopg
from psycopg.rows import dict_row
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta, timezone
import os
import logging
import sys
from contextlib import contextmanager
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(format="%(asctime)s %(levelname)s %(process)d: \
                    %(filename)s:%(lineno)d %(message)s", level=logging.INFO,
                    stream=sys.stdout)
logger = logging.getLogger("fastapi")

app = FastAPI(
    title="Crypto Mindshare API",
    description="API for tracking cryptocurrency project mindshare on social media",
    version="0.0.1"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DB_CONFIG = {
    "url": os.environ.get("MINDSHARE_DB_URL")
}

REDIS_CONFIG = {
    "host": os.environ.get("REDIS_HOST", "localhost"),
    "port": int(os.environ.get("REDIS_PORT", 6379)),
    "db": int(os.environ.get("REDIS_DB", 0)),
}

@contextmanager
def get_db_connection():
    conn = psycopg.connect(DB_CONFIG["url"])
    try:
        yield conn
    finally:
        conn.close()

@contextmanager
def get_redis_connection():
    r = redis.Redis(**REDIS_CONFIG)
    try:
        yield r
    finally:
        r.close()

def get_active_weight_id():
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT config_id FROM weight_config WHERE is_active = TRUE")
            result = cur.fetchone()
            if not result:
                raise HTTPException(status_code=500, detail="No active weight configuration found")
            return result[0]

class Project(BaseModel):
    project_id: int
    project_name: str
    # token_symbol: str

class InfluentialTweet(BaseModel):
    tweet_timestamp: datetime
    project_id: int
    tweet_id: str
    weighted_score: float
    tweet_text: str

class MindshareRanking(BaseModel):
    project_id: int
    project_name: str
    # token_symbol: str
    ranking: int
    mindshare_percentage: float

class MindshareHistory(BaseModel):
    project_id: int
    project_name: str
    # token_symbol: str
    data: List[Dict[str, Any]]
    
class MindshareChange(BaseModel):
    project_id: int
    project_name: str
    # token_symbol: str
    current_percentage: float
    previous_percentage: float
    absolute_change: float
    percentage_change: float
    
class WeightConfig(BaseModel):
    config_id: int
    name: str
    description: str
    is_active: bool

@app.get("/projects", response_model=List[Project], tags=["Projects"])
async def get_projects():
    """
    Get a list of all tracked cryptocurrency projects.
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute("""
                    SELECT project_id, project_name
                    FROM projects
                    WHERE is_active = TRUE
                    ORDER BY project_name
                """)
                projects = cur.fetchall()
                return [dict(project) for project in projects]
    except Exception as e:
        logger.error(f"Error fetching projects: {str(e)}")
        raise HTTPException(status_code=500, detail="Database error")
    
@app.get("/projects/influential_tweets/{project_id}", response_model=Dict[str, List[InfluentialTweet]], tags=["Projects"])
async def get_influential_tweets(
    project_id: int,
    start_date: datetime = Query(..., description="Start date for the range to fetch tweets"),
    end_date: Optional[datetime] = Query(datetime.now(timezone.utc), description="End date for the range to fetch tweets (defaults to today)")
):
    """
    Get influential tweets for a specific project grouped by day for the specified date range.
    Returns a dictionary with dates as keys and lists of tweets as values.
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute("""
                    SELECT 
                        created_at,
                        project_id,
                        tweet_id,
                        weighted_score,
                        tweet_text
                    FROM influential_tweets
                    WHERE project_id = %s
                    AND created_at BETWEEN %s AND %s
                    ORDER BY weighted_score DESC, created_at DESC
                """, (project_id, start_date, end_date))
                tweets = cur.fetchall()
                
                if not tweets:
                    raise HTTPException(status_code=404, detail="No influential tweets found")
                
                tweets_by_day = {}
                for tweet in tweets:
                    day_key = tweet["created_at"].strftime('%Y-%m-%d')
                    
                    if day_key not in tweets_by_day:
                        tweets_by_day[day_key] = []
                    
                    tweets_by_day[day_key].append({
                        "tweet_timestamp": tweet["created_at"],
                        "project_id": tweet["project_id"],
                        "tweet_id": tweet["tweet_id"],
                        "weighted_score": tweet["weighted_score"],
                        "tweet_text": tweet["tweet_text"] or "No text available"
                    })
                
                sorted_tweets_by_day = {
                    k: tweets_by_day[k] 
                    for k in sorted(tweets_by_day.keys(), reverse=True)
                }
                
                return sorted_tweets_by_day
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error fetching influential tweets: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching influential tweets {str(e)}")
    
@app.get("/projects/search/{project_name}", response_model=List[Project], tags=["Projects"])
async def search_projects(
    project_name: str,
    limit: int = Query(10, ge=1, le=100, description="Number of results to return")
):
    """
    Search for projects by name.
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute("""
                    SELECT project_id, project_name
                    FROM projects
                    WHERE project_name ILIKE %s
                    ORDER BY project_name
                    LIMIT %s
                """, (f"%{project_name}%", limit))
                projects = cur.fetchall()
                if projects:
                    return [dict(project) for project in projects]
                else:
                    cur.execute("""
                        SELECT alias as project_name,
                               project_id
                        FROM project_aliases
                        WHERE alias ILIKE %s
                        ORDER BY alias
                        LIMIT %s
                    """, (f"%{project_name}%", limit))
                    project_aliases = cur.fetchall()
                    if project_aliases:
                        for alias in project_aliases:
                            projects = []
                            cur.execute("""
                                SELECT project_id, project_name
                                FROM projects
                                WHERE project_id = %s
                            """, (alias['project_id'],))
                            project_details = cur.fetchone()
                            if project_details:
                                projects.append(dict(project_details))
                        return projects
                    else:
                        raise HTTPException(status_code=404, detail="No projects found")
    except Exception as e:
        logger.error(f"Error searching projects: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error searching projects {str(e)}")
                
                
@app.get("/mindshare/ranking", response_model=List[MindshareRanking], tags=["Mindshare"])
async def get_mindshare_ranking(
    limit: int = Query(100, ge=1, le=1000, description="Number of projects to return")
    # weight_id: Optional[int] = Query(None, description="Weight configuration ID (defaults to active)")
):
    """
    Get current mindshare ranking for top projects.
    """
    try:
        weight_id = None
        if weight_id is None:
            weight_id = get_active_weight_id()
        
        with get_redis_connection() as redis_conn:
            timestamp = datetime.now()
            cache_key = f"mindshare_ranking:{weight_id}:{timestamp.strftime('%Y-%m-%d-%H')}"
            
            cached_ranking = redis_conn.zrevrange(cache_key, 0, limit-1, withscores=True)
            
            if cached_ranking:
                project_ids = [int(pid) for pid, _ in cached_ranking]
                
                with get_db_connection() as db_conn:
                    with db_conn.cursor(row_factory=dict_row) as cur:
                        placeholders = ','.join(['%s'] * len(project_ids))
                        cur.execute(f"""
                            SELECT project_id, project_name 
                            FROM projects 
                            WHERE project_id IN ({placeholders})
                        """, project_ids)
                        
                        projects = {row['project_id']: dict(row) for row in cur.fetchall()}
                
                # Combine data
                result = []
                for i, (pid_str, score) in enumerate(cached_ranking, 1):
                    pid = int(pid_str)
                    if pid in projects:
                        result.append({
                            **projects[pid],
                            "ranking": i,
                            "mindshare_percentage": round(score, 3)
                        })
                
                return result
        
        # If not in cache, fetch from database
        with get_db_connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute("""
                    SELECT p.project_id, p.project_name, 
                           ms.ranking, ms.mindshare_percentage
                    FROM mindshare_snapshots ms
                    JOIN projects p ON ms.project_id = p.project_id
                    WHERE ms.weight_config_id = %s
                    AND ms.snapshot_timestamp = (
                        SELECT MAX(snapshot_timestamp) FROM mindshare_snapshots 
                        WHERE weight_config_id = %s
                    )
                    ORDER BY ms.ranking
                    LIMIT %s
                """, (weight_id, weight_id, limit))
                
                results = cur.fetchall()
                return [dict(row) for row in results]
                
    except Exception as e:
        logger.error(f"Error fetching mindshare ranking: {str(e)}")
        raise HTTPException(status_code=500, detail="Error fetching mindshare ranking")

@app.get("/mindshare/project/{project_id}", response_model=MindshareHistory, tags=["Mindshare"])
async def get_project_mindshare(
    project_id: int,
    days: int = Query(1, ge=1, le=30, description="Number of days of history"),
    weight_id: Optional[int] = Query(None, description="Weight configuration ID (defaults to active)")
):
    """
    Get mindshare history for a specific project.
    """
    try:
        if weight_id is None:
            weight_id = get_active_weight_id()
            logger.info(f"Using active weight_id: {weight_id}")
            
        with get_db_connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute("""
                    SELECT project_id, project_name
                    FROM projects 
                    WHERE project_id = %s
                """, (project_id,))
                
                project = cur.fetchone()
                if not project:
                    raise HTTPException(status_code=404, detail="Project not found")
                
                end_date = datetime.now(timezone.utc)
                end_date_historical = end_date - timedelta(days=1)
                start_date = end_date - timedelta(days=days)
                
                logger.info(f"Fetching mindshare history for project_id={project_id}, weight_id={weight_id}")
                logger.info(f"Date range: {start_date} to {end_date}")

                today_data = []
                historical_data = []
                cur.execute("""
                    SELECT 
                        snapshot_timestamp,
                        mindshare_percentage,
                        ranking
                    FROM mindshare_snapshots
                    WHERE project_id = %s
                    AND weight_config_id = %s
                    AND snapshot_timestamp BETWEEN %s AND %s
                    ORDER BY mindshare_snapshots.snapshot_timestamp DESC
                    LIMIT 1
                """, (project_id, weight_id, start_date, end_date))
                today_data = cur.fetchall()

                if days > 1:
                    cur.execute("""
                        SELECT
                            snapshot_timestamp,
                            avg_daily_mindshare AS mindshare_percentage,
                            ranking
                        FROM mindshare_daily_snapshots
                        WHERE project_id = %s
                        AND weight_config_id = %s
                        AND snapshot_timestamp BETWEEN %s AND %s
                        ORDER BY snapshot_timestamp DESC
                    """, (project_id, weight_id, start_date, end_date_historical))
                    historical_data = cur.fetchall()
                
                if today_data and historical_data:
                    history = today_data + historical_data
                elif today_data:
                    history = today_data
                elif historical_data:
                    history = historical_data
                else:
                    history = []
                
                logger.info(f"Query returned {len(history)} records")
                history_output = []
                if history:
                    for row in history:
                        history_output.append(
                            {
                                "time_logged": row["snapshot_timestamp"].isoformat() if hasattr(row["snapshot_timestamp"], 'isoformat') else row["snapshot_timestamp"],
                                "mindshare_percentage": float(row["mindshare_percentage"]),
                                "ranking": int(row["ranking"]) if row["ranking"] is not None else None
                            }
                        )
                else:
                    history_output = []
                
                return {
                    "project_id": project["project_id"],
                    "project_name": project["project_name"],
                    "data": history_output
                }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching project mindshare: {str(e)}")
        raise HTTPException(status_code=500, detail="Error fetching project mindshare")

@app.get("/mindshare/gainers", response_model=List[MindshareChange], tags=["Mindshare"])
async def get_mindshare_gainers(
    period: str = Query("24h", description="Time period (24h, 7d, 30d)"),
    limit: int = Query(10, ge=1, le=100, description="Max number of results to return"),
    weight_id: Optional[int] = Query(None, description="Weight configuration ID (defaults to active)")
):
    """
    Get top mindshare gainers for the specified period.
    """
    return await get_mindshare_changes(period, limit, weight_id, is_gainer=True)

@app.get("/mindshare/losers", response_model=List[MindshareChange], tags=["Mindshare"])
async def get_mindshare_losers(
    period: str = Query("24h", description="Time period (24h, 7d, 30d)"),
    limit: int = Query(10, ge=1, le=100, description="Number of results to return"),
    weight_id: Optional[int] = Query(None, description="Weight configuration ID (defaults to active)")
):
    """
    Get top mindshare losers for the specified period.
    """
    return await get_mindshare_changes(period, limit, weight_id, is_gainer=False)

async def get_mindshare_changes(period: str, limit: int, weight_id: Optional[int], is_gainer: bool):
    """
    Helper function to get mindshare gainers or losers.
    
    Uses Redis cache as primary source and falls back to database only if cache is empty.
    """
    try:
        period_days = {
            "24h": 1,
            "7d": 7,
            "30d": 30
        }.get(period, 1)
        
        if weight_id is None:
            weight_id = get_active_weight_id()
            
        with get_redis_connection() as redis_conn:
            cache_key = f"mindshare_{'gainers' if is_gainer else 'losers'}_{period_days}d:{weight_id}"
            logger.info(f"Checking Redis cache for key: {cache_key}")
            
            cached_data = redis_conn.zrange(cache_key, 0, limit-1, withscores=True, desc=True)
            logger.info(f"Cached data fetched: {cached_data}")

            cached_data = []
            
            if cached_data:
                logger.info(f"Found {len(cached_data)} entries in Redis cache")
                project_ids = [int(pid) for pid, _ in cached_data]
                
                with get_db_connection() as db_conn:
                    with db_conn.cursor(row_factory=dict_row) as cur:
                        placeholders = ','.join(['%s'] * len(project_ids))
                        cur.execute(f"""
                            SELECT project_id, project_name
                            FROM projects 
                            WHERE project_id IN ({placeholders})
                        """, project_ids)
                        
                        projects = {row['project_id']: dict(row) for row in cur.fetchall()}
                
                result = []
                for pid_str, change_value in cached_data:
                    pid = int(pid_str)
                    if pid in projects:
                        current_key = f"project_mindshare:{pid}:{weight_id}"
                        today_str = datetime.now().strftime("%Y-%m-%d")
                        previous_date = (datetime.now() - timedelta(days=period_days)).strftime("%Y-%m-%d")
                        
                        current_pct = redis_conn.hget(current_key, today_str)
                        previous_pct = redis_conn.hget(current_key, previous_date)
                        
                        if current_pct is not None:
                            current = float(current_pct)
                            previous = float(previous_pct or 0)
                            
                            result.append({
                                **projects[pid],
                                "current_percentage": current,
                                "previous_percentage": previous,
                                "absolute_change": current - previous,
                                "percentage_change": float(change_value) * (-1 if not is_gainer else 1)
                            })
                
                if result:
                    result.sort(key=lambda x: abs(x['percentage_change']), reverse=True)
                    return result
                logger.info("Redis cache data incomplete, falling back to database")
        
        logger.info("Fetching mindshare changes from database")
        with get_db_connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                comparison = '>' if is_gainer else '<'
                cur.execute(f"""
                    WITH current_data AS (
                        SELECT project_id, mindshare_percentage
                        FROM mindshare_snapshots
                        WHERE weight_config_id = %s
                        AND snapshot_timestamp = (
                            SELECT MAX(snapshot_timestamp) FROM mindshare_snapshots
                            WHERE weight_config_id = %s
                        )
                    ),
                    previous_data AS (
                        SELECT project_id, mindshare_percentage
                        FROM mindshare_snapshots
                        WHERE weight_config_id = %s
                        AND snapshot_timestamp = (
                            SELECT MAX(snapshot_timestamp) FROM mindshare_snapshots
                            WHERE weight_config_id = %s
                            AND snapshot_timestamp < NOW() - INTERVAL '{period_days} days'
                        )
                    ),
                    changes AS (
                        SELECT 
                            p.project_id,
                            p.project_name,
                            c.mindshare_percentage AS current_percentage,
                            COALESCE(prev.mindshare_percentage, 0) AS previous_percentage,
                            (c.mindshare_percentage - COALESCE(prev.mindshare_percentage, 0)) AS absolute_change,
                            CASE 
                                WHEN COALESCE(prev.mindshare_percentage, 0) = 0 THEN 0
                                ELSE (c.mindshare_percentage - prev.mindshare_percentage) 
                                      / NULLIF(prev.mindshare_percentage, 0) * 100
                            END AS percentage_change
                        FROM projects p
                        JOIN current_data c ON p.project_id = c.project_id
                        LEFT JOIN previous_data prev ON p.project_id = prev.project_id
                        WHERE (c.mindshare_percentage - COALESCE(prev.mindshare_percentage, 0)) {comparison} 0
                    )
                    SELECT *, ABS(percentage_change) AS sort_metric FROM changes
                    ORDER BY sort_metric DESC NULLS LAST
                    LIMIT %s
                """, (weight_id, weight_id, weight_id, weight_id, limit))
                
                results = cur.fetchall()
                return [dict(row) for row in results]
    except Exception as e:
        logger.error(f"Error fetching mindshare {'gainers' if is_gainer else 'losers'}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching mindshare {'gainers' if is_gainer else 'losers'}")

@app.get("/weights", response_model=List[WeightConfig], tags=["Configuration"])
async def get_weight_configurations():
    """
    Get all available weight configurations.
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute("""
                    SELECT
                        config_id,
                        name,
                        description,
                        is_active
                     FROM weight_config
                    ORDER BY is_active DESC, created_at DESC
                """)
                
                configs = cur.fetchall()
                return [dict(config) for config in configs]
    
    except Exception as e:
        logger.error(f"Error fetching weight configurations: {str(e)}")
        raise HTTPException(status_code=500, detail="Error fetching weight configurations")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)