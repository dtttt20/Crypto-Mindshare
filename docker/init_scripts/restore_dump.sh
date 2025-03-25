#!/bin/bash
set -e

if [ -f /data/tweet_archive.bak ]; then
  echo "Found dump file. Restoring database..."
  pg_restore -U postgres -d tweet_archive -v /data/tweet_archive.bak
  echo "Database restore completed!"
else
  echo "No dump file found at /data/tweet_archive.bak. Skipping restore."
fi