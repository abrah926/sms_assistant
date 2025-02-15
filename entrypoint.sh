#!/bin/bash

echo "Environment variables in container:"
env | grep PG

echo "Testing database connection..."
psql "postgresql://${PGUSER}:${PGPASSWORD}@${PGHOST}:${PGPORT}/${PGDATABASE}?sslmode=${PGSSLMODE}&sslrootcert=${PGSSLROOTCERT}"

exec "$@" 