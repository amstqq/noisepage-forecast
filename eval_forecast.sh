#!/bin/env bash

# set -euxo pipefail

export DB_USER="forecast_user"
export DB_PASS="forecast_pass"
export DB_NAME="forecast_db"

export DIR_BUILD="./build"
export DIR_BUILD_BENCHBASE="./benchbase"

export BENCHBASE_URL="https://github.com/cmu-db/benchbase.git"
export BENCHMARK="tpcc"

export DIR_TRAIN="./artifacts/train"
export DIR_EVAL="./artifacts/eval"
export DIR_DUMP="./artifacts/dump"

export ROOT_DIR=$(pwd)

export ARTIFACT_CONFIG="${ROOT_DIR}/artifacts/${BENCHMARK}_config.xml"

rm -rf artifacts build
mkdir -p ${DIR_TRAIN}
mkdir -p ${DIR_EVAL}

sudo --validate
sudo --login -u postgres psql -c "drop database if exists ${DB_NAME};"
sudo --login -u postgres psql -c "drop user if exists ${DB_USER};"
sudo --login -u postgres psql -c "create user ${DB_USER} with superuser encrypted password '${DB_PASS}';"

PGPASSWORD=${DB_PASS} dropdb --if-exists --host=localhost --username=${DB_USER} ${DB_NAME}
PGPASSWORD=${DB_PASS} createdb --host=localhost --username=${DB_USER} ${DB_NAME}

# BenchBase: get, build, extract.
# Use a cached copy if available since we don't update BenchBase much.
# rm -rf ${DIR_BUILD_BENCHBASE}
if [ ! -d ${DIR_BUILD_BENCHBASE} ]
then
    git clone ${BENCHBASE_URL} --depth 1 --branch main --single-branch ${DIR_BUILD_BENCHBASE}
    cd ${DIR_BUILD_BENCHBASE}
    ./mvnw clean package -Dmaven.test.skip=true -P postgres
    cd -
    cd ${DIR_BUILD_BENCHBASE}/target
    tar xvzf benchbase-postgres.tgz
    cd -
fi

# Copy and modify BenchBase config.
cp ${DIR_BUILD_BENCHBASE}/target/benchbase-postgres/config/postgres/sample_${BENCHMARK}_config.xml ${ARTIFACT_CONFIG}
xmlstarlet edit --inplace --update '/parameters/url' --value "jdbc:postgresql://localhost:5432/${DB_NAME}?preferQueryMode=extended" ${ARTIFACT_CONFIG}
xmlstarlet edit --inplace --update '/parameters/username' --value "${DB_USER}" ${ARTIFACT_CONFIG}
xmlstarlet edit --inplace --update '/parameters/password' --value "${DB_PASS}" ${ARTIFACT_CONFIG}
xmlstarlet edit --inplace --update '/parameters/scalefactor' --value "1" ${ARTIFACT_CONFIG}
xmlstarlet edit --inplace --update '/parameters/works/work/time' --value "30" ${ARTIFACT_CONFIG}
xmlstarlet edit --inplace --update '/parameters/works/work/rate' --value "unlimited" ${ARTIFACT_CONFIG}

# ---
# Generate the forecast training data.
# ---

# Load the database.
cd ${DIR_BUILD_BENCHBASE}/target/benchbase-postgres
java -jar benchbase.jar -b ${BENCHMARK} -c ${ARTIFACT_CONFIG} --create=true --load=true
cd -

# Dump the database.
PGPASSWORD=${DB_PASS} pg_dump --host=localhost --username=${DB_USER} --format=directory --file=${DIR_DUMP} ${DB_NAME}

# Clear old log files.
sudo bash -c "rm -rf /var/lib/postgresql/14/main/log/*"

# Enable logging.
PGPASSWORD=${DB_PASS} psql --host=localhost --dbname=${DB_NAME} --username=${DB_USER} --command="ALTER SYSTEM SET log_destination='csvlog'"
PGPASSWORD=${DB_PASS} psql --host=localhost --dbname=${DB_NAME} --username=${DB_USER} --command="ALTER SYSTEM SET logging_collector='on'"
PGPASSWORD=${DB_PASS} psql --host=localhost --dbname=${DB_NAME} --username=${DB_USER} --command="ALTER SYSTEM SET log_statement='all'"
PGPASSWORD=${DB_PASS} psql --host=localhost --dbname=${DB_NAME} --username=${DB_USER} --command="ALTER SYSTEM SET log_connections='on'"
PGPASSWORD=${DB_PASS} psql --host=localhost --dbname=${DB_NAME} --username=${DB_USER} --command="ALTER SYSTEM SET log_disconnections='on'"
sudo systemctl restart postgresql
until PGPASSWORD=${DB_PASS} pg_isready --host=localhost --dbname=${DB_NAME} --username=${DB_USER} ; do sleep 1 ; done

# Run Benchbase.
cd ${DIR_BUILD_BENCHBASE}/target/benchbase-postgres
java -jar benchbase.jar -b ${BENCHMARK} -c ${ARTIFACT_CONFIG} --execute=true
cd -

# Copy the log files.
sudo bash -c "cat /var/lib/postgresql/14/main/log/*.csv > ${DIR_TRAIN}/query_log.csv"

# Disable logging.
PGPASSWORD=${DB_PASS} psql --host=localhost --dbname=${DB_NAME} --username=${DB_USER} --command="ALTER SYSTEM SET log_destination='stderr'"
PGPASSWORD=${DB_PASS} psql --host=localhost --dbname=${DB_NAME} --username=${DB_USER} --command="ALTER SYSTEM SET logging_collector='off'"
PGPASSWORD=${DB_PASS} psql --host=localhost --dbname=${DB_NAME} --username=${DB_USER} --command="ALTER SYSTEM SET log_statement='none'"
PGPASSWORD=${DB_PASS} psql --host=localhost --dbname=${DB_NAME} --username=${DB_USER} --command="ALTER SYSTEM SET log_connections='off'"
PGPASSWORD=${DB_PASS} psql --host=localhost --dbname=${DB_NAME} --username=${DB_USER} --command="ALTER SYSTEM SET log_disconnections='off'"
sudo systemctl restart postgresql
until PGPASSWORD=${DB_PASS} pg_isready --host=localhost --dbname=${DB_NAME} --username=${DB_USER} ; do sleep 1 ; done

# Restore from dump.
PGPASSWORD=${DB_PASS} pg_restore --host=localhost --username=${DB_USER} --clean --if-exists --dbname=${DB_NAME} ${DIR_DUMP}

# ---
# Our pipeline.
# ---

# Run pipeline to create a forecast.
# Ideally, we would then create a query log to replay at this point, but pgreplay barfs too hard.
python3 ./convert_postgresql_to_split_postgresql.py
python3 ./convert_split_postgresql_to_parquet.py
python3 ./convert_parquet_to_split_parquet.py
python3 ./convert_parquet_to_forecast.py

# ---
# Evaluation.
# ---

# # Compress the train log.
# pgreplay -f -c -o "${DIR_EVAL}/pgreplay_train.out" "./artifacts/tmp/data/train.csv"
# # Compress the future log.
# pgreplay -f -c -o "${DIR_EVAL}/pgreplay_future.out" "./artifacts/tmp/data/future.csv"

# # --
# # Forecast log.
# # ---

# # Restore from dump.
# PGPASSWORD=${DB_PASS} pg_restore --host=localhost --username=${DB_USER} --clean --if-exists --dbname=${DB_NAME} ${DIR_DUMP}
# sudo systemctl restart postgresql
# # Replay the forecast log.
# PGUSER=${DB_USER} pgreplay -r -h localhost -p 5432 -W ${DB_PASS} "${DIR_EVAL}/pgreplay_train.out"
# python3 ./convert_forecast_to_query_log.py
# # Capture metrics.
# PGPASSWORD=${DB_PASS} psql --host=localhost --dbname=${DB_NAME} --username=${DB_USER} --command="CREATE EXTENSION IF NOT EXISTS pg_buffercache;"
# PGPASSWORD=${DB_PASS} psql --host=localhost --dbname=${DB_NAME} --username=${DB_USER} --file=./inspect_bufferpool.sql --output=./psql_forecast_log.csv --csv

# # Dump the database.
# PGPASSWORD=${DB_PASS} pg_dump --host=localhost --username=${DB_USER} --format=directory --file="./artifacts/forecast_dump" ${DB_NAME}

# # ---
# # Future log.
# # ---

# # Restore from dump.
# PGPASSWORD=${DB_PASS} pg_restore --host=localhost --username=${DB_USER} --clean --if-exists --dbname=${DB_NAME} ${DIR_DUMP}
# sudo systemctl restart postgresql
# # Replay the forecast log.
# PGUSER=${DB_USER} pgreplay -r -h localhost -p 5432 -W ${DB_PASS} "${DIR_EVAL}/pgreplay_train.out"
# PGUSER=${DB_USER} pgreplay -r -h localhost -p 5432 -W ${DB_PASS} "${DIR_EVAL}/pgreplay_future.out"
# # Capture metrics.
# PGPASSWORD=${DB_PASS} psql --host=localhost --dbname=${DB_NAME} --username=${DB_USER} --command="CREATE EXTENSION IF NOT EXISTS pg_buffercache;"
# PGPASSWORD=${DB_PASS} psql --host=localhost --dbname=${DB_NAME} --username=${DB_USER} --file=./inspect_bufferpool.sql --output=./psql_future_log.csv --csv

# # Dump the database.
# PGPASSWORD=${DB_PASS} pg_dump --host=localhost --username=${DB_USER} --format=directory --file="./artifacts/future_dump" ${DB_NAME}
