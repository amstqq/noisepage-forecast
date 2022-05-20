import pickle
import psycopg

from constants import DB_CONN_STRING, SCHEMA_INT, SCHEMA_NUMERIC, SCHEMA_STRING, SCHEMA_TIMESTAMP


class ColumnSchema:
    def __init__(self, column_name, table_name, column_type, nullable) -> None:
        self._column_name = column_name
        self._table_name = table_name
        self._type = column_type
        self._nullable = nullable
        self._unique = False

    def get_type(self):
        return self._type

    def set_unique(self, unique: bool):
        self._unique = unique

    def __repr__(self):
        return str(self)

    def __str__(self):
        return f"ColumnSchema[{self._table_name}.{self._column_name}, {self._type}, {self._unique}]"


class TableSchema:
    def __init__(self, name) -> None:
        self._name = name
        self._column_schemas = {}  # Map column name to ColumnSchema object
        self._columns = []

    def add_column(self, column_name, column_type, nullable):
        assert column_name not in self._columns, "Column already seen before?"
        self._columns.append(column_name)

        column_type = column_type.lower()
        if any(pattern in column_type for pattern in ["smallint", "integer", "bigint"]):
            column_type = SCHEMA_INT
        elif any(pattern in column_type for pattern in ["numeric", "double precision", "decimal", "real"]):
            column_type = SCHEMA_NUMERIC
        elif any(pattern in column_type for pattern in ["character", "character varying", "text"]):
            column_type = SCHEMA_STRING
        elif any(pattern in column_type for pattern in ["timestamp", "date", "time"]):
            column_type = SCHEMA_TIMESTAMP
        else:
            print(f"No handler exists for type: {column_type}")
            raise RuntimeError

        self._column_schemas[column_name] = ColumnSchema(column_name, column_type, nullable)

    def set_column_unique(self, column_name, unique):
        assert column_name in self._columns, "Column does not exist"
        self._column_schemas[column_name].set_unique(unique)

    def __repr__(self):
        return str(self)

    def __str__(self):
        return_str = f"TableSchema[{self._name}]\n=======================\n"
        for v in self._column_schemas.values():
            return_str += f"{v}\n"
        return return_str


def add_column(db_schema, column_constraints, column_name, table_name, column_type, nullable):
    assert column_name not in db_schema, "Column already seen before?"

    column_type = column_type.lower()
    if any(pattern in column_type for pattern in ["smallint", "integer", "bigint"]):
        column_type = SCHEMA_INT
    elif any(pattern in column_type for pattern in ["numeric", "double precision", "decimal", "real"]):
        column_type = SCHEMA_NUMERIC
    elif any(pattern in column_type for pattern in ["character", "character varying", "text"]):
        column_type = SCHEMA_STRING
    elif any(pattern in column_type for pattern in ["timestamp", "date", "time"]):
        column_type = SCHEMA_TIMESTAMP
    else:
        print(f"No handler exists for type: {column_type}")
        raise RuntimeError

    column_schema = ColumnSchema(column_name, table_name, column_type, nullable)

    if column_name in column_constraints:
        column_schema.set_unique(True)

    db_schema[column_name] = column_schema


def get_database_schema():
    table_names_query = """
    SELECT table_name FROM information_schema.tables WHERE table_schema='public' AND table_type='BASE TABLE';
    """

    table_schema_query = """                              
    SELECT column_name, data_type, is_nullable
    FROM information_schema.columns
    WHERE table_name = %s;
    """

    # column_constraint_query = """
    # SELECT con.contype, con.conkey, con.confkey FROM pg_catalog.pg_constraint con
    # INNER JOIN pg_catalog.pg_class rel ON rel.oid = con.conrelid
    # INNER JOIN pg_catalog.pg_namespace nsp ON nsp.oid = connamespace
    # CROSS JOIN LATERAL unnest(con.conkey) ak(k)
    # WHERE rel.relname = %s;
    # """

    # Return columns with 'primary' and 'unique' constraint
    column_constraint_query = """
    SELECT a.attname, con.contype FROM pg_catalog.pg_constraint con 
    INNER JOIN pg_catalog.pg_class rel ON rel.oid = con.conrelid 
    INNER JOIN pg_catalog.pg_namespace nsp ON nsp.oid = connamespace 
    CROSS JOIN LATERAL unnest(con.conkey) ak(k)
    INNER JOIN pg_attribute a
                       ON a.attrelid = con.conrelid
                          AND a.attnum = ak.k
    WHERE rel.relname = %s AND (con.contype = 'p' or con.contype = 'u');
    """

    # Currently assume column names are unique. Map column name to ColumnSchema object.
    db_schema = {}

    with psycopg.connect(DB_CONN_STRING, autocommit=True) as conn:
        cur = conn.cursor()
        cur.execute(table_names_query)
        table_names = cur.fetchall()
        table_names = [x[0] for x in table_names]

        for table_name in table_names:
            print(f"Processing {table_name}...")
            ts = TableSchema(table_name)

            # Get all columns in this table
            cur.execute(table_schema_query, (table_name,))  # (table_name,) passed as tuple
            column_info = cur.fetchall()

            # Get all columns with unique/pkey constraint
            cur.execute(column_constraint_query, (table_name,))
            column_constraints = cur.fetchall()
            column_constraints = set(x[0] for x in column_constraints)
            # print(column_constraints)

            for column_name, column_type, nullable in column_info:
                # print(column_name, column_type, nullable)
                add_column(db_schema, column_constraints, column_name, table_name, column_type, nullable)

    return db_schema


if __name__ == "__main__":
    get_database_schema()
