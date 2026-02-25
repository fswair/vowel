### DB Fixture Binding ###

## On setup, it will initialize a Connection instance.
## On teardown, it calls instance.close, takes no argument.
## On destruction, instance will be removed from fixture manager.
## NOTE: It depends on scope, will be removed if scope out of lifecycle.

## If you use function fixtures instead class fixtures
## Approach will be:

## On setup, create_db(**kwargs) -> Conn
## On teardown, close_db(*, db: Conn) -> None
## On destruction, instance will be removed from fixture manager.
## NOTE: It depends on scope, will be removed if scope out of lifecycle.

import logfire

# enable observability (optional)
# logfire.configure(service_name="db-fixture")


class NoTableError(Exception):
    """Exception raised when a specified table is not found in the database."""

    pass


class Connection:
    """
    Represents a database connection for fixture management.

    Args:
        db_path (str): Path to the database file.
    """

    def __init__(self, *, db_path: str):
        with logfire.span("fixture setup", id=id(self)):
            logfire.info("connection::init(db_path='{db_path}')", db_path=db_path)

    def execute(self, query: str, table: str = "users"):
        """
        Executes a SQL query on the specified table.

        Args:
            query (str): The SQL query to execute.
            table (str, optional): The table to query. Defaults to 'users'.

        Returns:
            list: Query results as a list of dictionaries.

        Raises:
            NoTableError: If the specified table is not found in the query.
        """
        with logfire.span("fixture interaction"):
            logfire.info("connection::exec(query='{query}')", query=query)
        if f"from {table}" not in query.lower():
            raise NoTableError("No such table.")
        if "id" in query:
            return [{"id": 1, "name": "Mert"}]
        return []

    def close(self, *args):
        """
        Closes the database connection.

        Args:
            *args: Optional arguments for closing the connection.
        """
        repr_args = str(args or "")
        with logfire.span("fixture teardown", id=id(self)):
            logfire.info(f"connection::close({repr_args})", repr_args=repr_args)
