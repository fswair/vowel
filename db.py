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

# (optional)

logfire.configure(service_name='db-fixture') 

class Connection:
    def __init__(self, *, db_path: str):
        with logfire.span("fixture setup", id=id(self)):
            logfire.info("connection::init(db_path='{db_path}')", db_path=db_path)

    def execute(self, query: str):
        with logfire.span("fixture interaction"):
            logfire.info("connection::exec(query='{query}')", query=query)
        if "id" in query:
            return [{"id": 1, "name": "Mert"}]
        return []

    def close(self, *args):
        repr_args = str(args or "")
        with logfire.span("fixture teardown", id=id(self)):
            logfire.info(f"connection::close({repr_args})", repr_args=repr_args)
