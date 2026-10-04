"""One sqlite3 connection for the dispatcher process."""

from local_ai.db.database import SCHEMA_SQL, Database

__all__ = ["Database", "SCHEMA_SQL"]
