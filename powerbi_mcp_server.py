"""Power BI Desktop MCP server.

Lets Claude Desktop read a locally open Power BI Desktop model (tables, columns,
measures, relationships) and run DAX queries against it. Connects to the
Analysis Services instance that Power BI Desktop hosts on localhost, via
pythonnet + System.Data.OleDb + the MSOLAP provider, using Windows integrated
security. Read-only: only DMV and DAX EVALUATE queries are issued.

stdout is reserved for the MCP stdio protocol, so all logging goes to stderr.
"""

import glob
import logging
import os
import re
import subprocess
import sys
from typing import Any

from mcp.server.mcpserver import MCPServer

logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("powerbi-mcp")

try:
    import clr  # pythonnet

    clr.AddReference("System.Data")
    from System import DBNull  # noqa: E402
    from System.Data.OleDb import OleDbConnection, OleDbCommand  # noqa: E402
except Exception as exc:  # pragma: no cover - environment dependent
    log.error("pythonnet / System.Data unavailable: %s", exc)
    OleDbConnection = None


def _load_adomd():
    """Load ADOMD.NET from the Power BI Desktop install.

    The Microsoft Store build of Power BI Desktop ships msolap.dll without
    registering it, so OLE DB fails there; its bundled ADOMD.NET client needs
    no registration. Returns (AdomdConnection, AdomdCommand) or (None, None).
    """
    if OleDbConnection is None:
        return None, None
    pf = os.environ.get("ProgramFiles", r"C:\Program Files")
    patterns = [
        os.path.join(pf, "WindowsApps", "Microsoft.MicrosoftPowerBIDesktop_*_x64__8wekyb3d8bbwe", "bin",
                     "Microsoft.PowerBI.AdomdClient.dll"),
        os.path.join(pf, "Microsoft Power BI Desktop", "bin", "Microsoft.PowerBI.AdomdClient.dll"),
        os.path.join(pf, "Microsoft Power BI Desktop", "bin", "Microsoft.AnalysisServices.AdomdClient.dll"),
    ]
    candidates = []
    for pattern in patterns:
        candidates += sorted(glob.glob(pattern), reverse=True)  # newest Store version first
    for dll in candidates:
        try:
            clr.AddReference(dll)
            from Microsoft.AnalysisServices.AdomdClient import AdomdConnection, AdomdCommand
            log.info("Using ADOMD.NET from %s", dll)
            return AdomdConnection, AdomdCommand
        except Exception as exc:
            log.warning("Could not load %s: %s", dll, exc)
    log.info("ADOMD.NET not found; falling back to OLE DB (MSOLAP)")
    return None, None


AdomdConnection, AdomdCommand = _load_adomd()

mcp = MCPServer("powerbi")

# TMSCHEMA enums
_DATA_TYPES = {
    1: "Automatic", 2: "String", 6: "Int64", 8: "Double", 9: "DateTime",
    10: "Decimal", 11: "Boolean", 17: "Binary", 19: "Unknown", 20: "Variant",
}
_COLUMN_TYPES = {1: "Data", 2: "Calculated", 3: "RowNumber", 4: "CalculatedTableColumn"}
_CARDINALITY = {1: "One", 2: "Many"}
_CROSS_FILTER = {1: "OneDirection", 2: "BothDirections", 3: "Automatic"}


# ---------------------------------------------------------------- connection

def _workspace_roots() -> list[str]:
    local = os.environ.get("LOCALAPPDATA", "")
    home = os.environ.get("USERPROFILE", "")
    return [
        os.path.join(local, "Microsoft", "Power BI Desktop", "AnalysisServicesWorkspaces"),
        os.path.join(home, "Microsoft", "Power BI Desktop Store App", "AnalysisServicesWorkspaces"),
        os.path.join(local, "Packages", "Microsoft.MicrosoftPowerBIDesktop_8wekyb3d8bbwe",
                     "LocalCache", "Local", "Microsoft", "Power BI Desktop", "AnalysisServicesWorkspaces"),
    ]


def _read_port_file(path: str) -> int | None:
    raw = open(path, "rb").read()
    for enc in ("utf-16", "utf-8"):
        try:
            text = raw.decode(enc).strip("﻿\x00 \r\n")
            if text.isdigit():
                return int(text)
        except UnicodeDecodeError:
            continue
    return None


def _ports_from_netstat() -> list[int]:
    """Fallback: listening ports owned by msmdsrv.exe processes."""
    try:
        tasks = subprocess.run(["tasklist", "/FI", "IMAGENAME eq msmdsrv.exe", "/FO", "CSV", "/NH"],
                               capture_output=True, text=True, timeout=10).stdout
        pids = set(re.findall(r'"msmdsrv\.exe","(\d+)"', tasks, re.I))
        if not pids:
            return []
        netstat = subprocess.run(["netstat", "-ano", "-p", "TCP"],
                                 capture_output=True, text=True, timeout=10).stdout
        ports = []
        for line in netstat.splitlines():
            parts = line.split()
            if len(parts) >= 5 and parts[3].upper() == "LISTENING" and parts[4] in pids:
                ports.append(int(parts[1].rsplit(":", 1)[1]))
        return sorted(set(ports))
    except Exception as exc:
        log.warning("netstat port detection failed: %s", exc)
        return []


def find_ports() -> list[int]:
    """All candidate ports of running Power BI Desktop models, newest first."""
    files = []
    for root in _workspace_roots():
        files += glob.glob(os.path.join(root, "*", "Data", "msmdsrv.port.txt"))
    files.sort(key=os.path.getmtime, reverse=True)
    ports = [p for p in (_read_port_file(f) for f in files) if p]
    live = set(_ports_from_netstat())
    if live:
        # Stale workspace folders can linger after Power BI closes; keep only live ports.
        ports = [p for p in ports if p in live] or sorted(live)
    return list(dict.fromkeys(ports))


def _connect(port: int | None = None, catalog: str | None = None):
    if OleDbConnection is None:
        raise RuntimeError("pythonnet is not available. Install it with: pip install pythonnet")
    if port is None:
        ports = find_ports()
        if not ports:
            raise RuntimeError("Could not find Power BI Desktop Analysis Services port. "
                               "Make sure Power BI Desktop is running with a report open.")
        port = ports[0]
    cs = f"Data Source=localhost:{port};Integrated Security=SSPI;"
    if catalog:
        cs += f"Initial Catalog={catalog};"
    if AdomdConnection is not None:
        conn = AdomdConnection(cs)
    else:
        conn = OleDbConnection("Provider=MSOLAP;" + cs)
    conn.Open()
    return conn


def _convert(value: Any) -> Any:
    if value is None or isinstance(value, DBNull):
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    type_name = value.GetType().FullName
    if type_name == "System.Decimal":
        return float(str(value))
    if type_name == "System.DateTime":
        return value.ToString("o")
    return str(value)


def _query(sql: str, port: int | None = None, max_rows: int | None = None) -> list[dict]:
    conn = _connect(port)
    try:
        cmd = AdomdCommand(sql, conn) if AdomdConnection is not None else OleDbCommand(sql, conn)
        reader = cmd.ExecuteReader()
        try:
            cols = [reader.GetName(i) for i in range(reader.FieldCount)]
            rows = []
            while reader.Read():
                if max_rows is not None and len(rows) >= max_rows:
                    break
                rows.append({c: _convert(reader.GetValue(i)) for i, c in enumerate(cols)})
            return rows
        finally:
            reader.Close()
    finally:
        conn.Close()


def _tables_by_id() -> dict[int, str]:
    return {r["ID"]: r["Name"] for r in _query("SELECT [ID], [Name] FROM $SYSTEM.TMSCHEMA_TABLES")}


def _columns_by_id() -> dict[int, dict]:
    return {r["ID"]: r for r in _query(
        "SELECT [ID], [TableID], [ExplicitName], [InferredName] FROM $SYSTEM.TMSCHEMA_COLUMNS")}


def _col_name(col: dict) -> str:
    return col.get("ExplicitName") or col.get("InferredName") or ""


def _error(exc: Exception) -> dict:
    log.exception("tool failed")
    return {"error": str(exc)}


# --------------------------------------------------------------------- tools

@mcp.tool()
def get_powerbi_models() -> dict:
    """List the Power BI Desktop models currently open on this machine (port and database name)."""
    try:
        models = []
        for port in find_ports():
            try:
                dbs = _query("SELECT [CATALOG_NAME], [DATE_MODIFIED] FROM $SYSTEM.DBSCHEMA_CATALOGS", port=port)
                models += [{"port": port, "database": d["CATALOG_NAME"], "modified": d["DATE_MODIFIED"]} for d in dbs]
            except Exception as exc:
                models.append({"port": port, "error": str(exc)})
        if not models:
            return {"models": [], "message": "No running Power BI Desktop model found. Open a report in Power BI Desktop."}
        return {"models": models, "active_port": models[0]["port"]}
    except Exception as exc:
        return _error(exc)


@mcp.tool()
def get_model_tables() -> dict:
    """List all tables in the active Power BI model, including hidden flag and description."""
    try:
        rows = _query("SELECT [ID], [Name], [Description], [IsHidden] FROM $SYSTEM.TMSCHEMA_TABLES")
        tables = [{"name": r["Name"], "description": r["Description"], "is_hidden": r["IsHidden"]}
                  for r in rows]
        tables.sort(key=lambda t: t["name"].lower())
        return {"count": len(tables), "tables": tables}
    except Exception as exc:
        return _error(exc)


@mcp.tool()
def get_table_columns(table_name: str) -> dict:
    """Get the columns of one table: name, data type, column type, hidden flag and DAX expression (for calculated columns)."""
    try:
        table_ids = [tid for tid, name in _tables_by_id().items() if name.lower() == table_name.lower()]
        if not table_ids:
            return {"error": f"Table '{table_name}' not found. Use get_model_tables to list tables."}
        rows = _query("SELECT [TableID], [ExplicitName], [InferredName], [ExplicitDataType], [Type], "
                      "[IsHidden], [Expression], [Description], [FormatString] FROM $SYSTEM.TMSCHEMA_COLUMNS")
        cols = [{
            "name": _col_name(r),
            "data_type": _DATA_TYPES.get(r["ExplicitDataType"], r["ExplicitDataType"]),
            "column_type": _COLUMN_TYPES.get(r["Type"], r["Type"]),
            "is_hidden": r["IsHidden"],
            "format_string": r["FormatString"],
            "description": r["Description"],
            "expression": r["Expression"],
        } for r in rows if r["TableID"] == table_ids[0] and r["Type"] != 3]  # skip internal RowNumber
        return {"table": table_name, "count": len(cols), "columns": cols}
    except Exception as exc:
        return _error(exc)


@mcp.tool()
def execute_dax_query(dax_query: str, max_rows: int = 1000) -> dict:
    """Run a DAX query (must start with EVALUATE or DEFINE) against the active model and return up to max_rows rows."""
    try:
        stripped = re.sub(r"^\s*(//[^\n]*\n|--[^\n]*\n|/\*.*?\*/)*\s*", "", dax_query, flags=re.S)
        if not re.match(r"(EVALUATE|DEFINE)\b", stripped, re.I):
            return {"error": "Only DAX queries starting with EVALUATE or DEFINE are allowed."}
        rows = _query(dax_query, max_rows=max_rows + 1)
        truncated = len(rows) > max_rows
        rows = rows[:max_rows]
        return {"row_count": len(rows), "truncated": truncated, "rows": rows}
    except Exception as exc:
        return _error(exc)


@mcp.tool()
def get_measures(table_name: str | None = None) -> dict:
    """List all measures with their DAX expressions, optionally filtered to one home table."""
    try:
        tables = _tables_by_id()
        rows = _query("SELECT [TableID], [Name], [Expression], [FormatString], [Description], "
                      "[IsHidden], [DisplayFolder] FROM $SYSTEM.TMSCHEMA_MEASURES")
        measures = [{
            "table": tables.get(r["TableID"]),
            "name": r["Name"],
            "expression": r["Expression"],
            "format_string": r["FormatString"],
            "display_folder": r["DisplayFolder"],
            "description": r["Description"],
            "is_hidden": r["IsHidden"],
        } for r in rows]
        if table_name:
            measures = [m for m in measures if (m["table"] or "").lower() == table_name.lower()]
        measures.sort(key=lambda m: ((m["table"] or "").lower(), m["name"].lower()))
        return {"count": len(measures), "measures": measures}
    except Exception as exc:
        return _error(exc)


@mcp.tool()
def get_relationships() -> dict:
    """List all relationships: from/to table and column, cardinality, cross-filter direction and active flag."""
    try:
        tables = _tables_by_id()
        cols = _columns_by_id()
        rows = _query("SELECT [FromTableID], [FromColumnID], [FromCardinality], [ToTableID], [ToColumnID], "
                      "[ToCardinality], [CrossFilteringBehavior], [IsActive] FROM $SYSTEM.TMSCHEMA_RELATIONSHIPS")
        rels = [{
            "from_table": tables.get(r["FromTableID"]),
            "from_column": _col_name(cols.get(r["FromColumnID"], {})),
            "to_table": tables.get(r["ToTableID"]),
            "to_column": _col_name(cols.get(r["ToColumnID"], {})),
            "cardinality": f'{_CARDINALITY.get(r["FromCardinality"], "?")}-to-{_CARDINALITY.get(r["ToCardinality"], "?")}',
            "cross_filter": _CROSS_FILTER.get(r["CrossFilteringBehavior"], r["CrossFilteringBehavior"]),
            "is_active": r["IsActive"],
        } for r in rows]
        return {"count": len(rels), "relationships": rels}
    except Exception as exc:
        return _error(exc)


if __name__ == "__main__":
    log.info("Starting Power BI MCP server (detected ports: %s)", find_ports() or "none")
    mcp.run()
