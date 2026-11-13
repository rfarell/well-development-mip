"""Currency presentation for generated TeX tables; never changes stored numbers.

Apply only to table output, after scientific summaries and numeric macros have
been formed. Currency-bearing table columns use caption-level units so narrow
headings retain their original width; prose amounts spell out USD and million.
"""
import re


_NUMBER = r"([0-9][0-9,.]*)"
_TABLE = re.compile(r"\\begin\{table\*?\}.*?\\end\{table\*?\}", re.DOTALL)
_UNIT_NOTE = " Monetary amounts, including risk tolerances, are in USD million."


def _prose_currency(text):
    # Legacy captions put the currency symbol inside the rho math span.
    text = re.sub(r"\$\\rho\s*=\s*\\\$" + _NUMBER + r"\$M",
                  lambda m: r"$\rho=$ USD~" + m[1] + " million", text)
    text = re.sub(r"\$\\rho\s*=\s*\$\s*\\\$" + _NUMBER + r"\$M",
                  lambda m: r"$\rho=$ USD~" + m[1] + " million", text)
    text = re.sub(r"\\\$" + _NUMBER + "M",
                  lambda m: "USD~" + m[1] + " million", text)
    return text.replace(r"\$M", "USD million")


def _table_currency(match):
    table = match[0]
    start = table.find(r"\caption{")
    if start < 0:
        return _prose_currency(table)
    start += len(r"\caption{")
    depth, end = 1, start
    while depth:
        if end >= len(table):
            raise ValueError("Unclosed generated-table caption")
        if table[end] in "{}" and table[end - 1] != "\\":
            depth += 1 if table[end] == "{" else -1
        end += 1
    caption = _prose_currency(table[start:end - 1])
    body = table[end:]
    if r"\$" in body:
        # Keep each numeric cell and rho heading unchanged in scale and position.
        body = re.sub(r"\\\$" + _NUMBER + r"\$M", lambda m: m[1] + "$", body)
        body = re.sub(r"\\\$" + _NUMBER + "M", lambda m: m[1], body)
        body = re.sub(r",\s*\\\$M", "", body)
        body = re.sub(r"\s+\(\\\$M\)", "", body)
        body = body.replace(r"\$M", "USD million")
        caption += _UNIT_NOTE
    return table[:start] + caption + "}" + body


def format_table_currency(text):
    """Return a presentation-only, idempotent USD rendering of table TeX."""
    return _prose_currency(_TABLE.sub(_table_currency, text))
