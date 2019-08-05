def color(f):
    formatters = {
    'RED': '\u001b[31;1m',
    'GREEN': '\u001b[32;1m',
    'YELLOW': '\u001b[33;1m',
    'WHITE': '\u001b[37;1m',
    'END': '\u001b[0m',
    }
    if f <= 0.1:
        return '{RED}{}{END}'.format(f, **formatters)
    if f <= 0.5:
        return '{YELLOW}{}{END}'.format(f, **formatters)
    if f <= 0.9:
        return '{GREEN}{}{END}'.format(f, **formatters)
    return '{WHITE}{}{END}'.format(f, **formatters)
