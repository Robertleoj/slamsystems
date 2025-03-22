def in_jupyter() -> bool:
    try:
        get_ipython = __import__("IPython").get_ipython
        return "zmqshell" in str(get_ipython())
    except Exception:
        return False
