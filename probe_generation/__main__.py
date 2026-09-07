"""Entry point for `python -m probe_generation`.

All argument parsing lives in cli.py; this module just delegates so the
package can be invoked uniformly.
"""

from .cli import main

if __name__ == "__main__":
    main()
