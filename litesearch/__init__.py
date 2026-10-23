__version__ = "0.1.55"
from .postfix import *
from .core import *
from .data import *
from .utils import *
from .topics import *
from .tree import *
from .sanskrit import *
from .api import *
from .quality import *
from .store import *
from pdf_oxide import *

def __getattr__(nm):
    "`SafeFastChunker` is built on first use so `import litesearch` does not load chonkie."
    if nm == 'SafeFastChunker': from .data import SafeFastChunker; return SafeFastChunker
    raise AttributeError(f"module 'litesearch' has no attribute {nm!r}")