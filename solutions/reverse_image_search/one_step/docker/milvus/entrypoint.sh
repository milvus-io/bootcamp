#!/bin/bash
export LD_PRELOAD=/usr/lib/embd-milvus.so
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib:/usr/local/lib

python -c "import ctypes;library = ctypes.cdll.LoadLibrary('/usr/lib/embd-milvus.so');library.startEmbedded();"