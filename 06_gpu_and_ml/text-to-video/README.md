Current state:
- currently stuck on @enter, issue below:
```
(T2VSynthMochiModel pid=198) Exception raised in creation task: The actor died because of an error raised in its creation task, ray::T2VSynthMochiModel.__init__() (pid=198, ip=172.20.0.86, actor_id=777ebb1a529d2a51042d5ad301000000, repr=<mochi_preview.t2v_synth_mochi.T2VSynthMochiModel object at 0x7f88d1b98650>)
(T2VSynthMochiModel pid=198)            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(T2VSynthMochiModel pid=198)            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(T2VSynthMochiModel pid=198)   File "/usr/local/lib/python3.11/site-packages/mochi_preview/t2v_synth_mochi.py", line 257, in __init__
(T2VSynthMochiModel pid=198)     config_resolved.pop("_target_")
(T2VSynthMochiModel pid=198) KeyError: '_target_'
```
- build seems to work, after much fiddling
- model incredibly slow to iterate on. 15 min download, @enter time unknown due to crash
- wrote but haven't tested Gradio app
- referenced genmoai code in comments at bottom of mochi.py for reference