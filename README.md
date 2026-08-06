```
       .__         .__        
  ____ |  | _____  |__| ____  
_/ __ \|  | \__  \ |  |/  _ \ 
\  ___/|  |__/ __ \|  (  <_> )
 \___  >____(____  /__|\____/ 
     \/          \/           
``` 

Status
======
work in progress\
net can learn to calculate basic equations

Requires
========
scala

Introduction
============
elaio is a learning machine based on a neural network

Quick Run
=========
install sbt (Scala Build Tool) and execute `sbt run <task>` where `<task>` is one of
- Addition
- Subtraction
- Multiplication
- Division
- Potential (wip)
- Calculator (can learn all 4 basic operations at once)

Persistence
===========
for saving/loading network data (weights and biases) execute\
`sbt run <task> --save-file <path> | --load-file <path>`

Reference Implementations
=========================
...to be done (wip)...
