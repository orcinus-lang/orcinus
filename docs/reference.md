[TOC]

Orcinus reference
=================

This document is a reference manual for the Orcinus programming language.

Orcinus is high-level programming language with static typing. Like Python it
uses indentation instead of braces to structure program's blocks.


```python
from system.io import print


def identity[T](value: T) -> T:
    return value


def main() -> int:
    print(identitity("Hello world!"))
    return 0

```


Module
------

Each source file is defined module. Module contains types, function and other declarations.

```abnf
module = imports members
```


Imports
-------

In beginning of each module you can import symbols/members from another module. 

```abnf
imports         = { import } 

import          = "import" alias
                = "from" qualified-name "import" alias

qualified-name  = Name [ '.' qualified-name ]
```


Module members
--------------

```abnf
members = { member }
member  = function | class | struct | interface | enum 
```

Function definition
-------------------

```abnf
function        = attributes 'def' ID [ "[" generic-parameters "]" ] 
                  "(" parameters-list ")" [ "->" type ] ":" function-scope

parameters-list = [ [ parameters ] [ "*" "," parameters ] ]
parameters      = parameter [ "," parameter ]+
parameter       = ID [ ":" type ]

function-scope  = scope-statement
                = "..." "\n"
```

Function is defined with name, optional parameter list and optional result type.

```python
def fibonacci(n: int) -> int:
    if n > 2:
        return fibonacci(n - 2) + fibonacci(n - 1)
    return 1
```

You can define function without implementation using ellipsis technique:

```python
def func(): ...
```

Type definition 
---------------

```abnf
class     = attributes "class" ID [ "[" generic-parameters "]" ] ":" 
            type-members
              
struct    = attributes "struct" ID [ "[" generic-parameters "]" ] ":" 
            type-members
              
interface = attributes "interface" ID [ "[" generic-parameters "]" ] ":" 
            type-members
            
enum      = attributes "class" ID [ "[" generic-parameters "]" ] ":" 
            type-members                                
```

```abnf
type-members = "..." "\n"
type-members = "\n" Indent { type-member } Undent

type-member  = function | class | struct | interface | enum | "pass"
```

Generic functions and types
---------------------------

You can define generic functions and types

```abnf
generic-parameters = generic-parameter { ',' generic-parameter } 
generic-parameter  = ID [ ":" generic-concepts ]
generic-concepts   = type { ',' type } 
```
