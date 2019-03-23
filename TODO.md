Work in progress
================

Minimum viable product (MVP)
----------------------------

- [ ] Language reference:
    - [x] Condition statement (`if`):
        - [x] `if` ...
        - [x] `if` ... `else` ...
        - [x] `if` ... `elif` ... `else` ...
        - [x] Unreachable code         
        - [ ] Implicit boolean check (`bool`)
    - [ ] Condition loop statement (`while`):
        - [x] `while` ...        
        - [x] `while` ... `else` ...
        - [x] Unreachable code
        - [ ] Implicit boolean check (`bool`)
        - [ ] `break`
        - [ ] `continue`
    - [ ] Function call
        - [x] Return value from function
        - [x] Return void value from function, e.g. function without return        
        - [ ] Type checking
        - [ ] Function overload (required type checking)
    - [ ] Type checks:
        - [ ] Check function parameters and arguments 
        - [x] Check function return type and type of `return` statement        
        - [ ] Generics:
            - [ ] Separate type checking
            - [ ] Separate compilation
            - [ ] Type inference
    - [ ] Function statements:
        - [x] Uniform function call syntax
        - [ ] Uniform property and field access 
        - [x] Uniform operators calls, e.g. `a + b` is syntax sugar for `__add__(a, b)`
        - [x] Function call 
        - [ ] Variable declaration and initialization
    - [ ] Constants:
        - [x] Literal constants for integer type: e.g. `1`, `2`, `-3`
        - [ ] Literal constants for float type: e.g. `1`, `2`, `-3`
        - [x] Literal constants for boolean type: `True` and `False`
        - [x] Literal constant for void type: `None`
        - [ ] Constants folding
    - [ ] String literals
    - [ ] Types:
        - [ ] Function type (`List` generic!)
        - [ ] Array type (`Array`, generic!)
        - [ ] Boolean type (`bool`)
- [ ] Standard library:        
    - [ ] Integer type:
        - [ ] Unary operators
        - [ ] Binary operators
        - [ ] Compare operators
    - [ ] Boolean type:
        - [ ] Unary operators
        - [ ] Binary operators
        - [ ] Compare operators
    - [ ] String type:
        - [ ] Length (`__len__`)
        - [ ] Binary operators
        - [ ] Compare operators
    - [ ] Function for retrieve length: `len`
    - [ ] Function for boolean tests: `bool`

TODO
----

- [ ] Scanner:
    - [ ] Store trivia and errors in tokens
- [ ] Parser:
    - [ ] Speed up expression parsing, because is recursive-descent parser is not good choice fot this work
- [ ] Language reference:    
    - [ ] Concepts for generics

