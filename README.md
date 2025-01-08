# Bangla Toy Programming Language

This repository contains the **Bangla Toy Programming Language**, which is designed to help teach basic programming concepts using the Bengali language.

## Grammar

<program> ::= "চল" <statement_list> "থাম"

<statement_list> ::= <statement> | <statement> <statement_list>

<statement> ::= <assignment> | <array_declaration> | <array_assignment> | <if_statement> | <while_statement> | <print_statement> | <input_statement>

<assignment> ::= <identifier> "=" <expression> ";"

<array_declaration> ::= "তালিকা" <identifier> "[" <expression>"]" ";"

<array_assignment> ::= <identifier> "[" <expression> "]" "=" <expression> ";"

<if_statement> ::= "হয়" "(" <condition> ")" "{" <statement_list> "}" "নয়" "{" <statement_list> "}"

<while_statement> ::= "যখন" "(" <condition> ")" "{" <statement_list> "}"

<print_statement> ::= "প্রিন্ট" "(" <expression> ")" ";"

<input_statement> ::= "ইনপুট" "(" <expression> ")" ";"

<expression> ::= <term> | <term> "+" <expression> | <term> "-" <expression>

<term> ::= <factor> | <factor> "*" <term> | <factor> "/" <term>

<factor> ::= <number> | <identifier> | <identifier> "[" <expression> "]" | "(" <expression> ")"

<condition> ::= <expression> <comparison_op> <expression>

<comparison_op> ::= "==" | "!=" | "<" | ">" | "<=" | ">="

<identifier> ::= letter (letter | digit)*

<number> ::= digit+

# Example Program

চল
ত=5;
তালিকা তাল[ত];
ল=0;
যখন(ল<ত){
    কা=0;
    ইনপুট(কা);
    তাল[ল]=কা;
    ল=ল+1;    
}
ল=0;
যখন(ল<ত){
    প্রিন্ট(তাল[ল]);
    ল=ল+1;    
}
থাম

