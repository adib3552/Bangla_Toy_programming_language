#include <iostream>
#include <string>
#include <vector>
#include <cctype>
#include<map>
#pragma warning(push)
#pragma warning(disable: 4996) // Suppresses deprecation warnings in MSVC
#include <locale>
#include <codecvt> 
#include <cstdint>
#include <stdexcept>
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/raw_os_ostream.h"

#define _CRT_SECURE_NO_WARNINGS
#pragma warning(push)
#pragma warning(disable : 4146) // Disable unary minus warning for unsigned types
#pragma warning(disable : 4996) // Disable deprecation warnings for unsafe functions

#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/ExecutionEngine/Orc/ThreadSafeModule.h"
#include "llvm/Support/TargetSelect.h"

#pragma warning(pop)

#include <fstream>
enum TokenType {
    BEGIN, END, IF, WHILE, PRINT,
    IDENTIFIER, NUMBER, ASSIGN, PLUS, MINUS, MULTIPLY, DIVIDE,
    EQUAL, NOT_EQUAL, LESS, GREATER, LESS_EQUAL, GREATER_EQUAL,
    OPEN_PAREN, CLOSE_PAREN, OPEN_BRACE, CLOSE_BRACE, OPEN_BRACKET, CLOSE_BRACKET,
    SEMICOLON, ARRAY, END_OF_FILE, INPUT, ELSE
};

struct Token {
    TokenType type;
    std::wstring value;
};

std::string convertWStringToString(const std::wstring wstr) {
    std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;
    return converter.to_bytes(wstr);
}

bool isBanglaCharacter(wchar_t ch) {
    // Check if the character is within the Bengali Unicode range
    return (ch >= 0x0980 && ch <= 0x09FF);
}

#pragma warning(pop)

static std::unique_ptr<llvm::LLVMContext> TheContext;
static std::unique_ptr<llvm::IRBuilder<>>Builder;
static std::unique_ptr<llvm::Module> TheModule;
std::map<std::wstring, std::pair<llvm::AllocaInst*, llvm::Type*>> NamedValues;



struct ASTNode {
    virtual ~ASTNode() = default;
    virtual llvm::Value* codegen() = 0; 
};
struct ExpressionNode : ASTNode {};
struct StatementNode : ASTNode {};

struct NumberNode : ExpressionNode {
    int value;
    explicit NumberNode(int val) : value(val) {}

    llvm::Value* codegen() override {

        //std::cout << "in the number node\n";
        llvm::ConstantInt* num = llvm::ConstantInt::get(llvm::Type::getInt32Ty(*TheContext), value);


        //std::cout << "Found number" << "\n";

        return num;
    }
};

struct VariableNode : ExpressionNode {
    std::wstring name;
    explicit VariableNode(const std::wstring& var) : name(var) {}

    llvm::Value* codegen() override {

        //std::cout << "Inside the variable node\n";

        llvm::Value* value = NamedValues[name].first;
        llvm::Type* valType = NamedValues[name].second;

        if (!value) {
            throw std::runtime_error("unknown variable name -> "+convertWStringToString(name));
        }


        llvm::Value* loadedValue = Builder->CreateLoad(
            valType,       // Type of the value to load
            value,             // Pointer to load from
            convertWStringToString(name) + "_load"    // Debugging name
        );

        //std::cout << "Loaded value type: " << loadedValue->getType()->getTypeID() << std::endl;


        return loadedValue;


    }
};

struct AssignmentNode : StatementNode {
    std::wstring variableName;
    std::unique_ptr<ExpressionNode> value;

    AssignmentNode(const std::wstring& varName, std::unique_ptr<ExpressionNode> val)
        : variableName(varName), value(std::move(val)) {
    }

    llvm::Value* codegen() override {

        if (!value) {
            throw std::runtime_error("Expression in AssignmentNode is null.");
        }

        llvm::Value* val = value->codegen();

        std::cout << val->getType()->getTypeID() << "\n";

        if (!val) {
            throw std::runtime_error("Error on assigning value");
        }

        llvm::Function* CurrentFunction = Builder->GetInsertBlock()->getParent();

        if (!CurrentFunction) {
            throw std::runtime_error("AssignmentNode requires a valid function context.");
        }

        llvm::AllocaInst* var = NamedValues[variableName].first;

        if (!var) {

             // Allocate space for the variable in the entry block
            var = Builder->CreateAlloca(val->getType(), nullptr, convertWStringToString(variableName));
            NamedValues[variableName].first = var;
            NamedValues[variableName].second = val->getType();


            Builder->CreateStore(val, var);


            //std::cout << "Inside this\n";
        }
        else {
            Builder->CreateStore(val, var);
            //std::cout << "Now Inside this\n";
        }

        return val;
    }
};


struct BinaryNode : ExpressionNode {
    std::wstring op; // + - * /
    std::unique_ptr<ExpressionNode> left;
    std::unique_ptr<ExpressionNode> right;

    explicit BinaryNode(std::wstring opr, std::unique_ptr<ExpressionNode>l, std::unique_ptr<ExpressionNode>r) : op(opr), left(std::move(l)), right(std::move(r)) {}

    llvm::Value* codegen() override {
        llvm::Value* L = left->codegen();
        llvm::Value* R = right->codegen();

        if (!L || !R) throw std::runtime_error("can not complete the operation");

        switch (op[0]) {
        case '+':
            return Builder->CreateAdd(L, R, "addtmp");
        case '-':
            return Builder->CreateSub(L, R, "subtmp");
        case '*':
            return Builder->CreateMul(L, R, "multmp");
        case '/':
            return Builder->CreateSDiv(L, R, "divtmp");
        default:
            throw std::runtime_error("Unsupported Operator");
        }
    }
};

struct ConditionNode : ExpressionNode {
    std::wstring op; // Comparison operator (e.g., "==", "<", ">", ">=", "<=")
    std::unique_ptr<ExpressionNode> left;
    std::unique_ptr<ExpressionNode> right;

    ConditionNode(std::wstring opr, std::unique_ptr<ExpressionNode> l, std::unique_ptr<ExpressionNode> r)
        : op(std::move(opr)), left(std::move(l)), right(std::move(r)) {
    }

    llvm::Value* codegen() override {
        // Generate code for left and right operands
        llvm::Value* L = left->codegen();
        llvm::Value* R = right->codegen();

        if (!L || !R) {
            throw std::runtime_error("Cannot generate code for operands in ConditionNode.");
        }

        // Ensure types match
        if (L->getType() != R->getType()) {
            throw std::runtime_error("Type mismatch in ConditionNode operands.");
        }

        // Generate the comparison
        llvm::Value* result = nullptr;
        if (L->getType()->isIntegerTy()) {
            // Integer comparison
            if (op == L"==") {
                result = Builder->CreateICmpEQ(L, R, "cmp_eq");
            }
            else if (op == L"!=") {
                result = Builder->CreateICmpNE(L, R, "cmp_ne");
            }
            else if (op == L"<") {
                result = Builder->CreateICmpSLT(L, R, "cmp_lt");
            }
            else if (op == L"<=") {
                result = Builder->CreateICmpSLE(L, R, "cmp_le");
            }
            else if (op == L">") {
                result = Builder->CreateICmpSGT(L, R, "cmp_gt");
            }
            else if (op == L">=") {
                result = Builder->CreateICmpSGE(L, R, "cmp_ge");
            }
            else {
                throw std::runtime_error("Unsupported comparison operator for integers: " + convertWStringToString(op));
            }
        }
        else {
            throw std::runtime_error("Unsupported operand type for ConditionNode.");
        }

        return result;
    }

};


struct PrintNode : StatementNode {
    std::unique_ptr<ExpressionNode> expression;

    explicit PrintNode(std::unique_ptr<ExpressionNode> expr) : expression(std::move(expr)) {}

    llvm::Value* codegen() override {
        llvm::Value* val = expression->codegen();

        if (!val) {
            throw std::runtime_error("Error generating code for the expression in PrintNode");
        }

        llvm::Function* printFunc = TheModule->getFunction("printf");
        
        if (!printFunc) {
            // Define the printf function type: int printf(char*, ...)
            llvm::FunctionType* printFuncType = llvm::FunctionType::get(
                Builder->getInt32Ty(),                                         // Return type: int
                llvm::PointerType::get(llvm::Type::getInt8Ty(*TheContext), 0),  // First argument type: char*
                true                                                           // Variadic function
            );
            printFunc = llvm::Function::Create(
                printFuncType, llvm::Function::ExternalLinkage, "printf", *TheModule);
           
        }

        llvm::Value* formatStr = Builder->CreateGlobalStringPtr("%d\n");;
        std::cout << "Data type: " << val->getType()->getTypeID() << "\n";
        if (val->getType()->isIntegerTy(32)) {
            formatStr = Builder->CreateGlobalStringPtr("%d\n");
        }
        else {
            throw std::runtime_error("Unsupported type for PrintNode");
        }

        return Builder->CreateCall(printFunc, { formatStr, val });;
    }
};

struct BlockNode : StatementNode {
    std::vector<std::unique_ptr<StatementNode>> statements;

    llvm::Value* codegen() override {
        llvm::Value* lastValue = nullptr;

        for (auto& e : statements) {
            lastValue = e->codegen();
            if (!lastValue) {
                throw std::runtime_error("No block found");
            }
        }

        return lastValue;
    }
};

struct IfNode : StatementNode {
    std::unique_ptr<ExpressionNode> condition;
    std::unique_ptr<BlockNode> trueBlock;
    std::unique_ptr<BlockNode> elseBlock;

    IfNode(std::unique_ptr<ExpressionNode> cond, std::unique_ptr<BlockNode> tBlock, std::unique_ptr<BlockNode> eBlock)
        : condition(std::move(cond)), trueBlock(std::move(tBlock)), elseBlock(std::move(eBlock)) {
    }

    llvm::Value* codegen() override {
        // Generate the condition code
        llvm::Value* Cond = condition->codegen();
        if (!Cond) {
            throw std::runtime_error("Failed to generate code for condition in IfNode.");
        }

        // Ensure the condition is a boolean type
        Cond = Builder->CreateICmpNE(
            Cond,
            llvm::ConstantInt::get(Cond->getType(), 0, true),
            "ifcond"
        );

        // Get the current function
        llvm::Function* TheFunction = Builder->GetInsertBlock()->getParent();

        // Create basic blocks for "then", "else", and "merge"
        llvm::BasicBlock* ThenBB = llvm::BasicBlock::Create(*TheContext, "then", TheFunction);
        llvm::BasicBlock* ElseBB = llvm::BasicBlock::Create(*TheContext, "else", TheFunction);
        llvm::BasicBlock* MergeBB = llvm::BasicBlock::Create(*TheContext, "ifcont", TheFunction);

        // Add the conditional branch
        Builder->CreateCondBr(Cond, ThenBB, ElseBB);

        // Generate the "then" block
        Builder->SetInsertPoint(ThenBB);
        llvm::Value* ThenValue = trueBlock->codegen();
        if (!ThenValue) {
            throw std::runtime_error("Failed to generate code for true block in IfNode.");
        }
        // Add an unconditional branch to the merge block
        Builder->CreateBr(MergeBB);

        // Generate the "else" block
        Builder->SetInsertPoint(ElseBB);
        llvm::Value* ElseValue = elseBlock->codegen();
        // Add an unconditional branch to the merge block
        Builder->CreateBr(MergeBB);

        // Set the insertion point to the merge block
        Builder->SetInsertPoint(MergeBB);

        return ElseValue; // If no value is returned, just return nullptr
    }
};


struct ArrayDeclareNode : StatementNode {
    std::wstring name;
    std::unique_ptr<ExpressionNode> size;
    ArrayDeclareNode(std::wstring& arrName, std::unique_ptr<ExpressionNode>s) : name(arrName), size(std::move(s)) {}



    llvm::Value* codegen() override {


        //std::cout << "Inside Array declare\n";

        llvm::Value* arrSize = size->codegen();
        if (!arrSize) {
            throw std::runtime_error("Failed to generate array size.");
        }
        if (!arrSize->getType()->isIntegerTy()) {
            throw std::runtime_error("Array size must be an integer.");
        }

        // Ensure the size is of type i32 for alloca (LLVM expects i32 for sizes in alloca)
        if (arrSize->getType() != llvm::Type::getInt32Ty(*TheContext)) {
            arrSize = Builder->CreateIntCast(arrSize, llvm::Type::getInt32Ty(*TheContext), false, "arraysizecast");
        }
        //Allocate space for the array
        llvm::Type* elementType = llvm::Type::getInt32Ty(*TheContext); // Assuming int arrays for now

        llvm::AllocaInst* arrayAlloca = Builder->CreateAlloca(
            elementType,
            arrSize, // Size of the array
            convertWStringToString(name) + "_array"
        );

        //Store the allocation in the symbol table (NamedValues)
        NamedValues[name].first = arrayAlloca;
        //Return the allocation (useful for potential parent nodes)
        return arrayAlloca;
    }
};


struct ArrayAssignmentNode : StatementNode {
    std::wstring arrayName; // Name of the array
    std::unique_ptr<ExpressionNode> indexExpression; // The index being assigned to
    std::unique_ptr<ExpressionNode> valueExpression; // The value being assigned

    ArrayAssignmentNode(
        const std::wstring& name,
        std::unique_ptr<ExpressionNode> indexExpr,
        std::unique_ptr<ExpressionNode> valueExpr)
        : arrayName(name), indexExpression(std::move(indexExpr)), valueExpression(std::move(valueExpr)) {
    }

    llvm::Value* codegen() override {

        llvm::AllocaInst* arrPtr = NamedValues[arrayName].first;

        std::cout << "Inside array assign\n";

        if (!arrPtr) {
            throw std::runtime_error("Undefined array: " + convertWStringToString(arrayName));
        }

        llvm::Value* index = indexExpression->codegen();

        if (!index) {
            throw std::runtime_error("Failed to generate index expression.");
        }

        llvm::Function* CurrentFunction = Builder->GetInsertBlock()->getParent();

        //Calculate the element address
        llvm::Value* elementPtr = Builder->CreateGEP(
            arrPtr->getAllocatedType(),                          // Array type
            arrPtr,                                              // Array pointer
            index,                                               // Index
            "arrayelem"
        );

        //Generate the value
        llvm::Value* val = valueExpression->codegen();

        if (!val) {
            throw std::runtime_error("Failed to generate value expression.");
        }
        NamedValues[arrayName].second = val->getType();
        //Store the value at the calculated address
        auto stored = Builder->CreateStore(val, elementPtr);
        //std::cout << "Array Assign data type: " << val->getType()->getTypeID() << "\n";
        return val;

    }
};

struct ArrayElementNode : ExpressionNode {
    std::wstring arrayName;
    std::unique_ptr<ExpressionNode> index;

    ArrayElementNode(const std::wstring& name, std::unique_ptr<ExpressionNode> idx)
        : arrayName(name), index(std::move(idx)) {
    }

    llvm::Value* codegen() override {
        // Step 1: Retrieve the array pointer
        std::cout << "Inside the Array Element Node\n";
        llvm::AllocaInst* arrayPtr = NamedValues[arrayName].first;
        llvm::Type* valType = NamedValues[arrayName].second;
        if (!arrayPtr) {
            throw std::runtime_error("Undefined array: " + convertWStringToString(arrayName));
        }

        // Step 2: Generate the index
        llvm::Value* indexValue = index->codegen();
        if (!indexValue) {
            throw std::runtime_error("Failed to generate index.");
        }
        if (!indexValue->getType()->isIntegerTy()) {
            throw std::runtime_error("Array index must be an integer.");
        }

        

        //Compute the address of the element
        llvm::Value* elementPtr = Builder->CreateGEP(
            arrayPtr->getAllocatedType(),                          // Array type
            arrayPtr,                                              // Array pointer
            indexValue,                                            // Index
            convertWStringToString(arrayName) + "_element"
        );

        //Load the value at the element address
        llvm::Value* elementValue = Builder->CreateLoad(
            valType,                          // Element type
            elementPtr,                                     // Address of the element
            convertWStringToString(arrayName) + "_load"
        );
        
        //must be integer
        //std::cout <<"Array Data type: "<< elementValue->getType()->getTypeID() << "\n";

        return elementValue;
    }

};

struct WhileNode : StatementNode {
    std::unique_ptr<ExpressionNode> condition;
    std::unique_ptr<BlockNode> body;

    WhileNode(std::unique_ptr<ExpressionNode> cond, std::unique_ptr<BlockNode> b)
        : condition(std::move(cond)), body(std::move(b)) {
    }

    llvm::Value* codegen() override {

        llvm::Function* CurrentFunction = Builder->GetInsertBlock()->getParent();

        // Create labels for the loop:
        llvm::BasicBlock* loopConditionBB = llvm::BasicBlock::Create(*TheContext, "loop_condition", CurrentFunction);
        llvm::BasicBlock* loopBodyBB = llvm::BasicBlock::Create(*TheContext, "loop_body", CurrentFunction);
        llvm::BasicBlock* loopEndBB = llvm::BasicBlock::Create(*TheContext, "loop_end", CurrentFunction);

        // Emit the condition block:
        Builder->CreateBr(loopConditionBB);
        Builder->SetInsertPoint(loopConditionBB);

        // Generate the condition expression:
        llvm::Value* condVal = condition->codegen();
        if (!condVal) {
            throw std::runtime_error("Failed to generate condition expression.");
        }

        // Condition must be a boolean, so cast to i1 if necessary:
        if (condVal->getType() != llvm::Type::getInt1Ty(*TheContext)) {
            condVal = Builder->CreateICmpNE(condVal, llvm::ConstantInt::get(condVal->getType(), 0), "cond_cmp");
        }

        // Branch based on condition:
        Builder->CreateCondBr(condVal, loopBodyBB, loopEndBB);

        // Emit the loop body:
        Builder->SetInsertPoint(loopBodyBB);
        llvm::Value* BodyBlock = body->codegen();  // Generate the loop body

        // After the body, jump back to condition check:
        Builder->CreateBr(loopConditionBB);

        // Final exit block:
        Builder->SetInsertPoint(loopEndBB);

        return BodyBlock;

    }

};


struct InputNode : StatementNode {
    std::wstring name;
    InputNode(std::wstring var_name) : name(var_name) {}

    llvm::Value* codegen() override {
        // Step 1: Declare `scanf` if not already declared
        llvm::Function* ScanfFunc = TheModule->getFunction("scanf");
        if (!ScanfFunc) {
            llvm::FunctionType* ScanfType = llvm::FunctionType::get(
                Builder->getInt32Ty(),                      // Return type (int)
                llvm::PointerType::get(llvm::Type::getInt8Ty(TheModule->getContext()), 0), // Argument type: char* (i8*), 
                true 
            );
            ScanfFunc = llvm::Function::Create(ScanfType, llvm::Function::ExternalLinkage, "scanf", *TheModule);
        }

        // Step 2: Create the format string as a global constant
        llvm::Constant* FormatStr = Builder->CreateGlobalStringPtr("%d", "fmt");

        // Step 3: Allocate memory for the variable
        llvm::AllocaInst* Alloca = NamedValues[name].first;

        

        // Step 4: Call `scanf`
        Builder->CreateCall(ScanfFunc, { FormatStr, Alloca });

        // Step 5: Load the input value
        llvm::Value* InputValue = Builder->CreateLoad(Alloca->getType(), Alloca, "inputval");

        return InputValue;
    }

};

// Simple tokenizer function
std::vector<Token> tokenize(std::wstring& input) {
    std::vector<Token> tokens;
    size_t i = 0;
    
    while (i < input.length()) {
        if (isspace(input[i])) {
            i++;
        }
        else if (isBanglaCharacter(input[i])) {
            //std::cout << "Found one\n";
            std::wstring identifier;
            while (i < input.length() && (isBanglaCharacter(input[i]) || input[i] == U'_')) {
                //std::cout << "ok\n";
                identifier += input[i++];
            }
            
            if (identifier == L"চল") tokens.push_back({ BEGIN, identifier });
            else if (identifier == L"থাম") tokens.push_back({ END, identifier });
            else if (identifier == L"হয়") tokens.push_back({ IF, identifier });
            else if (identifier == L"নয়") tokens.push_back({ ELSE, identifier });
            else if (identifier == L"যখন") tokens.push_back({ WHILE, identifier });
            else if (identifier == L"প্রিন্ট") tokens.push_back({ PRINT, identifier });
            else if (identifier == L"ইনপুট") tokens.push_back({ INPUT, identifier });
            else if (identifier == L"তালিকা") tokens.push_back({ ARRAY, identifier });
            else tokens.push_back({ IDENTIFIER, identifier });
        }
        else if (std::isdigit(input[i])) {
            std::wstring number;
            while (i < input.length() && std::isdigit(input[i])) {
                number += input[i++];
            }
            tokens.push_back({ NUMBER, number });
        }
        else {
            switch (input[i]) {
            case '=':
                if (i + 1 < input.length() && input[i + 1] == '=') {
                    tokens.push_back({ EQUAL, L"==" });
                    i += 2;
                }
                else {
                    tokens.push_back({ ASSIGN, L"=" });
                    i++;
                }
                break;
            case '!':
                if (i + 1 < input.length() && input[i + 1] == '=') {
                    tokens.push_back({ NOT_EQUAL, L"!=" });
                    i += 2;
                }
                break;
            case '<':
                if (i + 1 < input.length() && input[i + 1] == '=') {
                    tokens.push_back({ LESS_EQUAL, L"<=" });
                    i += 2;
                }
                else {
                    tokens.push_back({ LESS, L"<" });
                    i++;
                }
                break;
            case '>':
                if (i + 1 < input.length() && input[i + 1] == '=') {
                    tokens.push_back({ GREATER_EQUAL, L">=" });
                    i += 2;
                }
                else {
                    tokens.push_back({ GREATER, L">" });
                    i++;
                }
                break;
            case '+': tokens.push_back({ PLUS, L"+" }); i++; break;
            case '-': tokens.push_back({ MINUS, L"-" }); i++; break;
            case '*': tokens.push_back({ MULTIPLY, L"*" }); i++; break;
            case '/': tokens.push_back({ DIVIDE, L"/" }); i++; break;
            case '(': tokens.push_back({ OPEN_PAREN, L"(" }); i++; break;
            case ')': tokens.push_back({ CLOSE_PAREN, L")" }); i++; break;
            case '{': tokens.push_back({ OPEN_BRACE, L"{" }); i++; break;
            case '}': tokens.push_back({ CLOSE_BRACE, L"}" }); i++; break;
            case '[': tokens.push_back({ OPEN_BRACKET, L"[" }); i++; break;
            case ']': tokens.push_back({ CLOSE_BRACKET, L"]" }); i++; break;
            case ';': tokens.push_back({ SEMICOLON, L";" }); i++; break;
            default: throw std::runtime_error("Unknown character");
            }
        }
    }
    tokens.push_back({ END_OF_FILE, L"" });
    return tokens;
}

// Recursive descent parser class
class Parser {
public:
    Parser(const std::vector<Token>& tokens) : tokens(tokens), pos(0) {}

    std::unique_ptr<BlockNode> parseProgram() {
        expect(BEGIN);
        auto block = parseStatementList();
        expect(END);
        return block;
    }

private:
    const std::vector<Token>& tokens;
    size_t pos;

    Token currentToken() {
        return tokens[pos];
    }

    Token expect(TokenType type) {
        if (currentToken().type == type) {
            return tokens[pos++];
        }
        else {
            throw std::runtime_error("Unexpected token: "+pos);
        }
    }

    std::unique_ptr<BlockNode> parseStatementList() {
        auto block = std::make_unique<BlockNode>();
        while (currentToken().type != END && currentToken().type != CLOSE_BRACE) {
            block->statements.push_back(parseStatement());
        }
        return block;
    }

    std::unique_ptr<StatementNode> parseStatement() {
        switch (currentToken().type) {
        case IDENTIFIER:
            return parseAssignmentOrArrayAssignment();
            break;
        case IF:
            return parseIfStatement();
            break;
        case WHILE:
            return parseWhileStatement();
            break;
        case PRINT:
            return parsePrintStatement();
            break;
        case INPUT:
            return parseInputStatement();
            break;
        case ARRAY:
            return parseArrayDeclaration();
            break;
        default:
            throw std::runtime_error("Unexpected statement");
        }
    }

    std::unique_ptr<StatementNode> parseAssignmentOrArrayAssignment() {
        //std::cout << "inside parsing\n";
        std::wstring variable = expect(IDENTIFIER).value;
        if (currentToken().type == OPEN_BRACKET) {
            // Array element assignment
            expect(OPEN_BRACKET);
            auto indexExpr = parseExpression(); // Parse the index expression
            expect(CLOSE_BRACKET);
            expect(ASSIGN); // Parse the '=' symbol
            auto valueExpr = parseExpression(); // Parse the value to assign
            expect(SEMICOLON);

            // Create and return an ArrayAssignmentNode
            return std::make_unique<ArrayAssignmentNode>(variable, std::move(indexExpr), std::move(valueExpr));
        }
        else {
            // Regular assignment
            expect(ASSIGN);
            auto value = parseExpression();
            expect(SEMICOLON);
            return std::make_unique<AssignmentNode>(variable, std::move(value));
        }
    }

    std::unique_ptr<ArrayDeclareNode> parseArrayDeclaration() {
        expect(ARRAY);
        std::wstring name = expect(IDENTIFIER).value;
        expect(OPEN_BRACKET);
        auto size = parseExpression();
        expect(CLOSE_BRACKET);
        expect(SEMICOLON);

        return std::make_unique<ArrayDeclareNode>(name, std::move(size));
    }

    std::unique_ptr<IfNode> parseIfStatement() {
        expect(IF);
        expect(OPEN_PAREN);
        auto condition = parseCondition();
        expect(CLOSE_PAREN);
        expect(OPEN_BRACE);
        auto block = parseStatementList();
        expect(CLOSE_BRACE);
        expect(ELSE);
        expect(OPEN_BRACE);
        auto elseBlock = parseStatementList();
        expect(CLOSE_BRACE);

        return std::make_unique<IfNode>(std::move(condition), std::move(block), std::move(elseBlock));
    }

    std::unique_ptr<WhileNode> parseWhileStatement() {
        expect(WHILE);
        expect(OPEN_PAREN);
        auto condition = parseCondition();
        expect(CLOSE_PAREN);
        expect(OPEN_BRACE);
        auto block = parseStatementList();
        expect(CLOSE_BRACE);

        return std::make_unique<WhileNode>(std::move(condition), std::move(block));
    }

    std::unique_ptr<PrintNode> parsePrintStatement() {
        expect(PRINT);
        expect(OPEN_PAREN);
        auto expr = parseExpression();
        expect(CLOSE_PAREN);
        expect(SEMICOLON);

        return std::make_unique<PrintNode>(std::move(expr));
    }

    std::unique_ptr<InputNode> parseInputStatement() {
        expect(INPUT);
        expect(OPEN_PAREN);
        std::wstring variable = expect(IDENTIFIER).value;
        expect(CLOSE_PAREN);
        expect(SEMICOLON);

        return std::make_unique<InputNode>(variable);
    }

    std::unique_ptr<ExpressionNode> parseExpression() {
        auto left = parseTerm();
        while (currentToken().type == PLUS || currentToken().type == MINUS) {
            std::wstring op = currentToken().value;
            pos++;
            auto right = parseTerm();
            left = std::make_unique<BinaryNode>(op, std::move(left), std::move(right));
        }
        return left;
    }

    std::unique_ptr<ExpressionNode> parseTerm() {
        auto left = parseFactor();
        while (currentToken().type == MULTIPLY || currentToken().type == DIVIDE) {
            std::wstring op = currentToken().value;
            pos++;
            auto right = parseFactor();
            left = std::make_unique<BinaryNode>(op, std::move(left), std::move(right));
        }
        return left;
    }

    std::unique_ptr<ExpressionNode> parseFactor() {
        if (currentToken().type == NUMBER) {
            int value = std::stoi(expect(NUMBER).value);
            return std::make_unique<NumberNode>(value);
        }
        else if (currentToken().type == IDENTIFIER) {
            std::wstring name = expect(IDENTIFIER).value;

            if (currentToken().type == OPEN_BRACKET) {
                // Array element
                expect(OPEN_BRACKET);
                auto index = parseExpression();
                expect(CLOSE_BRACKET);
                return std::make_unique<ArrayElementNode>(name, std::move(index));
            }
            else {
                // Simple variable
                return std::make_unique<VariableNode>(name);
            }
        }
        else if (currentToken().type == OPEN_PAREN) {
            expect(OPEN_PAREN);
            auto expr = parseExpression();
            expect(CLOSE_PAREN);
            return expr; // Parenthesized expression
        }
        else {
            throw std::runtime_error("Unexpected factor: "+pos);
        }
    }


    std::unique_ptr<ConditionNode> parseCondition() {
        auto left = parseExpression();
        if (currentToken().type == EQUAL || currentToken().type == NOT_EQUAL ||
            currentToken().type == LESS || currentToken().type == GREATER ||
            currentToken().type == LESS_EQUAL || currentToken().type == GREATER_EQUAL) {
            std::wstring op = currentToken().value;
            pos++;
            auto right = parseExpression();

            return std::make_unique<ConditionNode>(op, std::move(left), std::move(right));
        }
        else {
            throw std::runtime_error("Unexpected condition");
        }
    }
};

int main() {



    std::wifstream file("C:\\Users\\User\\Desktop\\program.txt"); // Open the file
    file.imbue(std::locale("en_US.UTF-8"));
    if (!file.is_open()) {
        std::cerr << "Failed to open the file.\n";
        return 1;
    }


    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    llvm::InitializeNativeTargetAsmParser();

    TheContext = std::make_unique<llvm::LLVMContext>();
    TheModule = std::make_unique<llvm::Module>("MyModule", *TheContext);
    Builder = std::make_unique<llvm::IRBuilder<>>(*TheContext);

    // Create a simple function
    llvm::FunctionType* FT = llvm::FunctionType::get(Builder->getVoidTy(), false);
    llvm::Function* MainFunction = llvm::Function::Create(
        FT, llvm::Function::ExternalLinkage, "main", *TheModule);

    // Create a basic block and set the builder's insertion point
    llvm::BasicBlock* EntryBB = llvm::BasicBlock::Create(*TheContext, "entry", MainFunction);
    Builder->SetInsertPoint(EntryBB);


    //std::cout << "Enter your program (end with 'end'):\n";
    std::wstring line, input;

    while (std::getline(file, line)) {
        input += line + L"\n";
        if (line == L"থাম") break;
    }

    

    // Tokenize the input
    auto tokens = tokenize(input);

    // Debug: Print tokens
    //for (auto e : tokens) {
        //std::wcout << e.type << " " << e.value << std::endl;
    //}


    // Parse tokens into AST
    Parser parser(tokens);
    std::unique_ptr<BlockNode> ast;
    try {
        ast = parser.parseProgram();
    }
    catch (const std::exception& e) {
        std::cerr << "Parsing error: " << e.what() << "\n";
        return 0;
    }

    // Generate LLVM IR
    llvm::Value* programIR = nullptr;
    try {
        programIR = ast->codegen();

        Builder->CreateRetVoid();

        // Debug: Print variable names
        //std::cout << "Variables: \n";
        //for (auto e : NamedValues) {
            //std::cout << convertWStringToString(e.first) << std::endl;
        //}
        //TheModule->print(llvm::errs(), nullptr); // Print LLVM IR to stderr
        std::ofstream outFile("output.ll");
        llvm::raw_os_ostream outStream(outFile);
        //TheModule->print(outStream, nullptr);    // Save LLVM IR to a file
        //outFile.close();

    }
    catch (const std::exception& e) {
        std::cerr << "Code generation error: " << e.what() << "\n";
        return 0;
    }
    //TheModule = nullptr;
    //std::cout << TheModule << "\n";
    std::cout << "LLVM IR generated and saved to 'output.ll'.\n";

    
    // === JIT Execution ===
    try {
        // Create the JIT
        auto JIT = llvm::orc::LLJITBuilder().create();
        if (!JIT) {
            llvm::errs() << "Error creating JIT: " << llvm::toString(JIT.takeError()) << "\n";
            return 1;
        }
        // Add the generated module to the JIT
        auto TSM = llvm::orc::ThreadSafeModule(std::move(TheModule), std::move(TheContext));

        
        auto Err = JIT->get()->addIRModule(std::move(TSM));
        
        if (Err) {
            llvm::errs() << "Error adding module to JIT: " << llvm::toString(std::move(Err)) << "\n";
            return 1;
        }
        else {
            llvm::outs() << "Module successfully added to JIT.\n";
            //TheModule->print(llvm::outs(), nullptr);
        }
        
        // Look up the "main" function
        auto MainSym = JIT->get()->lookup("main");
        
        if (!MainSym) {
            llvm::errs() << "Error finding 'main' in JIT: " << llvm::toString(MainSym.takeError()) << "\n";
            return 1;
        }
        

        auto MainAddr = MainSym->getValue();
        if (!MainAddr) {
            llvm::errs() << "Error: main function address is null.\n";
            return 1;
        }

        // Cast the address to a callable function pointer
        auto* MainFunc = (void (*)())MainAddr;
        std::cout << "Executing JIT-compiled 'main' function:\n";
        MainFunc(); // Call the JIT-compiled function
        

    }
    catch (const std::exception& e) {
        std::cerr << "JIT execution error: " << e.what() << "\n";
        return 1;
    }
    
    
    return 0;



}


