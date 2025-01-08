//#include <iostream>
//#include <locale>
//#include <windows.h>
//#include<string>
//
//bool isBanglaCharacter(wchar_t ch) {
//    // Check if the character is within the Bengali Unicode range
//    return (ch >= 0x0980 && ch <= 0x09FF);
//}
//
//int main() {
//    // Set console to UTF-8
//    SetConsoleOutputCP(CP_UTF8);
//    SetConsoleCP(CP_UTF8);
//
//    // Set global locale to the user's locale
//    std::locale::global(std::locale(""));
//
//    // Configure wide-character streams
//    std::wcin.imbue(std::locale());
//    std::wcout.imbue(std::locale());
//
//    std::wstring banglaText=L"";
//    std::wstring inputP = L"hello বাংলা লেখা লিখুন: ";
//    for (int i = 0; i < inputP.size(); i++) {
//        if (isspace(inputP[i])) {
//            std::cout << "space\n";
//            if (banglaText == L"বাংলা") {
//                std::cout << "Milche\n";
//            }
//            banglaText = L"";
//        }
//        else if (isBanglaCharacter(inputP[i])) {
//            std::cout << "ache\n";
//            banglaText += inputP[i];
//        }
//        else {
//            std::cout << "Nai\n";
//        }
//    }
//    /*std::cout<<inputP.size()<<std::endl;
//    std::wcout << inputP;
//    std::getline(std::wcin, banglaText);
//
//    std::wcout << L"আপনার লেখা: " << banglaText << std::endl;*/
//    return 0;
//}
