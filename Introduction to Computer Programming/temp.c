#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct  word
{
    int cnt;
    float exact;
}mark;


typedef struct something
{
    int a;
    int b;
    mark *names;
}some;

some def = {0,0, NULL};

void func(char []);
void func2(char []);

int main(){
    char str[20] = "asdasd";
    func(str);
    /*some *ptr;
    int len = 1;
    ptr = (some*) malloc(len*sizeof(some));
    ptr[0].a = 5;
    ptr[0].b = 6;

    for(int i=0; i<len; ++i){
        printf("a: %d,  b: %d\n", ptr[i].a, ptr[i].b);
    }
    len++;
    ptr = (some*)realloc(ptr, len);
    ptr[1].a = 15;
    ptr[1].b = 16;

    printf("\n");
    for(int i=0; i<len; ++i){
        printf("a: %d,  b: %d\n", ptr[i].a, ptr[i].b);
    }
    */
    return 0;
}

void func(char str[20]){
    printf("%s\n", str);
    func2(str);
    printf("%s\n", str);
}

void func2(char str[20]){
    char letter = 'z';
    strncat(str, &letter, 1);
}