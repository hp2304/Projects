#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

typedef struct trieNode trieNode;

typedef struct  mapElement 
{
    char letter;
    trieNode *next;
}pair;

typedef struct trieNode{
    pair *map;
    unsigned char map_size;
    bool endOfWord; 
}trieNode;

trieNode defaultNode = {NULL, 0, false};

trieNode *createDictionaryTree();
bool validateWord(char *);
void addNodetoTree(trieNode*, char*);
void printTrie(trieNode *);
void suggest(trieNode *, char *);
void findWords(trieNode *, char *);

int main(){
    trieNode *root = createDictionaryTree();
    char inp[20];
    while(true){
        printf("\nEnter the string you want to search (0 to exit): ");
        scanf("%s", inp);
        if(!strcmp(inp, "0"))
            break;
        else
            suggest(root, inp);
    }
    return 0;
}

void suggest(trieNode *root, char *query){
    trieNode *curr = root;
    char prefix[20] = "", letter;
    bool found;
    while(*query){
        found = false;
        letter = *query;
        for(int i=0; i<(curr->map_size); ++i){
            if(letter == curr->map[i].letter){
                strncat(prefix, &letter, 1);
                found = true;
                curr = curr->map[i].next;
                break;
            }
        }
        if(!found)
            break;
        query++;
    }
    findWords(curr, prefix);
}

void findWords(trieNode *curr, char *prefix){
    if(curr->endOfWord == true)
        printf("%s\n", prefix);
    for(int i=0; i<curr->map_size; ++i){
        char temp[20];
        strcpy(temp, prefix);
        strncat(temp, &curr->map[i].letter, 1);
        findWords(curr->map[i].next, temp);
    }
}

void printTrie(trieNode *node){
    printf("\n");
    for(int i=0; i<node->map_size; ++i)
        printf("%c ", node->map[i].letter);
    printf("\nMap size: %d, endOfWord: %d\n", node->map_size, node->endOfWord);
    for(int i=0; i<node->map_size; ++i)
        printTrie(node->map[i].next);
}

trieNode *createDictionaryTree(){
    FILE *fp;
    char *word = NULL;
    size_t len = 0;
    ssize_t read;

    trieNode *root = (trieNode *)malloc(sizeof(trieNode));
    *root = defaultNode;

    fp = fopen("/usr/share/dict/american-english", "r");
    if (fp == NULL)
        exit(EXIT_FAILURE);

    int cnt=0;
    while((read = getline(&word, &len, fp)) != -1){
        word[strlen(word)-1] = '\0';
        if(validateWord(word))
            addNodetoTree(root, word);
    }
    fclose(fp);
    if(word)
        free(word);
    return root;
}

void addNodetoTree(trieNode *curr, char *word){
    trieNode *root = curr;
    while(*word){
        bool isElement = false;
        char letter = *word;
        for(int i=0; i<(curr->map_size); ++i){
            if(letter == curr->map[i].letter){
                isElement = true;
                curr = curr->map[i].next;
                break;
            }
        }
        if(!isElement){
            trieNode *new_trie_node = (trieNode *)malloc(sizeof(trieNode));
            *new_trie_node = defaultNode;
            curr->map_size = curr->map_size + 1;
            curr->map  = realloc(curr->map, sizeof(pair)*(curr->map_size));
            curr->map[(curr->map_size)-1].letter = letter;
            curr->map[(curr->map_size)-1].next = new_trie_node;
            curr = new_trie_node;
        }
        word++;
    }
    curr->endOfWord = true;
}

bool validateWord(char *word){
    while(*word){
        int ascii_val = (int)*word;
        if(!(ascii_val >= 0 && ascii_val <= 255))
            return false;
        word++;
    }
    return true;
} 