## Introduction to Computer Programming Project

### Problem Statement

In current time, every word processor is equipped with high-end spell checking algorithm, integrated into it. In this project, the task would be to develop a simple tool to implement spell checking and suggestion functionality. The application should take a file provided by the user as an input and generate a list of incorrect/misspelled words along with suggestions for correcting each such word. One can use Linux word database to build the essential dictionary. The students may need to implement the concepts of pointers, file I/O, linked lists, and/or any other data structures in order to complete this project.

### Implementation Details

* I have used **TRIE** data structure to solve this problem. [This](https://www.youtube.com/watch?v=AXjmTQ8LEoI) helped me to understand TRIE data structure.
* I have used **C** language to implement this.

### Usage
```bash
foo@bar:~$ gcc main.c -o main
foo@bar:~$ ./main
```
Enter string in input, it will show a list of words as suggestions.

---
#### Note
* Also words to build the vocabulary is taken from ubuntu's english dictionary. Which I have copied here as *words.txt*.
* I have avoided words which have special characters (other than *ASCII* like some french words) to build vocabulary. This is done in program. They do are present in *word.txt* file. 
---