/*
 * comprehensive_demo_part2.c
 *
 * A comprehensive demonstration of additional important C programming concepts:
 * - Preprocessor directives and macros
 * - Enumerations (enum)
 * - Unions
 * - Bit fields in structures
 * - Singly-linked list (dynamic data structure)
 * - File I/O operations (reading and writing files)
 * - Use of const and volatile qualifiers
 * - The assert macro for debugging
 * - Timing functions from time.h
 * - Inline functions
 * - Command-line arguments
 *
 * Compile with:
 *   gcc -std=c99 -Wall -o demo2 comprehensive_demo_part2.c
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <assert.h>
#include <time.h>

/* Preprocessor macros and global flags */
#define MAX_BUFFER 256
#define DEBUG_PRINT(...) do { if(debug_mode) { printf(__VA_ARGS__); } } while(0)
bool debug_mode = true;

/* --- Enumerations --- */
/* An enum representing days of the week */
typedef enum {
    SUNDAY,
    MONDAY,
    TUESDAY,
    WEDNESDAY,
    THURSDAY,
    FRIDAY,
    SATURDAY
} DayOfWeek;

/* --- Unions --- */
/* A union that can store either an int, a float, or a string */
typedef union {
    int i;
    float f;
    char str[20];
} DataUnion;

/* --- Bit Fields --- */
/* A structure with bit fields representing flags */
typedef struct {
    unsigned int is_valid : 1;
    unsigned int error_code : 3;
    unsigned int reserved : 4;
} BitFields;

/* --- Linked List Definitions --- */
typedef struct Node {
    int data;
    struct Node *next;
} Node;

/* Linked list function prototypes */
Node* createNode(int data);
void insertAtHead(Node **head, int data);
void printList(Node *head);
void freeList(Node *head);
Node* searchList(Node *head, int target);
void deleteNode(Node **head, int target);
Node* reverseList(Node *head);

/* --- Inline Function --- */
/* Changed to static inline to avoid linker issues */
static inline int square(int x) {
    return x * x;
}

/* --- File I/O Demonstration --- */
void demoFileIO(void) {
    FILE *fp = fopen("demo_output.txt", "w");
    if (!fp) {
        perror("Failed to open file for writing");
        return;
    }
    fprintf(fp, "This is a test file.\nLine 2 of the file.\n");
    fclose(fp);

    // Read and display the file contents
    fp = fopen("demo_output.txt", "r");
    if (!fp) {
        perror("Failed to open file for reading");
        return;
    }
    char buffer[MAX_BUFFER];
    printf("Demo: File I/O\nFile contents:\n");
    while (fgets(buffer, sizeof(buffer), fp)) {
        printf("%s", buffer);
    }
    fclose(fp);
    printf("\n");
}

/* --- Const and Volatile Demonstration --- */
void demoConstVolatile(void) {
    const int constant = 100;
    volatile int changing = 0;
    printf("Demo: const and volatile\n");
    printf("Constant value: %d\n", constant);
    changing = 50;  // Simulate a change (in real scenarios, hardware might modify volatile variables)
    printf("Volatile value: %d\n\n", changing);
}

/* --- Assert Demonstration --- */
void demoAssert(void) {
    printf("Demo: Assert Macro\n");
    int a = 5, b = 5;
    assert(a == b);  // This should pass.
    printf("Assert passed: a equals b\n\n");
}

/* --- Timing Demonstration --- */
void demoTiming(void) {
    printf("Demo: Timing with time.h\n");
    clock_t start = clock();
    volatile long sum = 0;  // volatile to prevent compiler optimizing the loop away
    for (long i = 0; i < 10000000; i++) {
        sum += i;
    }
    clock_t end = clock();
    double elapsed_secs = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Time taken to sum 10,000,000 integers: %.3f seconds\n\n", elapsed_secs);
}

/* --- Linked List Functions --- */
Node* createNode(int data) {
    Node *newNode = (Node*)malloc(sizeof(Node));
    if (!newNode) {
        perror("Unable to allocate memory for node");
        exit(EXIT_FAILURE);
    }
    newNode->data = data;
    newNode->next = NULL;
    return newNode;
}

void insertAtHead(Node **head, int data) {
    Node *newNode = createNode(data);
    newNode->next = *head;
    *head = newNode;
}

void printList(Node *head) {
    printf("Linked List: ");
    while (head) {
        printf("%d -> ", head->data);
        head = head->next;
    }
    printf("NULL\n");
}

void freeList(Node *head) {
    Node *temp;
    while (head) {
        temp = head;
        head = head->next;
        free(temp);
    }
}

Node* searchList(Node *head, int target) {
    while (head) {
        if (head->data == target)
            return head;
        head = head->next;
    }
    return NULL;
}

void deleteNode(Node **head, int target) {
    Node *temp = *head, *prev = NULL;
    while (temp) {
        if (temp->data == target) {
            if (!prev)
                *head = temp->next;
            else
                prev->next = temp->next;
            free(temp);
            return;
        }
        prev = temp;
        temp = temp->next;
    }
}

Node* reverseList(Node *head) {
    Node *prev = NULL, *current = head, *next = NULL;
    while (current) {
        next = current->next;
        current->next = prev;
        prev = current;
        current = next;
    }
    return prev;
}

/* --- Main Function --- */
int main(int argc, char *argv[]) {
    printf("=== Comprehensive C Program Demonstration: Part 2 ===\n\n");

    /* Preprocessor Macros */
    printf("Demo: Preprocessor Macros\n");
    #ifdef DEBUG
    printf("DEBUG mode is enabled.\n");
    #else
    printf("DEBUG mode is not enabled.\n");
    #endif
    printf("\n");

    /* Enumerations */
    printf("Demo: Enumerations\n");
    DayOfWeek today = WEDNESDAY;
    printf("Today is day number %d of the week (0=Sunday, 6=Saturday)\n\n", today);

    /* Unions */
    printf("Demo: Unions\n");
    DataUnion data;
    data.i = 42;
    printf("DataUnion as int: %d\n", data.i);
    data.f = 3.1415f;
    printf("DataUnion as float: %.4f\n", data.f);
    strcpy(data.str, "Hello, World!");
    printf("DataUnion as string: %s\n\n", data.str);

    /* Bit Fields */
    printf("Demo: Bit Fields in Structures\n");
    BitFields bf = {1, 5, 0};
    printf("BitFields: is_valid=%u, error_code=%u, reserved=%u\n\n",
           bf.is_valid, bf.error_code, bf.reserved);

    /* Linked List Operations */
    printf("Demo: Linked List Operations\n");
    Node *list = NULL;
    insertAtHead(&list, 10);
    insertAtHead(&list, 20);
    insertAtHead(&list, 30);
    printList(list);
    Node *found = searchList(list, 20);
    if (found)
        printf("Found node with data %d\n", found->data);
    deleteNode(&list, 20);
    printf("After deleting 20:\n");
    printList(list);
    list = reverseList(list);
    printf("After reversing the list:\n");
    printList(list);
    freeList(list);
    printf("\n");

    /* File I/O */
    demoFileIO();

    /* Const and Volatile */
    demoConstVolatile();

    /* Assert Macro */
    demoAssert();

    /* Timing */
    demoTiming();

    /* Inline Function */
    printf("Demo: Inline Function (square)\n");
    int num = 7;
    printf("Square of %d is %d\n\n", num, square(num));

    /* Command-line Arguments */
    printf("Demo: Command-line Arguments\n");
    if (argc > 1) {
        printf("Program arguments (%d):\n", argc);
        for (int i = 0; i < argc; i++) {
            printf("  argv[%d]: %s\n", i, argv[i]);
        }
    } else {
        printf("No command-line arguments provided.\n");
    }
    printf("\n");

    printf("=== End of Comprehensive Part 2 Demonstration ===\n");
    return 0;
}
